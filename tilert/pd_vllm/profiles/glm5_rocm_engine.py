"""ROCm (MI350X / MI355X) engine adapter for the GLM-5.2 / GLM-5.3 PD plane.

The ROCm tilert build serves GLM-5.2 (and GLM-5.3, same base model and
config) through ``tilert.models.glm_5``: ``Glm52Generator`` owns one
``Glm52ShowHands`` -- a single-process TP8 e2e whose API is
``prefill / step / decode_n / mtp_n / seed_draft / accepted_tokens /
set_cur_pos / reset_sequence / update_sampling``. Unlike the CUDA engine it
has no ``inject_cache``; its caches are plain per-rank tensors:

  rank 0     : [kv, pe] x n_layers (pure-MLA TP8 mode only; the full-layer
               pairs are never read) followed by one ki cache per FULL
               indexer layer (+1 for the MTP block when num_mtp > 0)
  ranks 1..7 : [kv, pe] x (n_layers + 1)  -- the last pair is the MTP block's

  kv : [B=1, max_seq_len, 512] bf16      pe : [B=1, max_seq_len, 64] bf16
  ki : [B=1, max_seq_len, 128] bf16, or with TILERT_GLM5_FP8_KI=1 one flat u8
       plane [B*L*128 fp8 e4m3 | B*L f32 per-token scales]

``cur_pos`` is the NEXT write row: after a prompt of P tokens the engine sits
at cur_pos == P, so after injecting P rows we ``set_cur_pos(P)`` and feed the
prefill-sampled first token exactly where the engine's own generate() would
be after its prompt prefill. The decode loop below is that generate() tail
(MTP: seed_draft + chained mtp_n; plain: one forced step + decode_n), reading
the emitted stream back from the AR flat buffer.

GPU-VERIFY (not checkable on the build machine): the ki fp8 plane layout and
e4m3 flavour, that reset_sequence leaves the caches intact, and that the MTP
block tolerates a cold last_hidden on the first verify step (the CUDA adapter
has the same warm-up situation and it is benign there).
"""

from __future__ import annotations

import logging
import os

import torch

from tilert.pd_vllm.grammar_spec import GrammarBackendUnavailable
from tilert.pd_vllm.sampling import resolve_top_p

logger = logging.getLogger("pd_vllm.profile.glm5_rocm")

_FP8_MAX = 448.0  # OCP e4m3fn range (CDNA4 fp8)
_INDEX_HEAD_DIM = 128


def is_rocm_torch() -> bool:
    """True when the installed torch is a ROCm build (HIP behind torch.cuda)."""
    return getattr(torch.version, "hip", None) is not None


def build_rocm_engine(model_weights_dir, max_seq_len, with_mtp, ar_steps, num_mtp=3):
    """Load the ROCm GLM-5.2/5.3 engine and wrap it as a PDEngine."""
    # Shipped by the ROCm tilert build; absent from the CUDA package tree.
    from tilert.models.glm_5.generator import Glm52Generator  # type: ignore[attr-defined]
    from tilert.models.glm_5.model_args import ModelArgsGlm52  # type: ignore[attr-defined]

    if with_mtp and num_mtp not in (1, 3):
        raise ValueError(f"the ROCm GLM-5.2 engine builds MTP at depth 1 or 3, " f"not {num_mtp}")
    gen = Glm52Generator(
        model_weights_dir=model_weights_dir,
        model_args=ModelArgsGlm52(),
        max_new_tokens=max(max_seq_len - 256, 4096 - 256),
        use_topp=True,
        num_mtp=num_mtp if with_mtp else 0,
        max_seq_len=max_seq_len,
    )
    gen.from_pretrained()
    return RocmGlm52EngineAdapter(gen, with_mtp, ar_steps=ar_steps)


class RocmGlm52EngineAdapter:
    """PDEngine over the ROCm ``Glm52Generator`` (see module docstring)."""

    def __init__(
        self,
        generator,
        with_mtp: bool,
        ar_steps: int = 8,
        *,
        pure_tp8: bool | None = None,
        fp8_ki: bool | None = None,
    ):
        self.gen = generator
        self.dl = generator.decode_layer
        self.with_mtp = bool(with_mtp) and self.dl.num_mtp > 0
        if with_mtp and not self.with_mtp:
            raise ValueError("--with-mtp requested but the engine was built " "with num_mtp=0")
        self.max_seq_len = int(self.dl.args.max_seq_len)
        self.n_layers = int(self.dl.n_layers)
        self.npes = int(self.dl.npes)
        self.mtp_seq_len = self.dl.num_mtp + 1
        self.ar_steps = max(1, min(1024, int(os.environ.get("GLM5_AR_N", str(ar_steps)))))
        self.stop_ids = {int(t) for t in generator.stop_token_ids}
        self.last_stats: dict = {}
        self._ignore_eos = False
        self._seq_len = 0
        # Cache geometry knobs. Read from the engine's own helpers so the
        # adapter follows whatever the C++ side was configured with; the
        # keyword overrides exist for the CPU tests.
        from tilert.models.glm_5.model_args import (  # type: ignore[attr-defined]
            full_layer_ordinals,
        )
        from tilert.models.glm_5.weight_converter import (  # type: ignore[import-not-found]
            fp8_ki_enabled,
            pure_tp8_enabled,
        )

        self._full_layers = list(full_layer_ordinals(self.n_layers))
        self._pure_tp8 = pure_tp8_enabled() if pure_tp8 is None else pure_tp8
        self._fp8_ki = fp8_ki_enabled() if fp8_ki is None else fp8_ki

    # ── capabilities ────────────────────────────────────────────────────
    def supports_logprobs(self) -> bool:
        return False

    def supports_penalties(self) -> bool:
        return False

    def supports_ignore_eos(self) -> bool:
        return True

    def prepare_grammar(self, grammar_spec, enable_thinking=True):
        if grammar_spec is None:
            return
        raise GrammarBackendUnavailable(
            "constrained decoding is not available on the ROCm GLM engine "
            "(no grammar bitmask path in this build)"
        )

    # ── inject ──────────────────────────────────────────────────────────
    def _write_ki(self, slot: torch.Tensor, ki: torch.Tensor, seq: int) -> None:
        """Write ``ki`` [seq,128] bf16 into one rank-0 ki cache slot.

        ``ki`` is already Hadamard-rotated by the profile's convert.
        """
        if not self._fp8_ki:
            slot[0, :seq].copy_(ki.to(slot.device, non_blocking=True))
            return
        # Flat u8 plane: [L*128 fp8 bytes | L f32 scales] (B == 1).
        L = self.max_seq_len
        nbytes = L * _INDEX_HEAD_DIM
        x = ki.to(slot.device).float()
        amax = x.abs().amax(dim=-1).clamp_(min=1e-12)
        scale = amax / _FP8_MAX  # [seq] f32
        q = (x / scale.unsqueeze(-1)).clamp_(-_FP8_MAX, _FP8_MAX)
        q = q.to(torch.float8_e4m3fn)
        slot[:nbytes].view(torch.float8_e4m3fn).view(L, _INDEX_HEAD_DIM)[:seq].copy_(q)
        slot[nbytes : nbytes + L * 4].view(torch.float32)[:seq].copy_(scale)

    def inject(self, req) -> None:
        dl = self.dl
        layers = req.layers
        seq = int(req.seq_len)
        n_extra = 1 if dl.num_mtp > 0 else 0
        if len(layers) not in (self.n_layers, self.n_layers + 1):
            raise RuntimeError(
                f"glm5_rocm inject: got {len(layers)} layers, engine has "
                f"{self.n_layers} (+1 MTP block)"
            )
        if seq <= 0 or seq > self.max_seq_len:
            raise RuntimeError(
                f"glm5_rocm inject: seq_len {seq} outside " f"(0, {self.max_seq_len}]"
            )
        if n_extra and len(layers) == self.n_layers:
            raise RuntimeError(
                "glm5_rocm inject: engine has an MTP block but "
                "the prefill sent no MTP-layer KV (prefill must "
                "run with --speculative-config mtp)"
            )
        # The profile always ships n_layers + 1 (the MTP tail); an engine
        # built without MTP has no slot for it.
        use = layers[: self.n_layers + n_extra]

        dl.reset_sequence()  # AR buffers + cur_pos; caches stay
        for rank in range(self.npes):
            if rank == 0 and not self._pure_tp8:
                continue  # TP7 arm: rank 0 holds no kv/pe
            caches = dl._caches[rank]
            n_pairs = self.n_layers + (n_extra if rank != 0 else 0)
            for lid in range(min(len(use), n_pairs)):
                _ki, kv, pe = use[lid]
                caches[2 * lid][0, :seq].copy_(kv, non_blocking=True)
                caches[2 * lid + 1][0, :seq].copy_(pe, non_blocking=True)
        caches0 = dl._caches[0]
        ki_base = 2 * self.n_layers if self._pure_tp8 else 0
        ki_layers = self._full_layers + ([self.n_layers] if n_extra else [])
        for ki_slot, lid in enumerate(ki_layers):
            if lid >= len(use):
                break
            self._write_ki(caches0[ki_base + ki_slot], use[lid][0], seq)
        torch.cuda.synchronize()
        dl.set_cur_pos(seq)
        self._seq_len = seq

    # ── decode ──────────────────────────────────────────────────────────
    def decode(
        self,
        first_token_id,
        max_tokens,
        sampling,
        on_token=None,
        cancel_event=None,
        grammar_session=None,
        top_logprobs=None,
    ):
        if grammar_session is not None:
            raise GrammarBackendUnavailable(
                "constrained decoding is not available on the ROCm GLM engine"
            )
        if top_logprobs:
            raise NotImplementedError("logprobs are not available on the ROCm GLM engine")
        sampling = sampling or {}
        rep = float(sampling.get("repetition_penalty", 1.0) or 1.0)
        presence = float(sampling.get("presence_penalty", 0.0) or 0.0)
        if rep != 1.0 or presence != 0.0:
            raise NotImplementedError(
                "repetition/presence penalties are not supported by this " "model's decode runtime"
            )
        temp = float(sampling.get("temperature", 1.0))
        if temp < 1e-5:
            self.dl.update_sampling(False, 1.0, 1.0)  # greedy arm
        else:
            self.dl.update_sampling(True, temp, resolve_top_p(sampling))
        self._ignore_eos = bool(sampling.get("ignore_eos"))
        first = int(first_token_id)
        budget = min(int(max_tokens), self.max_seq_len - self._seq_len - 1)
        if budget <= 0:
            self.last_stats = {"finish_reason": "length"}
            return [first]
        stop_ids = set() if self._ignore_eos else self.stop_ids
        if first in stop_ids:
            self.last_stats = {"finish_reason": "stop"}
            return []
        tokens = [first]
        if on_token:
            on_token(first)
        if self.with_mtp:
            finish = self._decode_mtp(first, budget, tokens, stop_ids, on_token, cancel_event)
        else:
            finish = self._decode_plain(first, budget, tokens, stop_ids, on_token, cancel_event)
        self.last_stats = {"finish_reason": finish}
        return tokens

    def _emit(self, new, tokens, budget, stop_ids, on_token):
        """Append ``new`` to ``tokens`` honouring stop / budget.

        Returns 'stop' | 'length' | None (keep going).
        """
        for tok in new:
            tok = int(tok)
            if tok in stop_ids:
                return "stop"
            if len(tokens) >= budget:
                return "length"
            tokens.append(tok)
            if on_token:
                on_token(tok)
        return "length" if len(tokens) >= budget else None

    def _decode_mtp(self, first, budget, tokens, stop_ids, on_token, cancel_event):
        dl = self.dl
        pos_limit = self.max_seq_len
        mtp_seq = self.mtp_seq_len
        chain_slack = max(0, dl.num_mtp - 1)
        # First draft is unknown: seed it with the token itself; a wrong draft
        # is simply rejected (one accepted token that step).
        dl.seed_draft(first, first)
        base = dl.accepted_count
        produced = 0
        while True:
            if cancel_event is not None and cancel_event.is_set():
                return "cancelled"
            room = pos_limit - (self._seq_len + produced)
            k = min(self.ar_steps, (room - chain_slack) // mtp_seq)
            if k < 1:
                return "length"
            got = int(dl.mtp_n(k))
            new = dl.accepted_tokens(base + produced)
            produced += got
            verdict = self._emit(new, tokens, budget, stop_ids, on_token)
            if verdict:
                return verdict

    def _decode_plain(self, first, budget, tokens, stop_ids, on_token, cancel_event):
        dl = self.dl
        pos_limit = self.max_seq_len
        base = dl.accepted_count
        dl.step(first)  # row seq_len <- first, samples t1
        produced = 0
        while True:
            new = dl.accepted_tokens(base + produced)
            produced += len(new)
            verdict = self._emit(new, tokens, budget, stop_ids, on_token)
            if verdict:
                return verdict
            if cancel_event is not None and cancel_event.is_set():
                return "cancelled"
            room = pos_limit - (self._seq_len + 1 + produced)
            if room < 1:
                return "length"
            n = min(8, budget - len(tokens), room)
            dl.decode_n(n)

    def reset(self) -> None:
        pass
