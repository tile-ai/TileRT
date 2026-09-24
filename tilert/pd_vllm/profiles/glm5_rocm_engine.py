from __future__ import annotations

import ctypes
import logging
import os

import torch

from tilert.pd_vllm.grammar_spec import GrammarUnsupported
from tilert.pd_vllm.sampling import resolve_top_p

logger = logging.getLogger("pd_vllm.profile.glm5_rocm")
_FP8_MAX = 448.0
_INDEX_HEAD_DIM = 128

_INJECT_MODE = os.environ.get("TILERT_INJECT_MODE", "rccl").strip().lower()


class _RcclBroadcast:
    _NCCL_CHAR = 0

    def __init__(self, npes: int):
        self.npes = npes
        self.lib = None
        self.path = None
        err = None
        for cand in (
            "librccl.so.1",
            "librccl.so",
            os.path.join(os.path.dirname(torch.__file__), "lib", "librccl.so.1"),
        ):
            try:
                self.lib = ctypes.CDLL(cand)
                self.path = cand
                break
            except OSError as exc:
                err = exc
        if self.lib is None:
            raise RuntimeError(f"librccl not loadable: {err}")
        ver = ctypes.c_int()
        self.lib.ncclGetVersion(ctypes.byref(ver))
        self.version = ver.value
        comms = (ctypes.c_void_p * npes)()
        devs = (ctypes.c_int * npes)(*range(npes))
        rc = self.lib.ncclCommInitAll(comms, npes, devs)
        if rc != 0:
            raise RuntimeError(f"ncclCommInitAll rc={rc}")
        self.comms = comms
        self.streams = [torch.cuda.Stream(device=d) for d in range(npes)]

    def run(self, items) -> None:
        lib = self.lib
        rc = lib.ncclGroupStart()
        if rc != 0:
            raise RuntimeError(f"ncclGroupStart rc={rc}")
        for root, src_ptr, nbytes, dsts in items:
            for r in range(self.npes):
                send = src_ptr if r == root else dsts[r]
                rc = lib.ncclBroadcast(
                    ctypes.c_void_p(send),
                    ctypes.c_void_p(dsts[r]),
                    ctypes.c_size_t(nbytes),
                    self._NCCL_CHAR,
                    ctypes.c_int(root),
                    self.comms[r],
                    ctypes.c_void_p(self.streams[r].cuda_stream),
                )
                if rc != 0:
                    raise RuntimeError(f"ncclBroadcast rc={rc}")
        rc = lib.ncclGroupEnd()
        if rc != 0:
            raise RuntimeError(f"ncclGroupEnd rc={rc}")
        for st in self.streams:
            st.synchronize()


def is_rocm_torch() -> bool:
    return getattr(torch.version, "hip", None) is not None


def build_rocm_engine(model_weights_dir, max_seq_len, with_mtp, ar_steps, num_mtp=3):
    import tilert

    if hasattr(tilert, "load_backend"):
        tilert.load_backend("glm5_2_rocm")
    from tilert.models.glm_5_2_rocm.generator import Glm52Generator
    from tilert.models.glm_5_2_rocm.model_args import ModelArgsGlm52

    if with_mtp and num_mtp not in (1, 3):
        raise ValueError(f"the ROCm GLM-5.2 engine builds MTP at depth 1 or 3, not {num_mtp}")
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
        if with_mtp and (not self.with_mtp):
            raise ValueError("--with-mtp requested but the engine was built with num_mtp=0")
        self.max_seq_len = int(self.dl.args.max_seq_len)
        self.n_layers = int(self.dl.n_layers)
        self.npes = int(self.dl.npes)
        self.mtp_seq_len = self.dl.num_mtp + 1
        self.ar_steps = max(1, min(1024, int(os.environ.get("GLM5_AR_N", str(ar_steps)))))
        self.stop_ids = {int(t) for t in generator.stop_token_ids}
        self.last_stats: dict = {}
        self._ignore_eos = False
        self._seq_len = 0
        from tilert.models.glm_5_2_rocm.model_args import full_layer_ordinals
        from tilert.models.glm_5_2_rocm.weight_converter import fp8_ki_enabled

        self._full_layers = list(full_layer_ordinals(self.n_layers))
        self._pure_tp8 = True if pure_tp8 is None else pure_tp8
        self._fp8_ki = fp8_ki_enabled() if fp8_ki is None else fp8_ki
        self._rot_streams = None
        self._rccl = None
        self._inject_mode = _INJECT_MODE if _INJECT_MODE in ("rccl", "rotate", "legacy") else "rccl"
        if self._inject_mode == "rccl":
            try:
                self._rccl = _RcclBroadcast(self.npes)
                logger.info(
                    "inject mode=rccl (RCCL %d via %s, %d comms)",
                    self._rccl.version,
                    self._rccl.path,
                    self.npes,
                )
            except Exception as exc:
                logger.warning("inject: RCCL unavailable (%s); falling back to rotate", exc)
                self._inject_mode = "rotate"
        else:
            logger.info("inject mode=%s", self._inject_mode)

    def supports_logprobs(self) -> bool:
        return False

    def supports_penalties(self) -> bool:
        return False

    def supports_ignore_eos(self) -> bool:
        return True

    def prepare_grammar(self, grammar_spec, enable_thinking=True):
        if grammar_spec is None:
            return None
        raise GrammarUnsupported("constrained decoding is not supported")

    def _write_ki(self, slot: torch.Tensor, ki: torch.Tensor, seq: int) -> None:
        if not self._fp8_ki:
            slot[0, :seq].copy_(ki.to(slot.device, non_blocking=True))
            return
        L = self.max_seq_len
        nbytes = L * _INDEX_HEAD_DIM
        x = ki.to(slot.device).float()
        amax = x.abs().amax(dim=-1).clamp_(min=1e-12)
        scale = amax / _FP8_MAX
        q = (x / scale.unsqueeze(-1)).clamp_(-_FP8_MAX, _FP8_MAX)
        q = q.to(torch.float8_e4m3fn)
        slot[:nbytes].view(torch.float8_e4m3fn).view(L, _INDEX_HEAD_DIM)[:seq].copy_(q)
        slot[nbytes : nbytes + L * 4].view(torch.float32)[:seq].copy_(scale)

    def _kv_targets(self, use, n_extra) -> dict:
        out = {}
        for rank in range(self.npes):
            if rank == 0 and (not self._pure_tp8):
                continue
            n_pairs = self.n_layers + (n_extra if rank != 0 else 0)
            out[rank] = min(len(use), n_pairs)
        return out

    def _move_kv(self, use, seq, n_extra) -> None:
        if self._inject_mode == "rccl":
            try:
                self._move_kv_rccl(use, seq, n_extra)
                return
            except Exception:
                logger.exception(
                    "inject: RCCL path failed; falling back to rotate for the rest of this run"
                )
                self._inject_mode = "rotate"
        if self._inject_mode == "rotate":
            self._move_kv_rotate(use, seq, n_extra)
            return
        self._move_kv_legacy(use, seq, n_extra)

    def _move_kv_legacy(self, use, seq, n_extra) -> None:
        dl = self.dl
        for rank, upto in self._kv_targets(use, n_extra).items():
            caches = dl._caches[rank]
            for lid in range(upto):
                _ki, kv, pe = use[lid]
                caches[2 * lid][0, :seq].copy_(kv, non_blocking=True)
                caches[2 * lid + 1][0, :seq].copy_(pe, non_blocking=True)

    def _move_kv_rotate(self, use, seq, n_extra) -> None:
        dl = self.dl
        upto = self._kv_targets(use, n_extra)
        by_src = {}
        for lid in range(len(use)):
            by_src.setdefault(use[lid][1].device.index, []).append(lid)
        if self._rot_streams is None:
            width = self.npes if len(by_src) < self.npes else 1
            self._rot_streams = {
                s: [torch.cuda.Stream(device=s) for _ in range(width)] for s in by_src
            }
        for k in range(self.npes):
            for s, lids in by_src.items():
                r = (s + k) % self.npes
                if r not in upto:
                    continue
                pool = self._rot_streams.get(s)
                if not pool:
                    pool = self._rot_streams[s] = [torch.cuda.Stream(device=s)]
                caches = dl._caches[r]
                with torch.cuda.device(s), torch.cuda.stream(pool[r % len(pool)]):
                    for lid in lids:
                        if lid >= upto[r]:
                            continue
                        _ki, kv, pe = use[lid]
                        caches[2 * lid][0, :seq].copy_(kv, non_blocking=True)
                        caches[2 * lid + 1][0, :seq].copy_(pe, non_blocking=True)
        for pool in self._rot_streams.values():
            for st in pool:
                st.synchronize()

    def _move_kv_rccl(self, use, seq, n_extra) -> None:
        if self._rccl is None:
            raise RuntimeError("RCCL broadcaster not initialised")
        dl = self.dl
        upto = self._kv_targets(use, n_extra)
        if len(upto) != self.npes:
            raise RuntimeError("rank 0 is excluded from KV injection; RCCL needs every rank")
        items = []
        leftovers = []
        for lid in range(len(use)):
            if any(lid >= upto[r] for r in range(self.npes)):
                leftovers.append(lid)
                continue
            _ki, kv, pe = use[lid]
            for ti, src in ((0, kv), (1, pe)):
                if not src.is_contiguous():
                    raise RuntimeError(f"layer {lid} tensor {ti} is not contiguous")
                dsts = []
                for r in range(self.npes):
                    dst = dl._caches[r][2 * lid + ti][0, :seq]
                    if not dst.is_contiguous():
                        raise RuntimeError(f"cache slot {lid}/{ti} on rank {r} is not contiguous")
                    dsts.append(dst.data_ptr())
                items.append(
                    (src.device.index, src.data_ptr(), src.numel() * src.element_size(), dsts)
                )
        self._rccl.run(items)
        for lid in leftovers:
            _ki, kv, pe = use[lid]
            for r in range(self.npes):
                if lid >= upto[r]:
                    continue
                caches = dl._caches[r]
                caches[2 * lid][0, :seq].copy_(kv, non_blocking=True)
                caches[2 * lid + 1][0, :seq].copy_(pe, non_blocking=True)

    def inject(self, req) -> None:
        dl = self.dl
        layers = req.layers
        seq = int(req.seq_len)
        n_extra = 1 if dl.num_mtp > 0 else 0
        if len(layers) not in (self.n_layers, self.n_layers + 1):
            raise RuntimeError(
                f"glm5_rocm inject: got {len(layers)} layers, engine has {self.n_layers} (+1 MTP block)"
            )
        if seq <= 0 or seq > self.max_seq_len:
            raise RuntimeError(f"glm5_rocm inject: seq_len {seq} outside (0, {self.max_seq_len}]")
        if n_extra and len(layers) == self.n_layers:
            raise RuntimeError(
                "glm5_rocm inject: engine has an MTP block but the prefill sent no MTP-layer KV (prefill must run with --speculative-config mtp)"
            )
        use = layers[: self.n_layers + n_extra]
        dl.reset_sequence()
        self._move_kv(use, seq, n_extra)
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
            raise GrammarUnsupported("constrained decoding is not supported")
        if top_logprobs:
            raise NotImplementedError("logprobs are not supported")
        sampling = sampling or {}
        rep = float(sampling.get("repetition_penalty", 1.0) or 1.0)
        pres = float(sampling.get("presence_penalty", 0.0) or 0.0)
        if rep != 1.0 or pres != 0.0:
            raise NotImplementedError(
                "repetition/presence penalties are not supported by this model's decode runtime"
            )
        temp = float(sampling.get("temperature", 1.0))
        if temp < 1e-05:
            self.dl.update_sampling(False, 1.0, 1.0)
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
        base_step = self.dl.step_count
        if self.with_mtp:
            finish = self._decode_mtp(first, budget, tokens, stop_ids, on_token, cancel_event)
        else:
            finish = self._decode_plain(first, budget, tokens, stop_ids, on_token, cancel_event)
        self.last_stats = {"finish_reason": finish}
        if self.with_mtp:
            self.last_stats.update(self._mtp_stats(base_step))
        return tokens

    def _mtp_stats(self, base_step: int) -> dict:
        per_step = self.dl.accepted_step_counts(base_step)
        if not per_step:
            return {}
        return {
            "mtp_verify_calls": len(per_step),
            "mtp_accept_mean": round(sum(per_step) / len(per_step), 3),
        }

    def _emit(self, new, tokens, budget, stop_ids, on_token):
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
        dl.step(first)
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
