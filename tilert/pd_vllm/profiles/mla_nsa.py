"""Shared MLA + NSA-KI data plane for the DeepSeek-family models."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import torch

from tilert.pd_vllm import wire

logger = logging.getLogger("pd_vllm.profile.mla_nsa")

KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
INDEX_HEAD_DIM = 128
KI_QUANT_BLOCK = 128
KV_QUANT_BLOCK = 128  # per-128 fp8 scale on the kv latent
PAGE_SIZE = 64

# The MLA KV cache dtype is a launch choice (vLLM ``--kv-cache-dtype``), NOT
# tied to the (fp8) model weights — both are supported and selected at runtime:
#
#   fp8_ds_mla : cache tensor [nblk, page, 656] u8; per token 512 fp8 kv_c +
#                16 B (4 fp32) scale + 128 B bf16 k_pe. Split into a 528-B
#                kv_merged plane + 128-B pe plane; kv dequantized fp8->bf16 on
#                the decode side. (recommended, aligns with SGLang fp8)
#   bf16       : cache tensor [nblk, page, 576] bf16; per token 512 bf16 kv_c +
#                64 bf16 k_pe = 1024-B kv plane + 128-B pe plane, no dequant.
KV_FP8_BYTES = KV_LORA_RANK  # 512 (fp8, 1 B each)
KV_SCALE_BYTES = KV_LORA_RANK // KV_QUANT_BLOCK * 4  # 16 (4 fp32 scales)
KV_BYTES_FP8 = KV_FP8_BYTES + KV_SCALE_BYTES  # 528 B/token
KV_BYTES_BF16 = KV_LORA_RANK * 2  # 1024 B/token
PE_BPT = QK_ROPE_HEAD_DIM * 2  # 128 B/token bf16 (both)
MLA_BPT_FP8 = KV_BYTES_FP8 + PE_BPT  # 656 (fp8 cache stride)
MLA_BPT_BF16 = (KV_LORA_RANK + QK_ROPE_HEAD_DIM) * 2  # 1152 (bf16 cache stride)
_VERSION_BF16_OFFSET = 40  # bf16 layout_version = base + 40
KI_PAGE_BYTES = (
    PAGE_SIZE * INDEX_HEAD_DIM + PAGE_SIZE * INDEX_HEAD_DIM // KI_QUANT_BLOCK * 4
)  # 8448


def _max_pages(max_seq_len: int) -> int:
    return (max_seq_len + PAGE_SIZE - 1) // PAGE_SIZE


def _hadamard(x: torch.Tensor) -> torch.Tensor:
    """Hadamard rotation of the last dim (scale d^-0.5), matching TileRT's indexer.

    Uses fast_hadamard_transform if present, else a scipy matmul.
    """
    d = x.shape[-1]
    try:
        from fast_hadamard_transform import hadamard_transform

        return hadamard_transform(x, scale=d**-0.5)
    except Exception:
        from scipy.linalg import hadamard as _h

        H = torch.from_numpy(_h(d).astype("float32")).to(x.device) * (d**-0.5)
        return (x.float() @ H).to(x.dtype)


@dataclass
class ConvertedRequest:
    rid: str
    seq_len: int
    last_prompt_token: int
    first_token_id: int | None
    sampling: dict | None
    layers: list  # [(ki[seq,128], kv[seq,512], pe[seq,64]) bf16] x num_layers


@dataclass
class _Reg:
    mla_layers: list  # [(lid, name, kv_t, gi)] sorted
    ki_layers: list  # [(lid, name, ki_t, gi)] sorted


class MlaNsaProfile:
    """Config-driven MLA+NSA profile.

    ``engine_factory(weights, max_seq, with_mtp, ar_steps) -> adapter`` builds
    the model-specific engine.
    """

    num_ranks = wire.NUM_RANKS
    sender_ranks = frozenset({0})  # MLA latent replicated across TP

    def __init__(
        self, name: str, num_layers: int, layout_version: int, engine_factory, mla_fp8: bool = True
    ):
        self.name = name
        self.num_layers = num_layers
        self._base_version = layout_version
        self._engine_factory = engine_factory
        self.mla_fp8 = mla_fp8  # fp8_ds_mla (True) vs bf16 (False) MLA cache

    def configure(self, kv_cache_dtype: str) -> MlaNsaProfile:
        """Select the MLA cache dtype (decode side; prefill auto-detects)."""
        d = (kv_cache_dtype or "").lower()
        if d in ("fp8_ds_mla", "fp8", "fp8_e4m3"):
            self.mla_fp8 = True
        elif d in ("bf16", "bfloat16", "auto"):
            self.mla_fp8 = False
        else:
            raise ValueError(
                f"unknown kv_cache_dtype {kv_cache_dtype!r}; " f"want fp8_ds_mla or bf16"
            )
        return self

    @property
    def layout_version(self) -> int:
        # distinct wire version per cache dtype so a mismatched pairing
        # (prefill fp8 vs decode bf16) is rejected at hello, not corrupted
        return self._base_version + (0 if self.mla_fp8 else _VERSION_BF16_OFFSET)

    @property
    def _kv_bpt(self) -> int:
        return KV_BYTES_FP8 if self.mla_fp8 else KV_BYTES_BF16

    @property
    def _mla_bpt(self) -> int:
        return MLA_BPT_FP8 if self.mla_fp8 else MLA_BPT_BF16

    # ── plane sizing (depends on num_layers + cache dtype) ──
    def _kv_plane(self, max_seq_len: int) -> int:
        return self.num_layers * max_seq_len * self._kv_bpt

    def _pe_plane(self, max_seq_len: int) -> int:
        return self.num_layers * max_seq_len * PE_BPT

    def _ki_plane(self, max_seq_len: int) -> int:
        return self.num_layers * _max_pages(max_seq_len) * KI_PAGE_BYTES

    # ── receive side ──
    def buffer_bytes(self, max_seq_len: int) -> int:
        return (
            self._kv_plane(max_seq_len) + self._pe_plane(max_seq_len) + self._ki_plane(max_seq_len)
        )

    def hello_layout(self, base_ptr: int, max_seq_len: int) -> dict[str, int]:
        kv = base_ptr
        pe = kv + self._kv_plane(max_seq_len)
        ki = pe + self._pe_plane(max_seq_len)
        return {"kv_base": kv, "pe_base": pe, "ki_base": ki}

    @torch.inference_mode()
    def convert(self, buffer, base_ptr, max_seq_len, received, num_devices=1):
        seq = received.seq_len
        npages = _max_pages(seq)
        pe_base = self._kv_plane(max_seq_len)
        ki_base = pe_base + self._pe_plane(max_seq_len)
        kv_bpt = self._kv_bpt
        layers = []
        for lid in range(self.num_layers):
            ko = lid * max_seq_len * kv_bpt
            kv_raw = buffer[ko : ko + seq * kv_bpt].view(seq, kv_bpt)
            if self.mla_fp8:
                kv = self._dequant_kv(kv_raw, seq)  # fp8+scale -> bf16 512
            else:
                kv = (
                    kv_raw.view(torch.bfloat16).view(seq, KV_LORA_RANK).contiguous()
                )  # already bf16
            po = pe_base + lid * max_seq_len * PE_BPT
            pe = (
                buffer[po : po + seq * PE_BPT]
                .view(torch.bfloat16)
                .view(seq, QK_ROPE_HEAD_DIM)
                .contiguous()
            )
            io = ki_base + lid * _max_pages(max_seq_len) * KI_PAGE_BYTES
            ki_raw = buffer[io : io + npages * KI_PAGE_BYTES].view(npages, KI_PAGE_BYTES)
            layers.append((self._dequant_ki(ki_raw, seq), kv, pe))
        torch.cuda.synchronize()
        return ConvertedRequest(
            rid=received.rid,
            seq_len=seq,
            last_prompt_token=received.last_prompt_token,
            first_token_id=received.first_token_id,
            sampling=received.sampling,
            layers=layers,
        )

    @staticmethod
    def _dequant_kv(kv_raw: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Dequantize kv_merged [seq,528] u8 (512 fp8 + 4 fp32 scale) -> bf16 [seq,512].

        Per-128-block scale: kv[:, b*128:(b+1)*128] *= scale[:, b].
        """
        nblk = KV_LORA_RANK // KV_QUANT_BLOCK
        fp8 = (
            kv_raw[:, :KV_FP8_BYTES]
            .reshape(-1)
            .view(torch.float8_e4m3fn)
            .reshape(seq_len, KV_LORA_RANK)
        )
        scale = (
            kv_raw[:, KV_FP8_BYTES:]
            .reshape(-1)
            .contiguous()
            .view(torch.float32)
            .reshape(seq_len, nblk)
        )
        fp32 = fp8.float().view(seq_len, nblk, KV_QUANT_BLOCK)
        deq = (fp32 * scale.unsqueeze(-1)).view(seq_len, KV_LORA_RANK)
        return deq.to(torch.bfloat16)

    @staticmethod
    def _dequant_ki(ki_raw: torch.Tensor, seq_len: int) -> torch.Tensor:
        npages = ki_raw.shape[0]
        fp8_bytes = PAGE_SIZE * INDEX_HEAD_DIM
        ki_fp8 = (
            ki_raw[:, :fp8_bytes]
            .reshape(-1)
            .view(torch.float8_e4m3fn)
            .reshape(npages * PAGE_SIZE, INDEX_HEAD_DIM)
        )
        scale = (
            ki_raw[:, fp8_bytes:]
            .reshape(-1)
            .contiguous()
            .view(torch.float32)
            .reshape(npages * PAGE_SIZE, INDEX_HEAD_DIM // KI_QUANT_BLOCK)
        )
        deq = (ki_fp8[:seq_len].float() * scale[:seq_len]).to(torch.bfloat16)
        return _hadamard(deq)

    # ── prefill side ──
    def classify_layers(self, kv_caches: dict, kv_cache_config) -> _Reg:
        group_of = {}
        for gi, g in enumerate(getattr(kv_cache_config, "kv_cache_groups", []) or []):
            for ln in getattr(g, "layer_names", []):
                group_of[ln] = gi

        def lid_of(name):
            m = re.search(r"\.(\d+)\.", name)
            base_i = int(m.group(1)) if m else -1
            return self.num_layers - 1 if name.startswith("mtp.") else base_i

        mla, ki = [], []
        for name, cache in kv_caches.items():
            t = cache[0] if isinstance(cache, (tuple, list)) else cache
            gi = group_of.get(name, -1)
            if "indexer" in name.lower() or "index_k" in name.lower():
                ki.append((lid_of(name), name, t, gi))
            else:
                mla.append((lid_of(name), name, t, gi))
        mla.sort(key=lambda x: x[0])
        ki.sort(key=lambda x: x[0])
        if len(mla) != self.num_layers or len(ki) != self.num_layers:
            raise RuntimeError(
                f"{self.name} classify: {len(mla)} MLA + {len(ki)} KI layers "
                f"(expected {self.num_layers} each); check --speculative-config"
                f" and the vLLM layer naming"
            )
        # auto-detect MLA cache dtype from the actual cache stride (the prefill
        # cache is ground truth; the decode side is told via --kv-cache-dtype)
        t0 = mla[0][2]
        bpt = t0.shape[-1] * t0.element_size()
        if bpt == MLA_BPT_FP8:
            self.mla_fp8 = True
        elif bpt == MLA_BPT_BF16:
            self.mla_fp8 = False
        else:
            raise RuntimeError(
                f"{self.name}: unexpected MLA cache stride {bpt} B/token; "
                f"expected {MLA_BPT_FP8} (fp8_ds_mla) or {MLA_BPT_BF16} (bf16)"
            )
        logger.info(
            "%s registered %d MLA + %d KI layers, MLA cache=%s",
            self.name,
            len(mla),
            len(ki),
            "fp8_ds_mla" if self.mla_fp8 else "bf16",
        )
        return _Reg(mla_layers=mla, ki_layers=ki)

    def staging_bytes(self, reg, tp_rank, max_seq_len):
        if tp_rank not in self.sender_ranks:
            return 4
        return self.buffer_bytes(max_seq_len)

    @torch.inference_mode()
    def extract(self, reg: _Reg, m, tp_rank, staging, max_seq_len):
        torch.cuda.synchronize()
        seq = m.num_tokens
        npages = _max_pages(seq)
        mla_ids = m.block_ids_per_group[reg.mla_layers[0][3]]
        bt = torch.tensor(mla_ids, dtype=torch.long)
        offs = torch.arange(PAGE_SIZE)
        slots = (offs.reshape(1, -1) + bt.reshape(-1, 1) * PAGE_SIZE).flatten()[:seq]
        ki_ids = m.block_ids_per_group[reg.ki_layers[0][3]]
        ki_bt = torch.tensor(ki_ids[:npages], dtype=torch.long)

        pe_base = self._kv_plane(max_seq_len)
        ki_base = pe_base + self._pe_plane(max_seq_len)
        kv_bpt, mla_bpt = self._kv_bpt, self._mla_bpt
        for lid in range(self.num_layers):
            # raw-byte split of the MLA cache row: works for both dtypes
            # (fp8 656 -> 528+128, bf16 1152 -> 1024+128), no conversion here
            kv_t = reg.mla_layers[lid][2]
            raw = kv_t if kv_t.dtype == torch.uint8 else kv_t.view(torch.uint8)
            flat = raw.reshape(-1, mla_bpt)  # [ntok, mla_bpt] u8
            rows = flat[slots.to(flat.device)]  # [seq, mla_bpt] u8
            kv_merged = rows[:, :kv_bpt].contiguous()  # kv_c (+scale)
            pe = rows[:, kv_bpt:].contiguous()  # 64 bf16 (128 B)
            ko = lid * max_seq_len * kv_bpt
            po = pe_base + lid * max_seq_len * PE_BPT
            staging[ko : ko + seq * kv_bpt].copy_(kv_merged.flatten())
            staging[po : po + seq * PE_BPT].copy_(pe.flatten())

            ki_t = reg.ki_layers[lid][2]
            ki_pages = ki_t[ki_bt.to(ki_t.device)].reshape(npages, -1)
            io = ki_base + lid * _max_pages(max_seq_len) * KI_PAGE_BYTES
            staging[io : io + npages * KI_PAGE_BYTES].copy_(
                ki_pages.contiguous().view(torch.uint8).flatten()
            )
        torch.cuda.synchronize()
        return {"seq": seq, "npages": npages, "stage_max": max_seq_len}

    def rdma_plan(self, hello, sections, tp_rank, seq_len, base):
        remote_max = int(hello["max_seq_len"])
        stage_max = sections["stage_max"]
        npages = sections["npages"]
        srcs, dsts, lens = [], [], []
        s_pe = self._kv_plane(stage_max)
        s_ki = s_pe + self._pe_plane(stage_max)
        kv_bpt = self._kv_bpt
        r_kv, r_pe, r_ki = (int(hello["kv_base"]), int(hello["pe_base"]), int(hello["ki_base"]))
        for lid in range(self.num_layers):
            srcs.append(base + lid * stage_max * kv_bpt)
            dsts.append(r_kv + lid * remote_max * kv_bpt)
            lens.append(seq_len * kv_bpt)
            srcs.append(base + s_pe + lid * stage_max * PE_BPT)
            dsts.append(r_pe + lid * remote_max * PE_BPT)
            lens.append(seq_len * PE_BPT)
            srcs.append(base + s_ki + lid * _max_pages(stage_max) * KI_PAGE_BYTES)
            dsts.append(r_ki + lid * _max_pages(remote_max) * KI_PAGE_BYTES)
            lens.append(npages * KI_PAGE_BYTES)
        return srcs, dsts, lens

    def build_engine(self, model_weights_dir, max_seq_len, with_mtp, ar_steps):
        return self._engine_factory(model_weights_dir, max_seq_len, with_mtp, ar_steps)


class MlaNsaEngineAdapter:
    """Shared decode adapter for GLM-5 / DSV3.2 (same inject + 3-phase MTP).

    ``generator`` is a ready ``from_pretrained``'d GLM5Generator /
    DSAv32Generator; both expose inject_cache / set_cur_pos / decode_layer
    with forward / get_next_draft_tokens / get_num_accepted /
    get_predicted_tokens / reset_sequence, and share DSV3.2's TOKEN_OUT index.
    """

    def __init__(self, generator, with_mtp: bool):
        import torch as _torch

        self._torch = _torch
        self.gen = generator
        self.with_mtp = with_mtp
        self.mtp_seq_len = getattr(generator, "mtp_seq_len", 4)
        self.max_seq_len = getattr(generator.decode_layer, "max_seq_len", 200000)
        self.last_stats: dict = {}
        self.stop_ids = self._resolve_stop_ids(generator)

    @staticmethod
    def _resolve_stop_ids(generator) -> set:
        # GLM-5 exposes a stop_token_ids set; DSV3.2 exposes only eos_id.
        sids = getattr(generator, "stop_token_ids", None)
        if sids:
            return set(sids)
        eos = getattr(generator, "eos_id", None)
        return {int(eos)} if eos is not None else set()

    def inject(self, req) -> None:
        self.gen.inject_cache(req.layers, start_pos=0)
        self.gen.set_cur_pos(req.seq_len - 1)
        self._last_prompt_token = req.last_prompt_token
        self._seq_len = req.seq_len

    def decode(self, first_token_id, max_tokens, sampling, on_token=None, cancel_event=None):
        sampling = sampling or {}
        temp = float(sampling.get("temperature", 1.0))
        if temp < 1e-5:
            self.gen.update_sampling_params(temperature=1.0, top_p=1.0, top_k=1, use_topp=False)
        else:
            self.gen.update_sampling_params(
                temperature=temp,
                top_p=float(sampling.get("top_p", 0.95)),
                top_k=int(sampling.get("top_k", 256)),
                use_topp=True,
            )
        budget = min(int(max_tokens), self.max_seq_len - self._seq_len - 1)
        if budget <= 0:
            self.last_stats = {"finish_reason": "length"}
            return [int(first_token_id)]
        if self.with_mtp:
            return self._decode_mtp(first_token_id, budget, on_token, cancel_event)
        return self._decode_standard(first_token_id, budget, on_token, cancel_event)

    def _decode_mtp(self, first_token_id, budget, on_token, cancel_event):
        dl = self.gen.decode_layer
        T = self.mtp_seq_len
        stop_ids = self.stop_ids
        torch = self._torch
        tokens = [int(first_token_id)]
        if on_token:
            on_token(int(first_token_id))
        if int(first_token_id) in stop_ids:
            self.last_stats = {"finish_reason": "stop"}
            return []
        dl.set_prefill_valid_tokens(0)
        draft = torch.full((1, T), int(self._last_prompt_token), dtype=torch.int32, device="cuda:0")
        accepted, finish, fwd, finished = [], "length", 0, False
        while not finished and len(tokens) < budget:
            if cancel_event is not None and cancel_event.is_set():
                finish = "cancelled"
                break
            if fwd == 1:
                draft = torch.full((1, T), int(first_token_id), dtype=torch.int32, device="cuda:0")
            elif fwd > 1:
                draft = dl.get_next_draft_tokens(0).reshape(1, T)
            dl.forward(draft)
            n_acc = dl.get_num_accepted(0)
            pred = dl.get_predicted_tokens(0).flatten()
            if fwd == 0:
                fwd += 1
                continue
            accepted.append(n_acc)
            fwd += 1
            for i in range(n_acc):
                if len(tokens) >= budget:
                    break
                tok = int(pred[i].item())
                if tok in stop_ids:
                    finished = True
                    finish = "stop"
                    break
                tokens.append(tok)
                if on_token:
                    on_token(tok)
        dl.reset_sequence()
        self.last_stats = {
            "finish_reason": finish,
            "mtp_accept_mean": round(sum(accepted) / max(1, len(accepted)), 3),
            "mtp_verify_calls": len(accepted),
        }
        return tokens

    def _decode_standard(self, first_token_id, budget, on_token, cancel_event):
        from tilert.models.deepseek_v3_2.temp_var_indices import Idx

        dl = self.gen.decode_layer
        stop_ids = self.stop_ids
        torch = self._torch
        tokens = [int(first_token_id)]
        if on_token:
            on_token(int(first_token_id))
        if int(first_token_id) in stop_ids:
            self.last_stats = {"finish_reason": "stop"}
            return []
        finish = "length"
        cur = torch.tensor(int(first_token_id), dtype=torch.long, device="cuda:0")
        while len(tokens) < budget:
            if cancel_event is not None and cancel_event.is_set():
                finish = "cancelled"
                break
            res = dl.forward(cur)
            intermediates, *_ = res[0]
            nxt = intermediates[Idx.TOKEN_OUT][0][0]
            tok = int(nxt.item())
            if tok in stop_ids:
                finish = "stop"
                break
            tokens.append(tok)
            if on_token:
                on_token(tok)
            cur = nxt
        dl.reset_sequence()
        self.last_stats = {"finish_reason": finish}
        return tokens

    def reset(self) -> None:
        pass
