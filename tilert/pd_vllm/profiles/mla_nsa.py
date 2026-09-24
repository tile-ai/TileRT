from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

import torch

from tilert.pd_vllm import wire
from tilert.pd_vllm.grammar_spec import (
    GrammarUnsupported,
    GrammarViolationError,
    InvalidGrammarError,
)
from tilert.pd_vllm.profiles import base
from tilert.pd_vllm.sampling import resolve_top_k, resolve_top_p

logger = logging.getLogger("pd_vllm.profile.mla_nsa")
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
INDEX_HEAD_DIM = 128
KI_QUANT_BLOCK = 128
KV_QUANT_BLOCK = 128
PAGE_SIZE = 64
KI_TILE = 16
KV_FP8_BYTES = KV_LORA_RANK
KV_SCALE_BYTES = KV_LORA_RANK // KV_QUANT_BLOCK * 4
KV_BYTES_FP8 = KV_FP8_BYTES + KV_SCALE_BYTES
KV_BYTES_BF16 = KV_LORA_RANK * 2
PE_BPT = QK_ROPE_HEAD_DIM * 2
MLA_BPT_FP8 = KV_BYTES_FP8 + PE_BPT
MLA_BPT_BF16 = (KV_LORA_RANK + QK_ROPE_HEAD_DIM) * 2
KV_BYTES_ROCM_FP8 = KV_LORA_RANK
PE_BYTES_ROCM_FP8 = QK_ROPE_HEAD_DIM
MLA_BPT_ROCM_FP8 = KV_LORA_RANK + QK_ROPE_HEAD_DIM
_VERSION_BF16_OFFSET = 40
_VERSION_KI_TILED_OFFSET = 100
_VERSION_ROCM_FP8_OFFSET = 200
KI_PAGE_BYTES = PAGE_SIZE * INDEX_HEAD_DIM + PAGE_SIZE * INDEX_HEAD_DIM // KI_QUANT_BLOCK * 4


def _default_ki_tiled() -> bool:
    env = (os.environ.get("TILERT_PD_KI_TILED") or "").strip().lower()
    if env:
        return env not in ("0", "false", "no", "off")
    return getattr(torch.version, "hip", None) is not None


def _max_pages(max_seq_len: int) -> int:
    return (max_seq_len + PAGE_SIZE - 1) // PAGE_SIZE


def _hadamard(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    try:
        from fast_hadamard_transform import hadamard_transform

        return hadamard_transform(x, scale=d ** (-0.5))
    except Exception:
        from scipy.linalg import hadamard as _h

        H = torch.from_numpy(_h(d).astype("float32")).to(x.device) * d ** (-0.5)
        return (x.float() @ H).to(x.dtype)


@dataclass
class ConvertedRequest:
    rid: str
    seq_len: int
    last_prompt_token: int
    first_token_id: int | None
    sampling: dict | None
    layers: list


_EXTRACT_MODE = os.environ.get("TILERT_EXTRACT_MODE", "fast").strip().lower()
_PD_SENDERS = max(
    1, min(int((os.environ.get("TILERT_PD_SENDERS") or "1").strip() or 1), wire.NUM_RANKS)
)


@dataclass
class _Reg:
    mla_layers: list
    ki_layers: list


class MlaNsaProfile:
    num_ranks = wire.NUM_RANKS
    sender_ranks = frozenset(range(_PD_SENDERS))
    declares_penalties = False

    def __init__(
        self,
        name: str,
        num_layers: int,
        layout_version: int,
        engine_factory,
        mla_fp8: bool = True,
        ki_layer_ids: list[int] | None = None,
        ki_tiled: bool | None = None,
    ):
        self.name = name
        self.num_layers = num_layers
        self._base_version = layout_version
        self._engine_factory = engine_factory
        self.mla_fp8 = mla_fp8
        self.mla_rocm_fp8 = False
        self.kv_scales: list[float] | None = None
        self.ki_tiled = _default_ki_tiled() if ki_tiled is None else bool(ki_tiled)
        self.ki_layer_ids = ki_layer_ids

    def configure(self, kv_cache_dtype: str) -> MlaNsaProfile:
        d = (kv_cache_dtype or "").lower()
        if d in ("fp8_rocm", "rocm_fp8"):
            self.mla_fp8 = False
            self.mla_rocm_fp8 = True
        elif d in ("fp8_ds_mla", "fp8", "fp8_e4m3"):
            self.mla_fp8 = True
            self.mla_rocm_fp8 = False
        elif d in ("bf16", "bfloat16", "auto"):
            self.mla_fp8 = False
            self.mla_rocm_fp8 = False
        else:
            raise ValueError(
                f"unknown kv_cache_dtype {kv_cache_dtype!r}; want fp8_ds_mla, fp8_rocm or bf16"
            )
        return self

    @property
    def layout_version(self) -> int:
        return (
            self._base_version
            + (0 if self.mla_fp8 else _VERSION_BF16_OFFSET)
            + (_VERSION_KI_TILED_OFFSET if self.ki_tiled else 0)
            + (_VERSION_ROCM_FP8_OFFSET if self.mla_rocm_fp8 else 0)
        )

    @property
    def _kv_bpt(self) -> int:
        if self.mla_rocm_fp8:
            return KV_BYTES_ROCM_FP8
        return KV_BYTES_FP8 if self.mla_fp8 else KV_BYTES_BF16

    @property
    def _pe_bpt(self) -> int:
        return PE_BYTES_ROCM_FP8 if self.mla_rocm_fp8 else PE_BPT

    @property
    def _mla_bpt(self) -> int:
        if self.mla_rocm_fp8:
            return MLA_BPT_ROCM_FP8
        return MLA_BPT_FP8 if self.mla_fp8 else MLA_BPT_BF16

    def _kv_plane(self, max_seq_len: int) -> int:
        return self.num_layers * max_seq_len * self._kv_bpt

    def _pe_plane(self, max_seq_len: int) -> int:
        return self.num_layers * max_seq_len * self._pe_bpt

    def _ki_plane(self, max_seq_len: int) -> int:
        return self.num_layers * _max_pages(max_seq_len) * KI_PAGE_BYTES

    def buffer_bytes(self, max_seq_len: int) -> int:
        return (
            self._kv_plane(max_seq_len) + self._pe_plane(max_seq_len) + self._ki_plane(max_seq_len)
        )

    def _shard_layers(self, nshards: int) -> int:
        return -(-self.num_layers // nshards)

    def shard_bytes(self, max_seq_len: int, nshards: int) -> int:
        n = self._shard_layers(nshards)
        return (
            n * max_seq_len * self._kv_bpt
            + n * max_seq_len * self._pe_bpt
            + n * _max_pages(max_seq_len) * KI_PAGE_BYTES
        )

    def hello_layout(self, base_ptr, max_seq_len: int) -> dict:
        if not isinstance(base_ptr, (list, tuple)):
            kv = base_ptr
            pe = kv + self._kv_plane(max_seq_len)
            ki = pe + self._pe_plane(max_seq_len)
            return {"senders": len(self.sender_ranks), "kv_base": kv, "pe_base": pe, "ki_base": ki}
        n = self._shard_layers(len(base_ptr))
        kv_plane = n * max_seq_len * self._kv_bpt
        pe_plane = n * max_seq_len * self._pe_bpt
        return {
            "senders": len(self.sender_ranks),
            "kv_base": [int(p) for p in base_ptr],
            "pe_base": [int(p) + kv_plane for p in base_ptr],
            "ki_base": [int(p) + kv_plane + pe_plane for p in base_ptr],
            "nshards": len(base_ptr),
        }

    @torch.inference_mode()
    def convert(self, buffer, base_ptr, max_seq_len, received, num_devices=1):
        seq = received.seq_len
        npages = _max_pages(seq)
        pe_base = self._kv_plane(max_seq_len)
        ki_base = pe_base + self._pe_plane(max_seq_len)
        kv_bpt = self._kv_bpt
        pe_bpt = self._pe_bpt
        layers = []
        bufs = list(buffer) if isinstance(buffer, (list, tuple)) else [buffer]
        nsh = len(bufs)
        host = not bufs[0].is_cuda
        scales = self.kv_scales if self.mla_rocm_fp8 else None
        if nsh > 1:
            lps = self._shard_layers(nsh)
            pe_base = lps * max_seq_len * self._kv_bpt
            ki_base = pe_base + lps * max_seq_len * pe_bpt
        for lid in range(self.num_layers):
            sh, li = (lid % nsh, lid // nsh) if nsh > 1 else (0, lid)
            buf = bufs[sh]
            ko = li * max_seq_len * kv_bpt
            kv_raw = buf[ko : ko + seq * kv_bpt]
            if host:
                kv_raw = kv_raw.to("cuda:0", non_blocking=True)
            kv_raw = kv_raw.view(seq, kv_bpt)
            if self.mla_rocm_fp8:
                kv = self._dequant_rocm_fp8(
                    kv_raw, seq, KV_LORA_RANK, scales[lid] if scales else 1.0
                )
            elif self.mla_fp8:
                kv = self._dequant_kv(kv_raw, seq)
            else:
                kv = kv_raw.view(torch.bfloat16).view(seq, KV_LORA_RANK).contiguous()
            po = pe_base + li * max_seq_len * pe_bpt
            pe_raw = buf[po : po + seq * pe_bpt]
            if host:
                pe_raw = pe_raw.to("cuda:0", non_blocking=True)
            if self.mla_rocm_fp8:
                pe = self._dequant_rocm_fp8(
                    pe_raw.view(seq, pe_bpt), seq, QK_ROPE_HEAD_DIM, scales[lid] if scales else 1.0
                )
            else:
                pe = pe_raw.view(torch.bfloat16).view(seq, QK_ROPE_HEAD_DIM).contiguous()
            io = ki_base + li * _max_pages(max_seq_len) * KI_PAGE_BYTES
            ki_raw = buf[io : io + npages * KI_PAGE_BYTES]
            if host:
                ki_raw = ki_raw.to("cuda:0", non_blocking=True)
            ki_raw = ki_raw.view(npages, KI_PAGE_BYTES)
            layers.append((self._dequant_ki(ki_raw, seq, self.ki_tiled), kv, pe))
        for d in sorted({b.device.index for b in bufs if b.is_cuda} or {0}):
            torch.cuda.synchronize(d)
        return ConvertedRequest(
            rid=received.rid,
            seq_len=seq,
            last_prompt_token=received.last_prompt_token,
            first_token_id=received.first_token_id,
            sampling=received.sampling,
            layers=layers,
        )

    @staticmethod
    def _dequant_rocm_fp8(
        raw: torch.Tensor, seq_len: int, width: int, scale: float
    ) -> torch.Tensor:
        x = raw.reshape(-1).contiguous().view(torch.float8_e4m3fn).reshape(seq_len, width)
        if scale == 1.0:
            return x.to(torch.bfloat16)
        return (x.float() * scale).to(torch.bfloat16)

    @staticmethod
    def _dequant_kv(kv_raw: torch.Tensor, seq_len: int) -> torch.Tensor:
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
    def _dequant_ki(ki_raw: torch.Tensor, seq_len: int, tiled: bool) -> torch.Tensor:
        npages = ki_raw.shape[0]
        fp8_bytes = PAGE_SIZE * INDEX_HEAD_DIM
        plane = ki_raw[:, :fp8_bytes]
        if tiled:
            plane = (
                plane.contiguous()
                .view(npages, PAGE_SIZE // KI_TILE, INDEX_HEAD_DIM // KI_TILE, KI_TILE, KI_TILE)
                .permute(0, 1, 3, 2, 4)
            )
        ki_fp8 = (
            plane.reshape(npages * PAGE_SIZE, INDEX_HEAD_DIM)[:seq_len]
            .contiguous()
            .view(torch.float8_e4m3fn)
        )
        scale = (
            ki_raw[:, fp8_bytes:]
            .reshape(-1)
            .contiguous()
            .view(torch.float32)
            .reshape(npages * PAGE_SIZE, INDEX_HEAD_DIM // KI_QUANT_BLOCK)
        )
        deq = (ki_fp8.float() * scale[:seq_len]).to(torch.bfloat16)
        return _hadamard(deq)

    def classify_layers(self, kv_caches: dict, kv_cache_config) -> _Reg:
        group_of = {}
        for gi, g in enumerate(getattr(kv_cache_config, "kv_cache_groups", []) or []):
            for ln in getattr(g, "layer_names", []):
                group_of[ln] = gi

        def lid_of(name):
            m = re.search("\\.(\\d+)\\.", name)
            base_i = int(m.group(1)) if m else -1
            return self.num_layers - 1 if name.startswith("mtp.") else base_i

        mla, ki = ([], [])
        for name, val in kv_caches.items():
            t = val[0] if isinstance(val, (tuple, list)) else val
            gi = group_of.get(name, -1)
            if "indexer" in name.lower() or "index_k" in name.lower():
                ki.append((lid_of(name), name, t, gi))
            else:
                mla.append((lid_of(name), name, t, gi))
        mla.sort(key=lambda x: x[0])
        ki.sort(key=lambda x: x[0])
        if len(mla) != self.num_layers:
            raise RuntimeError(
                f"{self.name} classify: {len(mla)} MLA layers (expected {self.num_layers}); check --speculative-config and the vLLM layer naming"
            )
        ki_ids = [x[0] for x in ki]
        if not ki or ki[0][0] != 0:
            raise RuntimeError(
                f"{self.name} classify: KI layer 0 missing (ids={ki_ids}); cannot expand sparse indexer set"
            )
        if len(ki) > self.num_layers or ki_ids != sorted(set(ki_ids)):
            raise RuntimeError(
                f"{self.name} classify: bad KI layer set {ki_ids} (num_layers={self.num_layers})"
            )
        if self.ki_layer_ids is not None:
            want = [l for l in self.ki_layer_ids if l < self.num_layers]
            want_no_mtp = [l for l in want if l != self.num_layers - 1]
            if ki_ids not in (want, want_no_mtp):
                raise RuntimeError(
                    f"{self.name} classify: KI layer ids {ki_ids} != expected {want} (or {want_no_mtp} without the MTP tail)"
                )
        ki_expanded, cur, idx = ([], None, 0)
        for L in range(self.num_layers):
            while idx < len(ki) and ki[idx][0] <= L:
                cur = ki[idx]
                idx += 1
            ki_expanded.append((L, cur[1], cur[2], cur[3]))
        if len(ki) < self.num_layers:
            logger.info(
                "%s: sparse KI %d full layers %s expanded to %d (shared layers reuse previous full layer's indexer)",
                self.name,
                len(ki),
                ki_ids,
                self.num_layers,
            )
        ki = ki_expanded
        t0 = mla[0][2]
        bpt = t0.shape[-1] * t0.element_size()
        if bpt == MLA_BPT_FP8:
            self.mla_fp8, self.mla_rocm_fp8 = (True, False)
        elif bpt == MLA_BPT_BF16:
            self.mla_fp8, self.mla_rocm_fp8 = (False, False)
        elif bpt == MLA_BPT_ROCM_FP8:
            self.mla_fp8, self.mla_rocm_fp8 = (False, True)
        else:
            raise RuntimeError(
                f"{self.name}: unexpected MLA cache stride {bpt} B/token; expected {MLA_BPT_FP8} (fp8_ds_mla), {MLA_BPT_BF16} (bf16) or {MLA_BPT_ROCM_FP8} (ROCm flat fp8)"
            )
        k0 = ki[0][2]
        ki_page = int(k0[0].numel() * k0.element_size()) if k0.dim() > 1 else 0
        if ki_page != KI_PAGE_BYTES:
            raise RuntimeError(
                f"{self.name}: KI cache page is {ki_page} B, expected {KI_PAGE_BYTES} "
                f"({PAGE_SIZE} tokens x {INDEX_HEAD_DIM} fp8 + one fp32 scale per token)"
            )
        layout = (
            "ROCm flat fp8" if self.mla_rocm_fp8 else ("fp8_ds_mla" if self.mla_fp8 else "bf16")
        )
        logger.info(
            "%s registered %d MLA + %d KI layers, MLA cache=%s (%d B/token, layout v%d)",
            self.name,
            len(mla),
            len(ki),
            layout,
            bpt,
            self.layout_version,
        )
        return _Reg(mla_layers=mla, ki_layers=ki)

    def set_kv_scales(self, scales) -> None:
        if scales is None:
            self.kv_scales = None
            return
        vals = [float(v) for v in scales]
        if len(vals) != self.num_layers:
            raise RuntimeError(
                f"{self.name}: got {len(vals)} kv scales for {self.num_layers} layers"
            )
        self.kv_scales = None if all(v == 1.0 for v in vals) else vals
        logger.info(
            "%s kv scales: %s",
            self.name,
            (
                "all 1.0 (no dequant multiply)"
                if self.kv_scales is None
                else f"min {min(vals):.6g} max {max(vals):.6g}"
            ),
        )

    def staging_bytes(self, reg, tp_rank, max_seq_len, nshards: int = 1):
        if tp_rank not in self.sender_ranks:
            return 4
        if _PD_SENDERS > 1:
            return self.shard_bytes(max_seq_len, _PD_SENDERS)
        return (
            self.shard_bytes(max_seq_len, nshards)
            if nshards > 1
            else self.buffer_bytes(max_seq_len)
        )

    @torch.inference_mode()
    def extract(self, reg: _Reg, m, tp_rank, staging, max_seq_len):
        global _EXTRACT_MODE
        if _EXTRACT_MODE == "legacy":
            return self._extract_legacy(reg, m, tp_rank, staging, max_seq_len)
        try:
            return self._extract_fast(reg, m, tp_rank, staging, max_seq_len)
        except Exception:
            logger.exception("extract: fast path failed; using legacy for the rest of this run")
            _EXTRACT_MODE = "legacy"
            return self._extract_legacy(reg, m, tp_rank, staging, max_seq_len)

    @torch.inference_mode()
    def _extract_fast(self, reg: _Reg, m, tp_rank, staging, max_seq_len):
        torch.cuda.synchronize()
        seq = m.num_tokens
        npages = _max_pages(seq)
        mla_ids = m.block_ids_per_group[reg.mla_layers[0][3]]
        bt = torch.tensor(mla_ids, dtype=torch.long)
        offs = torch.arange(PAGE_SIZE)
        slots_cpu = (offs.reshape(1, -1) + bt.reshape(-1, 1) * PAGE_SIZE).flatten()[:seq]
        ki_ids = m.block_ids_per_group[reg.ki_layers[0][3]]
        ki_bt_cpu = torch.tensor(ki_ids[:npages], dtype=torch.long)
        stgs = list(staging) if isinstance(staging, (list, tuple)) else [staging]
        nsh = len(stgs)
        if _PD_SENDERS > 1 and nsh != _PD_SENDERS:
            raise RuntimeError(
                f"TILERT_PD_SENDERS={_PD_SENDERS} needs the same number of staging shards, got {nsh}"
            )
        lps = self._shard_layers(nsh) if nsh > 1 else self.num_layers
        pe_bpt = self._pe_bpt
        pe_base = lps * max_seq_len * self._kv_bpt
        ki_base = pe_base + lps * max_seq_len * pe_bpt
        kv_bpt, mla_bpt = (self._kv_bpt, self._mla_bpt)
        idx, tmp, streams = ({}, {}, {})

        def index_for(dev):
            if dev not in idx:
                idx[dev] = (slots_cpu.to(dev), ki_bt_cpu.to(dev))
            return idx[dev]

        def scratch_for(dev, sh):
            key = (dev.index, sh)
            if key not in tmp:
                tmp[key] = (
                    torch.empty(seq, mla_bpt, dtype=torch.uint8, device=dev),
                    torch.empty(npages, KI_PAGE_BYTES, dtype=torch.uint8, device=dev),
                )
            return tmp[key]

        def stream_for(dev, sh):
            key = (dev.index, sh)
            if key not in streams:
                streams[key] = torch.cuda.Stream(device=dev)
            return streams[key]

        for lid in range(self.num_layers):
            sh, li = (lid % nsh, lid // nsh) if nsh > 1 else (0, lid)
            if _PD_SENDERS > 1 and sh != tp_rank:
                continue
            stg = stgs[sh]
            kv_t = reg.mla_layers[lid][2]
            raw = kv_t if kv_t.dtype == torch.uint8 else kv_t.view(torch.uint8)
            flat = raw.reshape(-1, mla_bpt)
            ki_t = reg.ki_layers[lid][2]
            ki_raw = ki_t if ki_t.dtype == torch.uint8 else ki_t.view(torch.uint8)
            ki_flat = ki_raw.reshape(ki_t.shape[0], -1)
            if ki_flat.shape[1] != KI_PAGE_BYTES:
                raise RuntimeError(f"KI page stride {ki_flat.shape[1]} B != {KI_PAGE_BYTES}")
            slots, ki_bt = index_for(flat.device)
            ko = li * max_seq_len * kv_bpt
            po = pe_base + li * max_seq_len * pe_bpt
            io = ki_base + li * _max_pages(max_seq_len) * KI_PAGE_BYTES
            dst_kv = stg[ko : ko + seq * kv_bpt].view(seq, kv_bpt)
            dst_pe = stg[po : po + seq * pe_bpt].view(seq, pe_bpt)
            dst_ki = stg[io : io + npages * KI_PAGE_BYTES].view(npages, KI_PAGE_BYTES)
            same = stg.is_cuda and stg.device == flat.device
            if same:
                torch.index_select(flat[:, :kv_bpt], 0, slots, out=dst_kv)
                torch.index_select(flat[:, kv_bpt:], 0, slots, out=dst_pe)
                torch.index_select(ki_flat, 0, ki_bt, out=dst_ki)
            else:
                rows, kirows = scratch_for(flat.device, sh)
                with torch.cuda.device(flat.device), torch.cuda.stream(stream_for(flat.device, sh)):
                    torch.index_select(flat, 0, slots, out=rows)
                    dst_kv.copy_(rows[:, :kv_bpt], non_blocking=True)
                    dst_pe.copy_(rows[:, kv_bpt:], non_blocking=True)
                    torch.index_select(ki_flat, 0, ki_bt, out=kirows)
                    dst_ki.copy_(kirows, non_blocking=True)
        for st in streams.values():
            st.synchronize()
        torch.cuda.synchronize()
        return {"seq": seq, "npages": npages, "stage_max": max_seq_len, "stage_shards": nsh}

    @torch.inference_mode()
    def _extract_legacy(self, reg: _Reg, m, tp_rank, staging, max_seq_len):
        torch.cuda.synchronize()
        seq = m.num_tokens
        npages = _max_pages(seq)
        mla_ids = m.block_ids_per_group[reg.mla_layers[0][3]]
        bt = torch.tensor(mla_ids, dtype=torch.long)
        offs = torch.arange(PAGE_SIZE)
        slots = (offs.reshape(1, -1) + bt.reshape(-1, 1) * PAGE_SIZE).flatten()[:seq]
        ki_ids = m.block_ids_per_group[reg.ki_layers[0][3]]
        ki_bt = torch.tensor(ki_ids[:npages], dtype=torch.long)
        stgs = list(staging) if isinstance(staging, (list, tuple)) else [staging]
        nsh = len(stgs)
        lps = self._shard_layers(nsh) if nsh > 1 else self.num_layers
        pe_bpt = self._pe_bpt
        pe_base = lps * max_seq_len * self._kv_bpt
        ki_base = pe_base + lps * max_seq_len * pe_bpt
        kv_bpt, mla_bpt = (self._kv_bpt, self._mla_bpt)
        for lid in range(self.num_layers):
            sh, li = (lid % nsh, lid // nsh) if nsh > 1 else (0, lid)
            if _PD_SENDERS > 1 and sh != tp_rank:
                continue
            stg = stgs[sh]
            kv_t = reg.mla_layers[lid][2]
            raw = kv_t if kv_t.dtype == torch.uint8 else kv_t.view(torch.uint8)
            flat = raw.reshape(-1, mla_bpt)
            rows = flat[slots.to(flat.device)]
            kv_merged = rows[:, :kv_bpt].contiguous()
            pe = rows[:, kv_bpt:].contiguous()
            ko = li * max_seq_len * kv_bpt
            po = pe_base + li * max_seq_len * pe_bpt
            stg[ko : ko + seq * kv_bpt].copy_(kv_merged.flatten())
            stg[po : po + seq * pe_bpt].copy_(pe.flatten())
            ki_t = reg.ki_layers[lid][2]
            ki_pages = ki_t[ki_bt.to(ki_t.device)].reshape(npages, -1)
            io = ki_base + li * _max_pages(max_seq_len) * KI_PAGE_BYTES
            stg[io : io + npages * KI_PAGE_BYTES].copy_(
                ki_pages.contiguous().view(torch.uint8).flatten()
            )
        torch.cuda.synchronize()
        return {"seq": seq, "npages": npages, "stage_max": max_seq_len, "stage_shards": nsh}

    def rdma_plan(self, hello, sections, tp_rank, seq_len, base):
        remote_max = int(hello["max_seq_len"])
        stage_max = sections["stage_max"]
        npages = sections["npages"]
        srcs, dsts, lens = ([], [], [])
        sbases = list(base) if isinstance(base, (list, tuple)) else [base]
        snsh = len(sbases)
        s_lps = self._shard_layers(snsh) if snsh > 1 else self.num_layers
        kv_bpt = self._kv_bpt
        pe_bpt = self._pe_bpt
        s_pe = s_lps * stage_max * kv_bpt
        s_ki = s_pe + s_lps * stage_max * pe_bpt
        dnsh = int(hello.get("nshards", 1) or 1)
        r_kv, r_pe, r_ki = (hello["kv_base"], hello["pe_base"], hello["ki_base"])
        for lid in range(self.num_layers):
            if dnsh > 1:
                dsh, dli = lid % dnsh, lid // dnsh
                d_kv, d_pe, d_ki = int(r_kv[dsh]), int(r_pe[dsh]), int(r_ki[dsh])
            else:
                dli = lid
                d_kv, d_pe, d_ki = int(r_kv), int(r_pe), int(r_ki)
            ssh, sli = (lid % snsh, lid // snsh) if snsh > 1 else (0, lid)
            if _PD_SENDERS > 1 and ssh != tp_rank:
                continue
            sb = int(sbases[ssh])
            srcs.append(sb + sli * stage_max * kv_bpt)
            dsts.append(d_kv + dli * remote_max * kv_bpt)
            lens.append(seq_len * kv_bpt)
            srcs.append(sb + s_pe + sli * stage_max * pe_bpt)
            dsts.append(d_pe + dli * remote_max * pe_bpt)
            lens.append(seq_len * pe_bpt)
            srcs.append(sb + s_ki + sli * _max_pages(stage_max) * KI_PAGE_BYTES)
            dsts.append(d_ki + dli * _max_pages(remote_max) * KI_PAGE_BYTES)
            lens.append(npages * KI_PAGE_BYTES)
        return (srcs, dsts, lens)

    def build_engine(self, model_weights_dir, max_seq_len, with_mtp, ar_steps, num_mtp=3):
        assert num_mtp in getattr(
            self, "supported_num_mtp", base.DEFAULT_SUPPORTED_NUM_MTP
        ), f"{self.name}: num_mtp={num_mtp} is not in this profile's supported set"
        return self._engine_factory(model_weights_dir, max_seq_len, with_mtp, ar_steps)


class MlaNsaEngineAdapter:

    def __init__(self, generator, with_mtp: bool):
        import torch as _torch

        self._torch = _torch
        self.gen = generator
        self.with_mtp = with_mtp
        self.mtp_seq_len = getattr(generator, "mtp_seq_len", 4)
        self.max_seq_len = getattr(generator.decode_layer, "max_seq_len", 200000)
        self.last_stats: dict = {}
        self.stop_ids = self._resolve_stop_ids(generator)
        self._ignore_eos = False

    @staticmethod
    def _resolve_stop_ids(generator) -> set:
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

    def prepare_grammar(self, grammar_spec, enable_thinking=True):
        if grammar_spec is None:
            return None
        raise GrammarUnsupported("constrained decoding is not supported")

    def supports_penalties(self) -> bool:
        return False

    def supports_ignore_eos(self) -> bool:
        return True

    def decode(
        self,
        first_token_id,
        max_tokens,
        sampling,
        on_token=None,
        cancel_event=None,
        grammar_session=None,
    ):
        sampling = sampling or {}
        rep = float(sampling.get("repetition_penalty", 1.0) or 1.0)
        pres = float(sampling.get("presence_penalty", 0.0) or 0.0)
        if rep != 1.0 or pres != 0.0:
            raise NotImplementedError(
                "repetition/presence penalties are not supported by this model's decode runtime"
            )
        temp = float(sampling.get("temperature", 1.0))
        if temp < 1e-05:
            self.gen.update_sampling_params(temperature=1.0, top_p=1.0, top_k=1, use_topp=False)
        else:
            self.gen.update_sampling_params(
                temperature=temp,
                top_p=resolve_top_p(sampling),
                top_k=resolve_top_k(sampling),
                use_topp=True,
            )
        self._ignore_eos = bool(sampling.get("ignore_eos"))
        budget = min(int(max_tokens), self.max_seq_len - self._seq_len - 1)
        if budget <= 0:
            self.last_stats = {"finish_reason": "length"}
            return [int(first_token_id)]
        if self.with_mtp:
            return self._decode_mtp(first_token_id, budget, on_token, cancel_event, grammar_session)
        return self._decode_standard(
            first_token_id, budget, on_token, cancel_event, grammar_session
        )

    def _decode_mtp(self, first_token_id, budget, on_token, cancel_event, grammar_session=None):
        dl = self.gen.decode_layer
        T = self.mtp_seq_len
        stop_ids = set() if self._ignore_eos else self.stop_ids
        torch = self._torch
        tokens = [int(first_token_id)]
        if on_token:
            on_token(int(first_token_id))
        if int(first_token_id) in stop_ids:
            self.last_stats = {"finish_reason": "stop"}
            return []
        finished = False
        if grammar_session is not None:
            try:
                if grammar_session.accept(int(first_token_id)) == "terminated":
                    finished = True
            except RuntimeError as e:
                raise GrammarViolationError(
                    f"prefill first token {first_token_id} violates the grammar"
                ) from e
        dl.set_prefill_valid_tokens(0)
        ar_steps = max(1, min(1024, int(os.environ.get("GLM5_AR_N", "8"))))
        ar_ok = hasattr(dl, "ar_accepted_tokens") and hasattr(dl, "ar_num_accepted")
        draft = torch.full((1, T), int(self._last_prompt_token), dtype=torch.int32, device="cuda:0")
        accepted, finish, fwd = ([], "length", 0)
        grammar_mask_written = False
        try:
            while not finished and len(tokens) < budget:
                if cancel_event is not None and cancel_event.is_set():
                    finish = "cancelled"
                    break
                if fwd == 1:
                    draft = torch.full(
                        (1, T), int(first_token_id), dtype=torch.int32, device="cuda:0"
                    )
                elif fwd > 1:
                    draft = dl.get_next_draft_tokens(0).reshape(1, T)
                if grammar_session is not None and fwd >= 1 and (not grammar_session.terminated):
                    chain = draft[0, 1:].cpu().tolist()
                    masks = grammar_session.fill_step_masks(chain)
                    if masks is not None:
                        dl.update_grammar_bitmask(masks)
                        grammar_mask_written = True
                if fwd == 0 or grammar_session is not None or (not ar_ok):
                    steps = 1
                else:
                    rem = budget - len(tokens)
                    steps = max(1, min(ar_steps, -(-rem // T)))
                if ar_ok:
                    dl.show_hands(draft, steps)
                    acc = dl.ar_accepted_tokens(0).cpu()[0]
                    num = dl.ar_num_accepted(0).cpu()[0]
                    n_tokens = int(acc[0].item())
                    n_steps = int(num[0].item())
                    emitted = acc[1 : 1 + n_tokens].tolist()
                    per_step = num[1 : 1 + n_steps].tolist()
                else:
                    dl.forward(draft)
                    n_acc = dl.get_num_accepted(0)
                    pred = dl.get_predicted_tokens(0).flatten()
                    emitted = [int(pred[i].item()) for i in range(n_acc)]
                    per_step = [n_acc]
                if fwd == 0:
                    fwd += 1
                    continue
                fwd += 1
                offset = 0
                for na in per_step:
                    step_emit = emitted[offset : offset + na]
                    offset += na
                    for tok in step_emit:
                        if len(tokens) >= budget:
                            break
                        tok = int(tok)
                        if tok in stop_ids:
                            finished = True
                            finish = "stop"
                            break
                        tokens.append(tok)
                        if on_token:
                            on_token(tok)
                        if (
                            grammar_session is not None
                            and grammar_session.accept(tok) == "terminated"
                        ):
                            finished = True
                            finish = "stop"
                            break
                    accepted.append(na)
                    if finished or len(tokens) >= budget:
                        break
        finally:
            if grammar_mask_written:
                dl.reset_grammar_bitmask()
            dl.reset_sequence()
        self.last_stats = {
            "finish_reason": finish,
            "mtp_accept_mean": round(sum(accepted) / max(1, len(accepted)), 3),
            "mtp_verify_calls": len(accepted),
        }
        return tokens

    def _decode_standard(
        self, first_token_id, budget, on_token, cancel_event, grammar_session=None
    ):
        dl = self.gen.decode_layer
        stop_ids = set() if self._ignore_eos else self.stop_ids
        torch = self._torch
        tokens = [int(first_token_id)]
        if on_token:
            on_token(int(first_token_id))
        if int(first_token_id) in stop_ids:
            self.last_stats = {"finish_reason": "stop"}
            return []
        finish = "length"
        finished = False
        grammar_mask_written = False
        ar_ok = hasattr(dl, "show_hands_no_mtp") and hasattr(dl, "ar_accepted_tokens_no_mtp")
        ar_steps = max(1, min(1024, int(os.environ.get("GLM5_AR_N", "8"))))
        cur = last_tok = None
        prev = None
        if ar_ok:
            dl.set_prefill_valid_tokens(0, with_mtp=False)
            last_tok = int(first_token_id)
            prev = torch.tensor([last_tok], dtype=torch.int32, device="cuda:0")
        else:
            cur = torch.tensor(int(first_token_id), dtype=torch.long, device="cuda:0")
        try:
            if grammar_session is not None:
                try:
                    if grammar_session.accept(int(first_token_id)) == "terminated":
                        finish, finished = ("stop", True)
                except RuntimeError as e:
                    raise GrammarViolationError(
                        f"prefill first token {first_token_id} violates the grammar"
                    ) from e
            while not finished and len(tokens) < budget:
                if cancel_event is not None and cancel_event.is_set():
                    finish = "cancelled"
                    break
                if (
                    grammar_session is not None
                    and grammar_session.active
                    and (not grammar_session.terminated)
                ):
                    masks = grammar_session.fill_step_masks([])
                    if masks is not None:
                        dl.update_grammar_bitmask(masks)
                        grammar_mask_written = True
                if ar_ok:
                    steps = (
                        1
                        if grammar_session is not None
                        else max(1, min(ar_steps, budget - len(tokens)))
                    )
                    dl.show_hands_no_mtp(prev, steps)
                    acc = dl.ar_accepted_tokens_no_mtp(0).cpu()[0]
                    n_tokens = int(acc[0].item())
                    emitted = acc[1 : 1 + n_tokens].tolist()
                else:
                    from tilert.models.deepseek_v3_2.temp_var_indices import Idx

                    res = dl.forward(cur)
                    intermediates, *_ = res[0]
                    nxt = intermediates[Idx.TOKEN_OUT][0][0]
                    emitted = [int(nxt.item())]
                    cur = nxt
                for tok in emitted:
                    if len(tokens) >= budget:
                        break
                    tok = int(tok)
                    if tok in stop_ids:
                        finished = True
                        finish = "stop"
                        break
                    tokens.append(tok)
                    last_tok = tok
                    if on_token:
                        on_token(tok)
                    if grammar_session is not None and grammar_session.accept(tok) == "terminated":
                        finished = True
                        finish = "stop"
                        break
                if ar_ok:
                    prev = torch.tensor([last_tok], dtype=torch.int32, device="cuda:0")
        finally:
            if grammar_mask_written:
                dl.reset_grammar_bitmask()
            dl.reset_sequence()
        self.last_stats = {"finish_reason": finish}
        return tokens

    def reset(self) -> None:
        pass
