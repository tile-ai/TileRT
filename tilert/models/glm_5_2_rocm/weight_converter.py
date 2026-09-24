"""GLM-5.2 weight converter: HF FP8 checkpoint -> device-sharded TileRT weights."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import zlib
from collections.abc import Callable
from contextlib import ExitStack

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from tilert import logger
from tilert.models.glm_5_2_rocm.checkpoint_config import (
    CONVERTER_VERSION,
    describe_hf_config,
    load_hf_config,
    sha256_file,
    validate_hf_config,
)
from tilert.models.glm_5_2_rocm.model_args import (
    KIND_DENSE,
    KIND_MOE_FULL,
    KIND_MOE_SHARED,
    ModelArgsGlm52,
    layer_kind,
)
from tilert.models.glm_5_2_rocm.ops.llm_preprocess import make_freqs_cis

__all__ = [
    "CheckpointReader",
    "Glm52WeightConverter",
    "load_rank_params",
    "random_rank_params",
    "selftest_swizzles",
]
FP8_MAX = 448.0
BLK = 128
N_BANK = 257
TP8_HEADS = 8


def _u8(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.uint8) if t.dtype == torch.float8_e4m3fn else t


def swizzle_fp8_contig8(w8: torch.Tensor) -> torch.Tensor:
    w8 = _u8(w8)
    *lead, rows, k = w8.shape
    assert rows % 16 == 0 and k % 64 == 0
    v = w8.reshape(*lead, rows // 16, 16, k // 64, 2, 4, 8)
    n = len(lead)
    perm = tuple(range(n)) + tuple(x + n for x in (0, 2, 4, 1, 3, 5))
    return v.permute(perm).reshape(*lead, rows * k).contiguous()


def swizzle_bf16_16x32(w: torch.Tensor) -> torch.Tensor:
    rows, k = w.shape
    assert rows % 16 == 0 and k % 32 == 0
    v = w.to(torch.bfloat16).view(torch.int16).reshape(rows // 16, 16, k // 32, 4, 8)
    return v.permute(0, 2, 3, 1, 4).reshape(-1).contiguous().view(torch.uint8)


def swizzle_wis_8x64(w: torch.Tensor) -> torch.Tensor:
    rows, k = w.shape
    assert rows % 8 == 0 and k % 64 == 0
    w16 = w.to(torch.bfloat16).view(torch.int16)
    rg = torch.arange(rows // 8)
    kc = torch.arange(k // 64)
    lane = torch.arange(64)
    i = torch.arange(8)
    RG, KC, L, II = torch.meshgrid(rg, kc, lane, i, indexing="ij")
    packed = w16[RG * 8 + L % 8, KC * 64 + L // 8 % 2 * 32 + L // 16 * 8 + II]
    return packed.reshape(-1).contiguous().view(torch.uint8)


def swizzle_fp8_m4(w8: torch.Tensor) -> torch.Tensor:
    w8 = _u8(w8)
    *lead, rows, k = w8.shape
    assert rows % 8 == 0 and k % 128 == 0
    v = w8.reshape(*lead, rows // 8, 2, 4, k // 128, 4, 8, 4)
    n = len(lead)
    perm = tuple(range(n)) + tuple(x + n for x in (0, 3, 1, 5, 2, 4, 6))
    return v.permute(perm).reshape(*lead, rows * k).contiguous()


def swizzle_fp8_v2(w8: torch.Tensor) -> torch.Tensor:
    from tilert.models.glm_5_2_rocm.ops.unprojo_allreduce import swizzle_v2

    return swizzle_v2(_u8(w8).view(torch.float8_e4m3fn))


def _pack_wo(w8: torch.Tensor) -> torch.Tensor:
    return swizzle_fp8_v2(w8)


def swizzle_bf16_m4(w: torch.Tensor) -> torch.Tensor:
    rows, k = w.shape
    assert rows % 8 == 0 and k % 64 == 0
    v = w.to(torch.bfloat16).view(torch.uint16).reshape(rows // 8, 2, 4, k // 64, 2, 8, 4)
    return v.permute(0, 3, 1, 5, 2, 4, 6).reshape(-1).contiguous().view(torch.uint8)


def pair_interleave(w8: torch.Tensor, inter: int) -> torch.Tensor:
    *lead, rows, k = w8.shape
    assert rows == 2 * inter and inter % 8 == 0
    v = w8.reshape(*lead, 2, inter // 8, 8, k)
    n = len(lead)
    perm = tuple(range(n)) + tuple(x + n for x in (1, 0, 2, 3))
    return v.permute(perm).reshape(*lead, rows, k)


def swizzle_pair_interleaved(w8: torch.Tensor, inter: int) -> torch.Tensor:
    return swizzle_fp8_contig8(pair_interleave(_u8(w8), inter))


def quantize_fp8_block(
    w: torch.Tensor, blk_k: int = BLK, blk_m: int = BLK
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, k = w.shape
    assert rows % blk_m == 0 and k % blk_k == 0
    blocks = w.float().view(rows // blk_m, blk_m, k // blk_k, blk_k)
    amax = blocks.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-12)
    scales = amax / FP8_MAX
    q = (blocks / scales).to(torch.float8_e4m3fn)
    return (q.reshape(rows, k).contiguous(), scales.view(rows // blk_m, k // blk_k).contiguous())


def quantize_fp8_block_padded(
    w: torch.Tensor, blk_k: int = BLK, blk_m: int = BLK
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, k = w.shape
    if rows % blk_m == 0:
        return quantize_fp8_block(w, blk_k, blk_m)
    n_blk = -(-rows // blk_m)
    padded = torch.zeros(n_blk * blk_m, k, dtype=torch.float32)
    padded[:rows] = w.float()
    q, scales = quantize_fp8_block(padded, blk_k, blk_m)
    return (q[:rows].contiguous(), scales)


def dequant_fp8(
    w8: torch.Tensor, scale_inv: torch.Tensor, blk_m: int = BLK, blk_k: int = BLK
) -> torch.Tensor:
    rows, k = w8.shape
    s = scale_inv.float()
    s = s.repeat_interleave(blk_m, dim=0)[:rows]
    s = s.repeat_interleave(blk_k, dim=1)[:, :k]
    return _u8(w8).view(torch.float8_e4m3fn).float() * s


ATTN_SCALE_BLK_M = 64
M4_SCALE_BLK_K = 64
M6_SCALE_BLK_K = 128


def attn_lossless_slices(
    qb8: torch.Tensor,
    qb_s: torch.Tensor,
    kvb8: torch.Tensor,
    kvb_s: torch.Tensor,
    h0: int,
    hv: int,
    heads: int,
    args: ModelArgsGlm52,
) -> dict[str, torch.Tensor]:
    nope, rope, vdim, kvr, qlr = (
        args.qk_nope_head_dim,
        args.qk_rope_head_dim,
        args.v_head_dim,
        args.kv_lora_rank,
        args.q_lora_rank,
    )
    qk = nope + rope
    kvd = nope + vdim
    sm = ATTN_SCALE_BLK_M
    qb8 = _u8(qb8).view(args.n_heads, qk, qlr)
    kvb8 = _u8(kvb8).view(args.n_heads, kvd, kvr)
    qb_s, kvb_s = (qb_s.float(), kvb_s.float())
    assert qb_s.shape == (args.n_heads * qk // BLK, qlr // BLK), qb_s.shape
    assert kvb_s.shape == (args.n_heads * kvd // BLK, kvr // BLK), kvb_s.shape
    hh = torch.arange(hv) + h0
    out: dict[str, torch.Tensor] = {}
    wqb = torch.zeros(heads, qk, qlr, dtype=torch.uint8)
    wqb[:hv] = qb8[h0 : h0 + hv]
    out["wqb"] = (
        torch.cat(
            [
                wqb[:, :nope, :].reshape(heads * nope, qlr),
                wqb[:, nope:, :].reshape(heads * rope, qlr),
            ]
        )
        .contiguous()
        .view(torch.float8_e4m3fn)
    )
    qs = torch.ones(heads * qk // sm, qlr // BLK, dtype=torch.float32)
    nope_stripes = nope // sm
    for si in range(nope_stripes):
        blk = (hh * qk + si * sm) // BLK
        qs[torch.arange(hv) * nope_stripes + si] = qb_s[blk]
    pe_base = heads * nope // sm
    for si in range(rope // sm):
        blk = (hh * qk + nope + si * sm) // BLK
        qs[pe_base + torch.arange(hv) * (rope // sm) + si] = qb_s[blk]
    out["wqb_scales"] = qs.contiguous()
    wkvb1 = torch.zeros(heads, kvr, nope, dtype=torch.uint8)
    wkvb1[:hv] = kvb8[h0 : h0 + hv, :nope, :].transpose(-1, -2)
    out["wkvb1"] = wkvb1.reshape(heads * kvr, nope).contiguous().view(torch.float8_e4m3fn)
    k1s = torch.ones(heads * kvr // sm, nope // M4_SCALE_BLK_K, dtype=torch.float32)
    c_stripes = kvr // sm
    for kb in range(nope // M4_SCALE_BLK_K):
        rblk = (hh * kvd + kb * M4_SCALE_BLK_K) // BLK
        for t in range(c_stripes):
            cblk = t * sm // BLK
            k1s[torch.arange(hv) * c_stripes + t, kb] = kvb_s[rblk, cblk]
    out["wkvb1_scales"] = k1s.contiguous()
    wkvb2 = torch.zeros(heads, vdim, kvr, dtype=torch.uint8)
    wkvb2[:hv] = kvb8[h0 : h0 + hv, nope:, :]
    out["wkvb2"] = wkvb2.reshape(heads * vdim, kvr).contiguous().view(torch.float8_e4m3fn)
    k2s = torch.ones(heads * vdim // sm, kvr // M6_SCALE_BLK_K, dtype=torch.float32)
    v_stripes = vdim // sm
    for si in range(v_stripes):
        rblk = (hh * kvd + nope + si * sm) // BLK
        k2s[torch.arange(hv) * v_stripes + si] = kvb_s[rblk]
    out["wkvb2_scales"] = k2s.contiguous()
    return out


class CheckpointReader:
    """safetensors index + lazily opened shard handles, with a layer cache."""

    def __init__(self, model_dir: str) -> None:
        self.model_dir = model_dir
        idx = os.path.join(model_dir, "model.safetensors.index.json")
        with open(idx) as f:
            self.weight_map: dict[str, str] = json.load(f)["weight_map"]
        self._stack = ExitStack()
        self._handles: dict[str, object] = {}
        self._cache: dict[str, torch.Tensor] | None = None

    def begin_layer(self) -> None:
        self._cache = {}

    def end_layer(self) -> None:
        self._cache = None

    def get(self, key: str) -> torch.Tensor:
        if self._cache is not None and key in self._cache:
            return self._cache[key]
        shard = self.weight_map[key]
        if shard not in self._handles:
            self._handles[shard] = self._stack.enter_context(
                safe_open(os.path.join(self.model_dir, shard), framework="pt")
            )
        t = self._handles[shard].get_tensor(key)
        if self._cache is not None:
            self._cache[key] = t
        return t

    def close(self) -> None:
        self._stack.close()
        self._handles.clear()
        self._cache = None


def _gamma(t: torch.Tensor) -> torch.Tensor:
    return t.float().contiguous()


def _head_slice(rank: int, args: ModelArgsGlm52) -> tuple[int, int]:
    h0 = (rank - 1) * args.local_heads
    return (h0, min(args.n_heads, h0 + args.local_heads) - h0)


def shard_mla_layer(
    rd: CheckpointReader,
    pre: str,
    rank: int,
    args: ModelArgsGlm52,
    work_dev: str,
    tp8: bool = False,
) -> dict[str, torch.Tensor]:
    if tp8:
        h0, hv, heads = (rank * TP8_HEADS, TP8_HEADS, TP8_HEADS)
    else:
        h0, hv = _head_slice(rank, args)
        heads = args.local_heads
    vdim = args.v_head_dim
    out: dict[str, torch.Tensor] = {}
    out["in_gamma"] = _gamma(rd.get(f"{pre}input_layernorm.weight"))
    out["wqkva"] = torch.cat(
        [
            rd.get(f"{pre}self_attn.q_a_proj.weight"),
            rd.get(f"{pre}self_attn.kv_a_proj_with_mqa.weight"),
        ]
    )
    out["wqkva_scales"] = torch.cat(
        [
            rd.get(f"{pre}self_attn.q_a_proj.weight_scale_inv").float(),
            rd.get(f"{pre}self_attn.kv_a_proj_with_mqa.weight_scale_inv").float(),
        ]
    ).contiguous()
    out["q_gamma"] = _gamma(rd.get(f"{pre}self_attn.q_a_layernorm.weight"))
    out.update(
        attn_lossless_slices(
            rd.get(f"{pre}self_attn.q_b_proj.weight"),
            rd.get(f"{pre}self_attn.q_b_proj.weight_scale_inv"),
            rd.get(f"{pre}self_attn.kv_b_proj.weight"),
            rd.get(f"{pre}self_attn.kv_b_proj.weight_scale_inv"),
            h0,
            hv,
            heads,
            args,
        )
    )
    out["kv_gamma"] = _gamma(rd.get(f"{pre}self_attn.kv_a_layernorm.weight"))
    wo8 = rd.get(f"{pre}self_attn.o_proj.weight")
    wo_si = rd.get(f"{pre}self_attn.o_proj.weight_scale_inv").float()
    wo = torch.zeros(args.dim, heads * vdim, dtype=torch.uint8)
    wo[:, : hv * vdim] = _u8(wo8[:, h0 * vdim : (h0 + hv) * vdim])
    out["wo"] = wo.view(torch.float8_e4m3fn)
    wos = torch.ones(args.dim // BLK, heads * vdim // BLK, dtype=torch.float32)
    wos[:, : hv * vdim // BLK] = wo_si[:, h0 * vdim // BLK : (h0 + hv) * vdim // BLK]
    out["wo_scales"] = wos.contiguous()
    return out


def shard_indexer_layer(
    rd: CheckpointReader, pre: str, args: ModelArgsGlm52
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    out["in_gamma"] = _gamma(rd.get(f"{pre}input_layernorm.weight"))
    out["wqaki"] = torch.cat(
        [rd.get(f"{pre}self_attn.q_a_proj.weight"), rd.get(f"{pre}self_attn.indexer.wk.weight")]
    )
    out["wqaki_scales"] = torch.cat(
        [
            rd.get(f"{pre}self_attn.q_a_proj.weight_scale_inv").float(),
            rd.get(f"{pre}self_attn.indexer.wk.weight_scale_inv").float(),
        ]
    ).contiguous()
    out["wis"] = (
        rd.get(f"{pre}self_attn.indexer.weights_proj.weight").to(torch.bfloat16).contiguous()
    )
    out["q_gamma"] = _gamma(rd.get(f"{pre}self_attn.q_a_layernorm.weight"))
    out["wqi"] = rd.get(f"{pre}self_attn.indexer.wq_b.weight")
    out["wqi_scales"] = rd.get(f"{pre}self_attn.indexer.wq_b.weight_scale_inv").float().contiguous()
    out["knorm_w"] = _gamma(rd.get(f"{pre}self_attn.indexer.k_norm.weight"))
    out["knorm_b"] = _gamma(rd.get(f"{pre}self_attn.indexer.k_norm.bias"))
    return out


def shard_dense_ffn(
    rd: CheckpointReader, pre: str, rank: int, args: ModelArgsGlm52
) -> dict[str, torch.Tensor]:
    inter = args.dense_inter_shard
    r0, sb, nb = (rank * inter, rank * inter // BLK, inter // BLK)
    out: dict[str, torch.Tensor] = {}
    out["post_gamma"] = _gamma(rd.get(f"{pre}post_attention_layernorm.weight"))
    out["wug"] = torch.cat(
        [
            rd.get(f"{pre}mlp.gate_proj.weight")[r0 : r0 + inter],
            rd.get(f"{pre}mlp.up_proj.weight")[r0 : r0 + inter],
        ]
    )
    out["wug_scales"] = torch.cat(
        [
            rd.get(f"{pre}mlp.gate_proj.weight_scale_inv").float()[sb : sb + nb],
            rd.get(f"{pre}mlp.up_proj.weight_scale_inv").float()[sb : sb + nb],
        ]
    ).contiguous()
    out["wdown"] = rd.get(f"{pre}mlp.down_proj.weight")[:, r0 : r0 + inter].contiguous()
    out["wdown_scales"] = (
        rd.get(f"{pre}mlp.down_proj.weight_scale_inv").float()[:, sb : sb + nb].contiguous()
    )
    return out


def shard_moe_ffn(
    rd: CheckpointReader, pre: str, rank: int, args: ModelArgsGlm52
) -> dict[str, torch.Tensor]:
    inter = args.moe_inter_shard
    r0, sb, nb = (rank * inter, rank * inter // BLK, inter // BLK)

    def ep(e: int) -> str:
        return f"{pre}mlp.shared_experts." if e == 0 else f"{pre}mlp.experts.{e - 1}."

    out: dict[str, torch.Tensor] = {}
    out["post_gamma"] = _gamma(rd.get(f"{pre}post_attention_layernorm.weight"))
    out["router"] = rd.get(f"{pre}mlp.gate.weight").to(torch.bfloat16).contiguous()
    out["moe_bias"] = _gamma(rd.get(f"{pre}mlp.gate.e_score_correction_bias"))
    ug = torch.empty(N_BANK, 2 * inter, args.dim, dtype=torch.uint8)
    ug_sc = torch.empty(N_BANK, 2 * nb, args.dim // BLK, dtype=torch.float32)
    dn = torch.empty(N_BANK, args.dim, inter, dtype=torch.uint8)
    dn_sc = torch.empty(N_BANK, args.dim // BLK, nb, dtype=torch.float32)
    for e in range(N_BANK):
        p = ep(e)
        ug[e, :inter] = _u8(rd.get(f"{p}gate_proj.weight")[r0 : r0 + inter])
        ug[e, inter:] = _u8(rd.get(f"{p}up_proj.weight")[r0 : r0 + inter])
        ug_sc[e, :nb] = rd.get(f"{p}gate_proj.weight_scale_inv").float()[sb : sb + nb]
        ug_sc[e, nb:] = rd.get(f"{p}up_proj.weight_scale_inv").float()[sb : sb + nb]
        dn[e] = _u8(rd.get(f"{p}down_proj.weight")[:, r0 : r0 + inter])
        dn_sc[e] = rd.get(f"{p}down_proj.weight_scale_inv").float()[:, sb : sb + nb]
    out["moe_ug"] = ug.view(torch.float8_e4m3fn)
    out["moe_ug_scales"] = ug_sc
    out["moe_down"] = dn.view(torch.float8_e4m3fn)
    out["moe_down_scales"] = dn_sc
    return out


def shard_layer(
    rd: CheckpointReader,
    i: int,
    rank: int,
    args: ModelArgsGlm52,
    work_dev: str,
    attn_tp8: bool = False,
) -> dict[str, torch.Tensor]:
    pre = f"model.layers.{i}."
    kind = layer_kind(i)
    out: dict[str, torch.Tensor] = {}
    if rank == 0:
        if kind != KIND_MOE_SHARED:
            out.update(shard_indexer_layer(rd, pre, args))
    else:
        out.update(shard_mla_layer(rd, pre, rank, args, work_dev))
    if attn_tp8 and kind == KIND_MOE_SHARED:
        for k, v in shard_mla_layer(rd, pre, rank, args, work_dev, tp8=True).items():
            out[f"attn_tp8.{k}"] = v
    if kind == KIND_DENSE:
        out.update(shard_dense_ffn(rd, pre, rank, args))
    else:
        out.update(shard_moe_ffn(rd, pre, rank, args))
    return out


def shard_tail(rd: CheckpointReader, rank: int, args: ModelArgsGlm52) -> dict[str, torch.Tensor]:
    vs = args.vocab_shard
    return {
        "final_gamma": _gamma(rd.get("model.norm.weight")),
        "head": (
            rd.get("lm_head.weight")[rank * vs : (rank + 1) * vs].to(torch.bfloat16).contiguous()
        ),
    }


def shard_mtp(
    rd: CheckpointReader, rank: int, args: ModelArgsGlm52, work_dev: str
) -> dict[str, torch.Tensor]:
    i = args.n_layers
    pre = f"model.layers.{i}."
    inter = args.dim // 4
    half, sl = (1, rank - 4) if rank >= 4 else (0, rank)
    c0 = half * args.dim + sl * inter
    eh = rd.get(f"{pre}eh_proj.weight").to(torch.bfloat16)
    out: dict[str, torch.Tensor] = {
        "mtp.eh_w": eh[:, c0 : c0 + inter].contiguous(),
        "mtp.e_gamma": _gamma(rd.get(f"{pre}enorm.weight")),
        "mtp.h_gamma": _gamma(rd.get(f"{pre}hnorm.weight")),
        "mtp.head_gamma": _gamma(rd.get(f"{pre}shared_head.norm.weight")),
    }
    for k, v in shard_layer(rd, i, rank, args, work_dev).items():
        out[f"mtp_layer.{k}"] = v
    return out


class _ShardWriter:
    """Accumulate tensors, flush ~shard_bytes files, write an index at close."""

    def __init__(self, out_dir: str, shard_bytes: int = 8 << 30) -> None:
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.shard_bytes = shard_bytes
        self.pending: dict[str, torch.Tensor] = {}
        self.pending_bytes = 0
        self.files: list[dict[str, torch.Tensor]] = []
        self.weight_map: dict[str, str] = {}
        self.n_files = 0
        self.total = 0

    def add(self, name: str, t: torch.Tensor) -> None:
        self.pending[name] = t
        self.pending_bytes += t.numel() * t.element_size()
        if self.pending_bytes >= self.shard_bytes:
            self.flush()

    def flush(self) -> None:
        if not self.pending:
            return
        self.n_files += 1
        fname = f"model-{self.n_files:05d}.safetensors"
        save_file(self.pending, os.path.join(self.out_dir, fname))
        for k, v in self.pending.items():
            self.weight_map[k] = fname
            self.total += v.numel() * v.element_size()
        self.pending = {}
        self.pending_bytes = 0

    def close(self) -> None:
        self.flush()
        index = {"metadata": {"total_size": self.total}, "weight_map": self.weight_map}
        with open(os.path.join(self.out_dir, "model.safetensors.index.json"), "w") as f:
            json.dump(index, f, indent=1)


class Glm52WeightConverter:
    """Offline stage-1 driver: HF checkpoint -> save_dir/{shared,rank0..7}."""

    _PASSTHROUGH = (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "config.json",
        "generation_config.json",
    )

    def __init__(
        self,
        model_dir: str,
        save_dir: str,
        args: ModelArgsGlm52 | None = None,
        n_layers: int | None = None,
        work_dev: str = "cpu",
        num_mtp: int | None = None,
        attn_tp8: bool = True,
    ) -> None:
        self.args = args or ModelArgsGlm52()
        self.model_dir = model_dir
        self.save_dir = save_dir
        self.n_layers = self.args.n_layers if n_layers is None else n_layers
        self.work_dev = work_dev
        self.num_mtp = self.args.num_mtp if num_mtp is None else num_mtp
        self.hf_config = load_hf_config(model_dir)
        validate_hf_config(self.hf_config, self.args, n_layers=self.n_layers, num_mtp=self.num_mtp)
        self.rd = CheckpointReader(model_dir)
        self.attn_tp8 = attn_tp8

    def convert(self) -> None:
        args = self.args
        writers = [
            _ShardWriter(os.path.join(self.save_dir, f"rank{r}")) for r in range(args.num_devices)
        ]
        for i in range(self.n_layers):
            t0 = time.time()
            self.rd.begin_layer()
            for r in range(args.num_devices):
                shard = shard_layer(self.rd, i, r, args, self.work_dev, attn_tp8=self.attn_tp8)
                for k, v in shard.items():
                    writers[r].add(f"layer_{i}.{k}", v)
            self.rd.end_layer()
            logger.info("converted layer %d/%d (%.1fs)", i + 1, self.n_layers, time.time() - t0)
        self.rd.begin_layer()
        for r in range(args.num_devices):
            for k, v in shard_tail(self.rd, r, args).items():
                writers[r].add(k, v)
            if self.num_mtp > 0:
                for k, v in shard_mtp(self.rd, r, args, self.work_dev).items():
                    writers[r].add(k, v)
            writers[r].close()
        self.rd.end_layer()
        if self.num_mtp > 0:
            logger.info("converted the MTP module (layer %d)", args.n_layers)
        shared = os.path.join(self.save_dir, "shared")
        os.makedirs(shared, exist_ok=True)
        save_file(
            {"embed": self.rd.get("model.embed_tokens.weight").to(torch.bfloat16).contiguous()},
            os.path.join(shared, "embed.safetensors"),
        )
        for name in self._PASSTHROUGH:
            src = os.path.join(self.model_dir, name)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(shared, name))
        with open(os.path.join(self.save_dir, "tilert_meta.json"), "w") as f:
            json.dump(
                {
                    "model": "glm_5_2",
                    "n_layers": self.n_layers,
                    "num_mtp": self.num_mtp,
                    "tp": 8,
                    "attn_tp8": self.attn_tp8,
                    "attn_fp8_lossless": True,
                    "converter_version": CONVERTER_VERSION,
                    "source_model_dir": os.path.abspath(self.model_dir),
                    "source_config_sha256": sha256_file(
                        os.path.join(self.model_dir, "config.json")
                    ),
                    "source_index_sha256": sha256_file(
                        os.path.join(self.model_dir, "model.safetensors.index.json")
                    ),
                    "kv_dtype": "bf16",
                    "weight_format": "fp8_e4m3_block128; attn 64-row scale stripes",
                    "hf_config": describe_hf_config(self.hf_config),
                },
                f,
                indent=1,
            )
        self.rd.close()
        logger.info(f"device-sharded weights written to {self.save_dir}")


def _write_json_atomic(path: str, obj: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def _provenance_fields(model_dir: str) -> dict[str, str]:
    return {
        "source_model_dir": os.path.abspath(model_dir),
        "source_config_sha256": sha256_file(os.path.join(model_dir, "config.json")),
        "source_index_sha256": sha256_file(os.path.join(model_dir, "model.safetensors.index.json")),
    }


def _stored_tensor(save_dir: str, rank: int, key: str) -> torch.Tensor:
    rank_dir = os.path.join(save_dir, f"rank{rank}")
    with open(os.path.join(rank_dir, "model.safetensors.index.json")) as f:
        wm = json.load(f)["weight_map"]
    if key not in wm:
        raise KeyError(f"rank{rank}: {key} is not in the conversion's index")
    with safe_open(os.path.join(rank_dir, wm[key]), framework="pt") as f:
        return f.get_tensor(key)


def bind_source_checkpoint(
    rd: CheckpointReader,
    model_dir: str,
    save_dir: str,
    meta: dict,
    args: ModelArgsGlm52 | None = None,
) -> dict[str, str]:
    args = args or ModelArgsGlm52()
    prov = _provenance_fields(model_dir)
    for key in ("source_config_sha256", "source_index_sha256"):
        if meta.get(key) and meta[key] != prov[key]:
            raise ValueError(
                f"{save_dir} was cut from a checkpoint whose {key[7:-7]} differs from {model_dir}'s ({key}: {meta[key][:12]}.. vs {prov[key][:12]}..); refusing to mix checkpoints"
            )
    shared_cfg = os.path.join(save_dir, "shared", "config.json")
    if os.path.isfile(shared_cfg) and sha256_file(shared_cfg) != prov["source_config_sha256"]:
        raise ValueError(
            f"{save_dir}/shared/config.json differs from {model_dir}/config.json; refusing to mix checkpoints"
        )

    def same(a: torch.Tensor, b: torch.Tensor) -> bool:
        if a.dtype != b.dtype or tuple(a.shape) != tuple(b.shape):
            return False
        return torch.equal(a.contiguous(), b.contiguous())

    rd.begin_layer()
    try:
        for r in (0, args.num_devices - 1):
            want = shard_tail(rd, r, args)
            for k, v in want.items():
                got = _stored_tensor(save_dir, r, k)
                if not same(got, v):
                    raise ValueError(
                        f"{save_dir} rank{r} {k} != the slice of {model_dir} ({got.dtype}{tuple(got.shape)} vs {v.dtype}{tuple(v.shape)}): the conversion was cut from a different checkpoint; refusing to mix them"
                    )
        with safe_open(os.path.join(save_dir, "shared", "embed.safetensors"), framework="pt") as f:
            emb = f.get_tensor("embed")
        if not same(emb, rd.get("model.embed_tokens.weight").to(torch.bfloat16)):
            raise ValueError(
                f"{save_dir}/shared/embed.safetensors != {model_dir}'s embed_tokens: the conversion was cut from a different checkpoint"
            )
    finally:
        rd.end_layer()
    logger.info("%s is bound to %s (tail + embed bit-exact)", save_dir, model_dir)
    return prov


def stamp_provenance(model_dir: str, save_dir: str, args: ModelArgsGlm52 | None = None) -> None:
    args = args or ModelArgsGlm52()
    meta_path = os.path.join(save_dir, "tilert_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    if not meta.get("attn_fp8_lossless"):
        raise RuntimeError(
            f"{save_dir} still holds re-quantized attention tensors; run --augment-attn-lossless first (converter format {CONVERTER_VERSION} is the lossless one)"
        )
    hf_cfg = load_hf_config(model_dir)
    validate_hf_config(
        hf_cfg,
        args,
        n_layers=int(meta.get("n_layers", args.n_layers)),
        num_mtp=int(meta.get("num_mtp", 0)),
        max_seq_len=1,
    )
    rd = CheckpointReader(model_dir)
    try:
        prov = bind_source_checkpoint(rd, model_dir, save_dir, meta, args)
    finally:
        rd.close()
    meta.update(prov)
    meta.setdefault("model", "glm_5_2")
    meta.setdefault("tp", 8)
    meta["converter_version"] = CONVERTER_VERSION
    meta.setdefault("kv_dtype", "bf16")
    meta.setdefault("weight_format", "fp8_e4m3_block128; attn 64-row scale stripes")
    meta["hf_config"] = describe_hf_config(hf_cfg)
    _write_json_atomic(meta_path, meta)
    logger.info("provenance stamped: %s <- %s", save_dir, model_dir)


def augment_attn_tp8(
    model_dir: str,
    save_dir: str,
    args: ModelArgsGlm52 | None = None,
    work_dev: str = "cpu",
    store_dir: str | None = None,
) -> None:
    args = args or ModelArgsGlm52()
    meta_path = os.path.join(save_dir, "tilert_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    if meta.get("attn_tp8"):
        logger.info("%s already carries the attn_tp8 set; nothing to do", save_dir)
        return
    n_layers = meta.get("n_layers", args.n_layers)
    shared = [i for i in range(n_layers) if layer_kind(i) == KIND_MOE_SHARED]
    indexes = []
    for r in range(args.num_devices):
        with open(os.path.join(save_dir, f"rank{r}", "model.safetensors.index.json")) as f:
            indexes.append(json.load(f))
    rd = CheckpointReader(model_dir)
    prov = bind_source_checkpoint(rd, model_dir, save_dir, meta, args)
    pending: list[dict[str, torch.Tensor]] = [{} for _ in range(args.num_devices)]
    for n, i in enumerate(shared):
        rd.begin_layer()
        for r in range(args.num_devices):
            tp8 = shard_mla_layer(rd, f"model.layers.{i}.", r, args, work_dev, tp8=True)
            for k, v in tp8.items():
                pending[r][f"layer_{i}.attn_tp8.{k}"] = v
        rd.end_layer()
        logger.info("augmented layer %d (%d/%d shared)", i, n + 1, len(shared))
    rd.close()
    fname = "attn-tp8-00001.safetensors"
    for r in range(args.num_devices):
        rank_dir = os.path.join(save_dir, f"rank{r}")
        if store_dir is not None:
            real_dir = os.path.join(store_dir, f"rank{r}")
            os.makedirs(real_dir, exist_ok=True)
            real = os.path.abspath(os.path.join(real_dir, fname))
            save_file(pending[r], real)
            link = os.path.join(rank_dir, fname)
            if os.path.islink(link) or os.path.exists(link):
                os.remove(link)
            os.symlink(real, link)
        else:
            save_file(pending[r], os.path.join(rank_dir, fname))
        idx = indexes[r]
        for k, v in pending[r].items():
            if k not in idx["weight_map"]:
                idx["metadata"]["total_size"] += v.numel() * v.element_size()
            idx["weight_map"][k] = fname
        _write_json_atomic(os.path.join(rank_dir, "model.safetensors.index.json"), idx)
        logger.info("rank %d: %d attn_tp8 tensors appended", r, len(pending[r]))
    meta["attn_tp8"] = True
    meta.update(prov)
    _write_json_atomic(meta_path, meta)
    logger.info("attn_tp8 augment complete: %s", save_dir)


ATTN_LOSSLESS_FILE = "attn-lossless-00001.safetensors"
ATTN_LOSSLESS_KEYS = ("wqb", "wqb_scales", "wkvb1", "wkvb1_scales", "wkvb2", "wkvb2_scales")


def _attn_lossless_targets(
    meta: dict, args: ModelArgsGlm52
) -> list[tuple[str, str, int, list[int], bool]]:
    n_layers = meta.get("n_layers", args.n_layers)
    tp7_ranks = list(range(1, args.num_devices))
    all_ranks = list(range(args.num_devices))
    out = []
    for i in range(n_layers):
        pre = f"model.layers.{i}."
        out.append((pre, f"layer_{i}.", i, tp7_ranks, False))
        if meta.get("attn_tp8") and layer_kind(i) == KIND_MOE_SHARED:
            out.append((pre, f"layer_{i}.attn_tp8.", i, all_ranks, True))
    if meta.get("num_mtp", 0) > 0:
        i = args.n_layers
        out.append((f"model.layers.{i}.", "mtp_layer.", i, tp7_ranks, False))
    return out


def _rank_geometry(r: int, tp8: bool, args: ModelArgsGlm52) -> tuple[int, int, int]:
    if tp8:
        return (r * TP8_HEADS, TP8_HEADS, TP8_HEADS)
    h0, hv = _head_slice(r, args)
    return (h0, hv, args.local_heads)


def augment_attn_lossless(
    model_dir: str, save_dir: str, args: ModelArgsGlm52 | None = None, store_dir: str | None = None
) -> None:
    args = args or ModelArgsGlm52()
    meta_path = os.path.join(save_dir, "tilert_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    if meta.get("attn_fp8_lossless"):
        logger.info("%s already holds lossless attention slices; nothing to do", save_dir)
        return
    targets = _attn_lossless_targets(meta, args)
    indexes = []
    for r in range(args.num_devices):
        with open(os.path.join(save_dir, f"rank{r}", "model.safetensors.index.json")) as f:
            indexes.append(json.load(f))
    rd = CheckpointReader(model_dir)
    prov = bind_source_checkpoint(rd, model_dir, save_dir, meta, args)
    pending: list[dict[str, torch.Tensor]] = [{} for _ in range(args.num_devices)]
    last_layer = None
    for n, (pre, kp, i, ranks, tp8) in enumerate(targets):
        if i != last_layer:
            rd.end_layer()
            rd.begin_layer()
            last_layer = i
        qb8 = rd.get(f"{pre}self_attn.q_b_proj.weight")
        qb_s = rd.get(f"{pre}self_attn.q_b_proj.weight_scale_inv")
        kvb8 = rd.get(f"{pre}self_attn.kv_b_proj.weight")
        kvb_s = rd.get(f"{pre}self_attn.kv_b_proj.weight_scale_inv")
        for r in ranks:
            h0, hv, heads = _rank_geometry(r, tp8, args)
            sl = attn_lossless_slices(qb8, qb_s, kvb8, kvb_s, h0, hv, heads, args)
            for k in ATTN_LOSSLESS_KEYS:
                key = f"{kp}{k}"
                if key not in indexes[r]["weight_map"]:
                    raise KeyError(f"rank{r}: {key} is not in the conversion's index")
                pending[r][key] = sl[k]
        logger.info("lossless attention set %d/%d (%s)", n + 1, len(targets), kp)
    rd.close()
    for r in range(args.num_devices):
        rank_dir = os.path.join(save_dir, f"rank{r}")
        if store_dir is not None:
            real_dir = os.path.join(store_dir, f"rank{r}")
            os.makedirs(real_dir, exist_ok=True)
            real = os.path.abspath(os.path.join(real_dir, ATTN_LOSSLESS_FILE))
            save_file(pending[r], real)
            link = os.path.join(rank_dir, ATTN_LOSSLESS_FILE)
            if os.path.islink(link) or os.path.exists(link):
                os.remove(link)
            os.symlink(real, link)
        else:
            save_file(pending[r], os.path.join(rank_dir, ATTN_LOSSLESS_FILE))
        idx = indexes[r]
        for k in pending[r]:
            idx["weight_map"][k] = ATTN_LOSSLESS_FILE
        _write_json_atomic(os.path.join(rank_dir, "model.safetensors.index.json"), idx)
        logger.info("rank %d: %d attention tensors repointed", r, len(pending[r]))
    meta["attn_fp8_lossless"] = True
    meta.update(prov)
    _write_json_atomic(meta_path, meta)
    logger.info("attn lossless augment complete: %s", save_dir)


def verify_attn_lossless(
    model_dir: str, save_dir: str, layers: list[int], args: ModelArgsGlm52 | None = None
) -> None:
    args = args or ModelArgsGlm52()
    with open(os.path.join(save_dir, "tilert_meta.json")) as f:
        meta = json.load(f)
    if not meta.get("attn_fp8_lossless"):
        raise RuntimeError(f"{save_dir} has no lossless attention set to verify")
    nope, rope, vdim, kvr = (
        args.qk_nope_head_dim,
        args.qk_rope_head_dim,
        args.v_head_dim,
        args.kv_lora_rank,
    )
    rd = CheckpointReader(model_dir)
    loaders: dict[int, tuple[str, dict]] = {}

    def conv(r: int, key: str) -> torch.Tensor:
        if r not in loaders:
            rank_dir = os.path.join(save_dir, f"rank{r}")
            with open(os.path.join(rank_dir, "model.safetensors.index.json")) as f:
                loaders[r] = (rank_dir, json.load(f)["weight_map"])
        rank_dir, wm = loaders[r]
        with safe_open(os.path.join(rank_dir, wm[key]), framework="pt") as f:
            return f.get_tensor(key)

    n_checked = 0
    for pre, kp, i, ranks, tp8 in _attn_lossless_targets(meta, args):
        if i not in layers:
            continue
        rd.begin_layer()
        qb = dequant_fp8(
            rd.get(f"{pre}self_attn.q_b_proj.weight"),
            rd.get(f"{pre}self_attn.q_b_proj.weight_scale_inv"),
        ).view(args.n_heads, nope + rope, -1)
        kvb = dequant_fp8(
            rd.get(f"{pre}self_attn.kv_b_proj.weight"),
            rd.get(f"{pre}self_attn.kv_b_proj.weight_scale_inv"),
        ).view(args.n_heads, nope + vdim, kvr)
        for r in ranks:
            h0, hv, heads = _rank_geometry(r, tp8, args)
            got = dequant_fp8(conv(r, f"{kp}wqb"), conv(r, f"{kp}wqb_scales"), ATTN_SCALE_BLK_M)
            ref = torch.zeros(heads, nope + rope, qb.shape[-1])
            ref[:hv] = qb[h0 : h0 + hv]
            ref = torch.cat(
                [ref[:, :nope].reshape(heads * nope, -1), ref[:, nope:].reshape(heads * rope, -1)]
            )
            assert torch.equal(got, ref), f"{kp}wqb rank{r} differs from the checkpoint"
            got = dequant_fp8(
                conv(r, f"{kp}wkvb1"),
                conv(r, f"{kp}wkvb1_scales"),
                ATTN_SCALE_BLK_M,
                M4_SCALE_BLK_K,
            )
            ref = torch.zeros(heads, kvr, nope)
            ref[:hv] = kvb[h0 : h0 + hv, :nope, :].transpose(-1, -2)
            assert torch.equal(got, ref.reshape(heads * kvr, nope)), f"{kp}wkvb1 rank{r}"
            got = dequant_fp8(
                conv(r, f"{kp}wkvb2"),
                conv(r, f"{kp}wkvb2_scales"),
                ATTN_SCALE_BLK_M,
                M6_SCALE_BLK_K,
            )
            ref = torch.zeros(heads, vdim, kvr)
            ref[:hv] = kvb[h0 : h0 + hv, nope:, :]
            assert torch.equal(got, ref.reshape(heads * vdim, kvr)), f"{kp}wkvb2 rank{r}"
            n_checked += 3
        rd.end_layer()
        logger.info("verified %s: bit-exact against the checkpoint", kp)
    rd.close()
    if n_checked == 0:
        raise RuntimeError(f"no attention set matched layers {layers}")
    logger.info("verify_attn_lossless: %d tensors bit-exact", n_checked)


Getter = Callable[[str], torch.Tensor]


def verify_shards_against_checkpoint(
    model_dir: str,
    save_dir: str,
    layers: list[int] | None = None,
    args: ModelArgsGlm52 | None = None,
) -> None:
    args = args or ModelArgsGlm52()
    with open(os.path.join(save_dir, "tilert_meta.json")) as f:
        meta = json.load(f)
    n_layers = int(meta.get("n_layers", args.n_layers))
    attn_tp8 = bool(meta.get("attn_tp8", False))
    rd = CheckpointReader(model_dir)
    loaders: list[tuple[str, dict[str, str]]] = []
    for r in range(args.num_devices):
        rank_dir = os.path.join(save_dir, f"rank{r}")
        with open(os.path.join(rank_dir, "model.safetensors.index.json")) as f:
            loaders.append((rank_dir, json.load(f)["weight_map"]))
    handles: dict[str, object] = {}

    def stored(r: int, key: str) -> torch.Tensor:
        rank_dir, wm = loaders[r]
        if key not in wm:
            raise RuntimeError(f"rank{r}: {key} missing from the index")
        path = os.path.join(rank_dir, wm[key])
        if path not in handles:
            handles[path] = safe_open(path, framework="pt").__enter__()
        return handles[path].get_tensor(key)

    def same(a: torch.Tensor, b: torch.Tensor) -> bool:
        if a.dtype != b.dtype or tuple(a.shape) != tuple(b.shape):
            return False
        if a.dtype == torch.float8_e4m3fn:
            a, b = (a.view(torch.uint8), b.view(torch.uint8))
        return torch.equal(a.contiguous(), b.contiguous())

    def check(r: int, d: dict[str, torch.Tensor], tag: str) -> int:
        n = 0
        for k, want in d.items():
            got = stored(r, k)
            if not same(got, want):
                raise RuntimeError(
                    f"rank{r} {k}: stored {got.dtype}{tuple(got.shape)} != checkpoint slice {want.dtype}{tuple(want.shape)} ({tag})"
                )
            n += 1
        return n

    if layers is None:
        layers = list(range(n_layers))
        if meta.get("num_mtp", 0) > 0:
            layers.append(args.n_layers)
        layers += [-1]
    total = 0
    t0 = time.time()
    for i in layers:
        for r in range(args.num_devices):
            if i == -1:
                total += check(r, shard_tail(rd, r, args), "tail")
            elif i == args.n_layers:
                total += check(r, shard_mtp(rd, r, args, "cpu"), "mtp")
            else:
                d = shard_layer(rd, i, r, args, "cpu", attn_tp8=attn_tp8)
                total += check(r, {f"layer_{i}.{k}": v for k, v in d.items()}, f"layer {i}")
        logger.info(
            "verify_shards_against_checkpoint: %s ok (%d tensors so far, %.0f s)",
            "tail" if i == -1 else f"layer {i}",
            total,
            time.time() - t0,
        )
    if -1 in layers:
        with safe_open(os.path.join(save_dir, "shared", "embed.safetensors"), framework="pt") as f:
            emb = f.get_tensor("embed")
        want = rd.get("model.embed_tokens.weight").to(torch.bfloat16)
        if not same(emb, want):
            raise RuntimeError("shared/embed.safetensors != checkpoint embed_tokens")
        total += 1
    for h in handles.values():
        h.__exit__(None, None, None)
    logger.info(
        "verify_shards_against_checkpoint %s: %d tensors bit-exact vs %s",
        save_dir,
        total,
        model_dir,
    )


def _st_header(path: str) -> tuple[int, dict]:
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        hdr = json.loads(f.read(n))
    return (8 + n, hdr)


def prune_unreferenced(
    save_dir: str,
    args: ModelArgsGlm52 | None = None,
    dry_run: bool = False,
    chunk_bytes: int = 256 << 20,
) -> None:
    args = args or ModelArgsGlm52()
    total_removed = 0
    for r in range(args.num_devices):
        rank_dir = os.path.join(save_dir, f"rank{r}")
        idx_path = os.path.join(rank_dir, "model.safetensors.index.json")
        with open(idx_path) as f:
            idx = json.load(f)
        wm: dict[str, str] = idx["weight_map"]
        for fname in sorted(os.listdir(rank_dir)):
            if not fname.endswith(".safetensors"):
                continue
            link = os.path.join(rank_dir, fname)
            real = os.path.realpath(link)
            base, hdr = _st_header(real)
            meta_entry = hdr.pop("__metadata__", None)
            stale = sorted(k for k in hdr if wm.get(k) != fname)
            if not stale:
                continue
            keep = [k for k in hdr if k not in set(stale)]
            removed = sum(hdr[k]["data_offsets"][1] - hdr[k]["data_offsets"][0] for k in stale)
            total_removed += removed
            logger.info(
                "rank %d %s: %d unreferenced tensors (%.2f GB) to drop, %d kept%s",
                r,
                fname,
                len(stale),
                removed / 1000000000.0,
                len(keep),
                " [dry run]" if dry_run else "",
            )
            for k in stale[:6]:
                logger.info("    - %s", k)
            if dry_run:
                continue
            for k in keep:
                if wm.get(k) != fname:
                    raise RuntimeError(f"{k} kept but not mapped to {fname}")
            new_hdr: dict = {}
            if meta_entry is not None:
                new_hdr["__metadata__"] = meta_entry
            off = 0
            for k in keep:
                e = hdr[k]
                n = e["data_offsets"][1] - e["data_offsets"][0]
                new_hdr[k] = {
                    "dtype": e["dtype"],
                    "shape": e["shape"],
                    "data_offsets": [off, off + n],
                }
                off += n
            hb = json.dumps(new_hdr, separators=(",", ":")).encode()
            hb += b" " * (-len(hb) % 8)
            need = 8 + len(hb) + off
            st = os.statvfs(os.path.dirname(real))
            if st.f_bavail * st.f_frsize < need + (1 << 30):
                free_gb = st.f_bavail * st.f_frsize / 1000000000.0
                raise RuntimeError(
                    f"{os.path.dirname(real)}: {free_gb:.1f} GB free, need {need / 1000000000.0:.1f} GB to rewrite {fname}"
                )
            tmp = real + ".prune.tmp"
            digests: dict[str, bytes] = {}
            buf = bytearray(chunk_bytes)
            mv = memoryview(buf)
            src = os.open(real, os.O_RDONLY)
            dst = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 384)
            try:
                os.write(dst, len(hb).to_bytes(8, "little") + hb)
                for k in keep:
                    lo, hi = hdr[k]["data_offsets"]
                    h = hashlib.blake2b(digest_size=16)
                    pos = base + lo
                    while pos < base + hi:
                        want = min(chunk_bytes, base + hi - pos)
                        got = os.preadv(src, [mv[:want]], pos)
                        if got <= 0:
                            raise OSError(f"{fname}:{k}: short read at {pos}")
                        h.update(mv[:got])
                        w = 0
                        while w < got:
                            w += os.write(dst, mv[w:got])
                        pos += got
                    digests[k] = h.digest()
                os.fsync(dst)
            finally:
                os.close(src)
                os.close(dst)
            nb, nh = _st_header(tmp)
            nh.pop("__metadata__", None)
            if list(nh) != keep or os.path.getsize(tmp) != need:
                raise RuntimeError(f"{tmp}: header/size mismatch after rewrite")
            src = os.open(tmp, os.O_RDONLY)
            try:
                for k in keep:
                    lo, hi = nh[k]["data_offsets"]
                    if nh[k]["shape"] != hdr[k]["shape"] or nh[k]["dtype"] != hdr[k]["dtype"]:
                        raise RuntimeError(f"{tmp}:{k}: entry mismatch")
                    h = hashlib.blake2b(digest_size=16)
                    pos = nb + lo
                    while pos < nb + hi:
                        want = min(chunk_bytes, nb + hi - pos)
                        got = os.preadv(src, [mv[:want]], pos)
                        if got <= 0:
                            raise OSError(f"{tmp}:{k}: short read at {pos}")
                        h.update(mv[:got])
                        pos += got
                    if h.digest() != digests[k]:
                        raise RuntimeError(f"{tmp}:{k}: blake2b mismatch after rewrite")
            finally:
                os.close(src)
            os.replace(tmp, real)
            dfd = os.open(os.path.dirname(real), os.O_RDONLY)
            os.fsync(dfd)
            os.close(dfd)
            logger.info("rank %d %s: rewritten, %d tensors verified bit-exact", r, fname, len(keep))
        if not dry_run:
            tot = 0
            for fname in sorted(set(wm.values())):
                _, h = _st_header(os.path.realpath(os.path.join(rank_dir, fname)))
                h.pop("__metadata__", None)
                tot += sum(e["data_offsets"][1] - e["data_offsets"][0] for e in h.values())
            idx.setdefault("metadata", {})["total_size"] = tot
            _write_json_atomic(idx_path, idx)
    logger.info(
        "prune_unreferenced %s: %.2f GB of unreferenced tensors%s",
        save_dir,
        total_removed / 1000000000.0,
        " would be removed" if dry_run else " removed",
    )


def audit_referenced(save_dir: str, args: ModelArgsGlm52 | None = None) -> None:
    args = args or ModelArgsGlm52()
    n = 0
    for r in range(args.num_devices):
        rank_dir = os.path.join(save_dir, f"rank{r}")
        with open(os.path.join(rank_dir, "model.safetensors.index.json")) as f:
            wm = json.load(f)["weight_map"]
        seen: set[str] = set()
        for fname in sorted(os.listdir(rank_dir)):
            if not fname.endswith(".safetensors"):
                continue
            _, hdr = _st_header(os.path.realpath(os.path.join(rank_dir, fname)))
            hdr.pop("__metadata__", None)
            for k in hdr:
                if wm.get(k) != fname:
                    raise RuntimeError(f"rank{r}/{fname}: unreferenced tensor {k}")
                seen.add(k)
            n += len(hdr)
        missing = set(wm) - seen
        if missing:
            raise RuntimeError(f"rank{r}: {len(missing)} indexed tensors missing from shards")
    logger.info("audit_referenced %s: %d tensors, all referenced, none stale", save_dir, n)


def _pack_attn(g: Getter, p: str, rank: int, kind: int) -> list[torch.Tensor]:
    if kind == KIND_MOE_SHARED:
        q = f"{p}attn_tp8."
        return [
            g(f"{q}in_gamma"),
            swizzle_fp8_contig8(g(f"{q}wqkva")),
            g(f"{q}wqkva_scales"),
            g(f"{q}q_gamma"),
            swizzle_fp8_contig8(g(f"{q}wqb")),
            g(f"{q}wqb_scales"),
            g(f"{q}kv_gamma"),
            swizzle_fp8_contig8(g(f"{q}wkvb1")),
            g(f"{q}wkvb1_scales"),
            swizzle_fp8_contig8(g(f"{q}wkvb2")),
            g(f"{q}wkvb2_scales"),
            _pack_wo(g(f"{q}wo")),
            g(f"{q}wo_scales"),
        ]
    if rank == 0:
        if kind == KIND_MOE_SHARED:
            return []
        return [
            g(f"{p}in_gamma"),
            swizzle_fp8_contig8(g(f"{p}wqaki")),
            g(f"{p}wqaki_scales"),
            swizzle_wis_8x64(g(f"{p}wis")),
            g(f"{p}q_gamma"),
            swizzle_fp8_contig8(g(f"{p}wqi")),
            g(f"{p}wqi_scales"),
            g(f"{p}knorm_w"),
            g(f"{p}knorm_b"),
        ]
    return [
        g(f"{p}in_gamma"),
        swizzle_fp8_contig8(g(f"{p}wqkva")),
        g(f"{p}wqkva_scales"),
        g(f"{p}q_gamma"),
        swizzle_fp8_contig8(g(f"{p}wqb")),
        g(f"{p}wqb_scales"),
        g(f"{p}kv_gamma"),
        swizzle_fp8_contig8(g(f"{p}wkvb1")),
        g(f"{p}wkvb1_scales"),
        swizzle_fp8_contig8(g(f"{p}wkvb2")),
        g(f"{p}wkvb2_scales"),
        _pack_wo(g(f"{p}wo")),
        g(f"{p}wo_scales"),
    ]


def swizzle_fp8_v4(w8: torch.Tensor) -> torch.Tensor:
    e, dim, inter = w8.shape
    assert dim % 24 == 0 and inter == 256
    v = w8.view(torch.uint8).view(e, dim // 24, 6, 4, 8, 2, 16)
    return v.permute(0, 1, 4, 5, 2, 3, 6).reshape(e, -1).contiguous()


def swizzle_pair_interleaved_k128(w8: torch.Tensor, inter: int) -> torch.Tensor:
    wp = pair_interleave(_u8(w8), inter)
    *lead, rows, k = wp.shape
    assert rows % 16 == 0 and k % 128 == 0
    v = wp.view(*lead, rows // 16, 16, k // 128, 4, 2, 16)
    n = len(lead)
    perm = tuple(range(n)) + tuple(x + n for x in (0, 2, 4, 3, 1, 5))
    return v.permute(perm).reshape(*lead, rows * k).contiguous()


def swizzle_fp8_down_k128(w8: torch.Tensor) -> torch.Tensor:
    e, dim, inter = w8.shape
    assert dim % 24 == 0 and inter == 256
    v = w8.view(torch.uint8).view(e, dim // 24, 24, 2, 4, 2, 16)
    mains = v[:, :, :16].permute(0, 1, 3, 5, 4, 2, 6).reshape(e, dim // 24, 4096)
    tails = v[:, :, 16:].permute(0, 1, 3, 5, 4, 2, 6).reshape(e, dim // 24, 2048)
    return torch.cat([mains, tails], dim=-1).reshape(e, -1).contiguous()


def _env_flag(name: str, dflt: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None or v == "":
        return dflt
    if v in ("0", "1"):
        return v == "1"
    raise RuntimeError(
        f'{name}: expected "0" or "1", got {v!r} (the lever vars are a loader/dispatch contract, so a value the two sides could read differently is refused)'
    )


def moe_w8a8_enabled() -> bool:
    return _env_flag("TILERT_GLM5_MOE_W8A8", dflt=True)


def add_moe_arm_cli(ap) -> None:
    ap.add_argument(
        "--w8a16moe",
        action="store_true",
        help="run the MoE chain on the w8a16 v4 monokernel (fp8 weights, bf16 activations) instead of the default W8A8 one",
    )


def apply_moe_arm(args) -> str:
    w8a16 = bool(getattr(args, "w8a16moe", False))
    os.environ["TILERT_GLM5_MOE_W8A8"] = "0" if w8a16 else "1"
    return "w8a16" if w8a16 else "w8a8"


def fp8_ki_enabled() -> bool:
    return _env_flag("TILERT_GLM5_FP8_KI")


def add_index_arm_cli(ap) -> None:
    ap.add_argument(
        "--fp8-ki",
        action="store_true",
        help="fp8 e4m3 ki cache + iq_rt with per-row scales; stage 1 on the fp8 MFMA kernel (default: bf16)",
    )


def apply_index_arm(args) -> str:
    on = bool(getattr(args, "fp8_ki", False))
    os.environ["TILERT_GLM5_FP8_KI"] = "1" if on else "0"
    return "fp8-ki" if on else "bf16-ki"


def fp8_kv_enabled() -> bool:
    return _env_flag("TILERT_GLM5_FP8_KV")


def add_kv_arm_cli(ap) -> None:
    ap.add_argument(
        "--fp8-kv",
        action="store_true",
        help="fp8 e4m3 kv latent cache with per-128-block scales (528 B rows), Q quantized in-kernel, fp8 MFMA score (default: bf16)",
    )


def apply_kv_arm(args) -> str:
    on = bool(getattr(args, "fp8_kv", False))
    os.environ["TILERT_GLM5_FP8_KV"] = "1" if on else "0"
    return "fp8-kv" if on else "bf16-kv"


def _pack_ffn(
    g: Getter, p: str, kind: int, args: ModelArgsGlm52, moe_w8a8: bool = False
) -> list[torch.Tensor]:
    if kind == KIND_DENSE:
        return [
            g(f"{p}post_gamma"),
            swizzle_pair_interleaved(g(f"{p}wug"), args.dense_inter_shard),
            g(f"{p}wug_scales"),
            _pack_wo(g(f"{p}wdown")),
            g(f"{p}wdown_scales"),
        ]
    if moe_w8a8:
        moe_ug = swizzle_pair_interleaved_k128(g(f"{p}moe_ug"), args.moe_inter_shard).reshape(-1)
    else:
        moe_ug = swizzle_pair_interleaved(g(f"{p}moe_ug"), args.moe_inter_shard).reshape(-1)
    if moe_w8a8:
        moe_down = swizzle_fp8_down_k128(g(f"{p}moe_down")).reshape(-1)
    else:
        moe_down = swizzle_fp8_v4(g(f"{p}moe_down")).reshape(-1)
    return [
        g(f"{p}post_gamma"),
        swizzle_bf16_m4(g(f"{p}router")),
        g(f"{p}moe_bias"),
        moe_ug,
        g(f"{p}moe_ug_scales"),
        moe_down,
        g(f"{p}moe_down_scales"),
    ]


def pack_rank_params(
    get: Getter,
    embed: torch.Tensor,
    args: ModelArgsGlm52,
    rank: int,
    device: str,
    n_layers: int,
    num_mtp: int = 0,
    freqs_cis: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    if freqs_cis is None:
        freqs_cis = make_freqs_cis(args.max_seq_len, theta=args.rope_theta, device="cpu")
    if tuple(freqs_cis.shape) != (args.max_seq_len, 64) or freqs_cis.dtype != torch.float32:
        raise ValueError(
            f"freqs_cis must be [{args.max_seq_len}, 64] f32, got {tuple(freqs_cis.shape)} {freqs_cis.dtype}"
        )

    def g(name: str) -> torch.Tensor:
        return get(name).to(device, non_blocking=True)

    params: list[torch.Tensor] = [embed.to(torch.bfloat16).to(device), freqs_cis.to(device)]
    w8a8 = moe_w8a8_enabled()
    for i in range(n_layers):
        kind = layer_kind(i)
        params += _pack_attn(g, f"layer_{i}.", rank, kind)
        params += _pack_ffn(g, f"layer_{i}.", kind, args, moe_w8a8=w8a8)
    params.append(g("final_gamma"))
    params.append(swizzle_bf16_16x32(g("head")).view(torch.uint8))
    if num_mtp > 0:
        params += [
            swizzle_bf16_16x32(g("mtp.eh_w")).view(torch.uint8),
            g("mtp.e_gamma"),
            g("mtp.h_gamma"),
            g("mtp.head_gamma"),
        ]
        params += _pack_attn(g, "mtp_layer.", rank, KIND_MOE_FULL)
        params += _pack_ffn(g, "mtp_layer.", KIND_MOE_FULL, args, moe_w8a8=w8a8)
    return params


_ST_DTYPE = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
    "F8_E4M3": getattr(torch, "float8_e4m3fn", None),
    "F8_E5M2": getattr(torch, "float8_e5m2", None),
}


class RankBlobLoader:
    """Read one converted rank shard with pread() and land it on its GPU."""

    def __init__(self, rank_dir: str, device: str, staging_bytes: int = 128 << 20) -> None:
        with open(os.path.join(rank_dir, "model.safetensors.index.json")) as f:
            self._map: dict[str, str] = json.load(f)["weight_map"]
        self._dir = rank_dir
        self._dev = device
        self._fds: dict[str, int] = {}
        self._hdr: dict[str, tuple[int, dict]] = {}
        self._buf = [
            torch.empty(staging_bytes, dtype=torch.uint8, pin_memory=True) for _ in range(2)
        ]
        self._mv = [memoryview(b.numpy()) for b in self._buf]
        self._ev = [torch.cuda.Event() for _ in range(2)]
        self._slot = 0

    def _shard(self, name: str):
        if name not in self._hdr:
            path = os.path.join(self._dir, name)
            with open(path, "rb") as f:
                n = int.from_bytes(f.read(8), "little")
                meta = json.loads(f.read(n))
            self._hdr[name] = (8 + n, meta)
            self._fds[name] = os.open(path, os.O_RDONLY)
        return (self._hdr[name], self._fds[name])

    def get(self, key: str) -> torch.Tensor:
        (base, meta), fd = self._shard(self._map[key])
        entry = meta[key]
        lo, hi = entry["data_offsets"]
        dt = _ST_DTYPE[entry["dtype"]]
        if dt is None:
            raise RuntimeError(f"{key}: this torch has no {entry['dtype']}")
        nbytes = hi - lo
        out = torch.empty(nbytes, dtype=torch.uint8, device=self._dev)
        cap = self._buf[0].numel()
        off = 0
        while off < nbytes:
            s = self._slot
            self._ev[s].synchronize()
            want = min(cap, nbytes - off)
            got = os.preadv(fd, [self._mv[s][:want]], base + lo + off)
            if got <= 0:
                raise OSError(f"{key}: short read at {off} of {nbytes}")
            out[off : off + got].copy_(self._buf[s][:got], non_blocking=True)
            self._ev[s].record()
            self._slot ^= 1
            off += got
        return out.view(dt).reshape(entry["shape"])

    def close(self) -> None:
        torch.cuda.synchronize(self._dev)
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()
        self._buf.clear()
        self._mv.clear()


def load_rank_params(
    save_dir: str,
    args: ModelArgsGlm52,
    rank: int,
    device: str,
    n_layers: int | None = None,
    embed: torch.Tensor | None = None,
    num_mtp: int | None = None,
    freqs_cis: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    n_layers = args.n_layers if n_layers is None else n_layers
    num_mtp = args.num_mtp if num_mtp is None else num_mtp
    rd = RankBlobLoader(os.path.join(save_dir, f"rank{rank}"), device)
    if embed is None:
        with safe_open(os.path.join(save_dir, "shared", "embed.safetensors"), framework="pt") as f:
            embed = f.get_tensor("embed")
    try:
        return pack_rank_params(
            rd.get, embed, args, rank, device, n_layers, num_mtp, freqs_cis=freqs_cis
        )
    finally:
        rd.close()


def random_rank_params(
    args: ModelArgsGlm52,
    rank: int,
    device: str,
    n_layers: int,
    seed: int = 0,
    num_mtp: int = 0,
    freqs_cis: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    gen = torch.Generator(device=device).manual_seed(seed)
    heads, dim = (args.local_heads, args.dim)
    cache: dict[str, torch.Tensor] = {}
    dk = {"device": device, "generator": gen}

    def rep_dk(name: str) -> dict:
        h = zlib.crc32(name.encode()) & 4294967295
        g = torch.Generator(device=device).manual_seed((seed * 1000003 + h) % 2**63)
        return {"device": device, "generator": g}

    def rep_gm(name: str, n: int) -> torch.Tensor:
        return 1.0 + 0.1 * torch.randn(n, **rep_dk(name))

    def rep_bf(name: str, rows: int, k: int) -> torch.Tensor:
        return (torch.randn(rows, k, **rep_dk(name)) * k ** (-0.5)).to(torch.bfloat16)

    def rq(
        rows: int, k: int, blk_k: int = BLK, blk_m: int = BLK
    ) -> tuple[torch.Tensor, torch.Tensor]:
        w = torch.randn(rows, k, **dk) * k ** (-0.5)
        return quantize_fp8_block_padded(w, blk_k, blk_m)

    def gm(n: int) -> torch.Tensor:
        return 1.0 + 0.1 * torch.randn(n, **dk)

    def bf(rows: int, k: int) -> torch.Tensor:
        return (torch.randn(rows, k, **dk) * k ** (-0.5)).to(torch.bfloat16)

    def fill_layer(p: str, kind: int) -> None:
        if rank == 0:
            if kind != KIND_MOE_SHARED:
                cache[f"{p}in_gamma"], cache[f"{p}q_gamma"] = (gm(dim), gm(2048))
                cache[f"{p}wqaki"], cache[f"{p}wqaki_scales"] = rq(2176, dim)
                cache[f"{p}wis"] = bf(32, dim)
                cache[f"{p}wqi"], cache[f"{p}wqi_scales"] = rq(4096, 2048)
                cache[f"{p}knorm_w"] = gm(128)
                cache[f"{p}knorm_b"] = 0.1 * gm(128)
        else:
            cache[f"{p}in_gamma"], cache[f"{p}q_gamma"] = (gm(dim), gm(2048))
            cache[f"{p}kv_gamma"] = gm(512)
            cache[f"{p}wqkva"], cache[f"{p}wqkva_scales"] = rq(2624, dim)
            sm = ATTN_SCALE_BLK_M
            cache[f"{p}wqb"], cache[f"{p}wqb_scales"] = rq(heads * 256, 2048, BLK, sm)
            cache[f"{p}wkvb1"], cache[f"{p}wkvb1_scales"] = rq(heads * 512, 192, 64, sm)
            cache[f"{p}wkvb2"], cache[f"{p}wkvb2_scales"] = rq(heads * 256, 512, BLK, sm)
            cache[f"{p}wo"], cache[f"{p}wo_scales"] = rq(dim, heads * 256)
        if kind == KIND_DENSE:
            cache[f"{p}post_gamma"] = rep_gm(f"{p}post_gamma", dim)
            cache[f"{p}wug"], cache[f"{p}wug_scales"] = rq(2 * args.dense_inter_shard, dim)
            cache[f"{p}wdown"], cache[f"{p}wdown_scales"] = rq(dim, args.dense_inter_shard)
        else:
            inter = args.moe_inter_shard
            cache[f"{p}post_gamma"] = rep_gm(f"{p}post_gamma", dim)
            cache[f"{p}router"] = rep_bf(f"{p}router", 256, dim)
            cache[f"{p}moe_bias"] = 0.01 * torch.randn(256, **rep_dk(f"{p}moe_bias"))
            ug8, ugs = rq(N_BANK * 2 * inter, dim)
            cache[f"{p}moe_ug"] = ug8.view(N_BANK, 2 * inter, dim)
            cache[f"{p}moe_ug_scales"] = ugs.view(N_BANK, 2 * inter // BLK, dim // BLK)
            dn8, dns = rq(N_BANK * dim, inter)
            cache[f"{p}moe_down"] = dn8.view(N_BANK, dim, inter)
            cache[f"{p}moe_down_scales"] = dns.view(N_BANK, dim // BLK, inter // BLK)

    def fill_attn_tp8(p: str, li: int) -> None:
        g2 = torch.Generator(device=device).manual_seed(seed * 7919 + 977 * li + 13)
        dk2 = {"device": device, "generator": g2}

        def rq2(rows: int, k: int, blk_k: int = BLK, blk_m: int = BLK):
            w = torch.randn(rows, k, **dk2) * k ** (-0.5)
            return quantize_fp8_block_padded(w, blk_k, blk_m)

        def gm2(n: int) -> torch.Tensor:
            return 1.0 + 0.1 * torch.randn(n, **dk2)

        q = f"{p}attn_tp8."
        h8 = TP8_HEADS
        cache[f"{q}in_gamma"], cache[f"{q}q_gamma"] = (gm2(dim), gm2(2048))
        cache[f"{q}kv_gamma"] = gm2(512)
        cache[f"{q}wqkva"], cache[f"{q}wqkva_scales"] = rq2(2624, dim)
        sm = ATTN_SCALE_BLK_M
        cache[f"{q}wqb"], cache[f"{q}wqb_scales"] = rq2(h8 * 256, 2048, BLK, sm)
        cache[f"{q}wkvb1"], cache[f"{q}wkvb1_scales"] = rq2(h8 * 512, 192, 64, sm)
        cache[f"{q}wkvb2"], cache[f"{q}wkvb2_scales"] = rq2(h8 * 256, 512, BLK, sm)
        cache[f"{q}wo"], cache[f"{q}wo_scales"] = rq2(dim, h8 * 256)

    for i in range(n_layers):
        fill_layer(f"layer_{i}.", layer_kind(i))
        if layer_kind(i) == KIND_MOE_SHARED:
            fill_attn_tp8(f"layer_{i}.", i)
    cache["final_gamma"] = rep_gm("final_gamma", dim)
    cache["head"] = bf(args.vocab_shard, dim)
    if num_mtp > 0:
        cache["mtp.eh_w"] = bf(dim, dim // 4)
        cache["mtp.e_gamma"] = rep_gm("mtp.e_gamma", dim)
        cache["mtp.h_gamma"] = rep_gm("mtp.h_gamma", dim)
        cache["mtp.head_gamma"] = rep_gm("mtp.head_gamma", dim)
        fill_layer("mtp_layer.", KIND_MOE_FULL)
    embed = torch.randn(args.vocab_size, dim, **rep_dk("embed")).to(torch.bfloat16)
    return pack_rank_params(
        lambda k: cache[k], embed, args, rank, device, n_layers, num_mtp, freqs_cis=freqs_cis
    )


def selftest_swizzles() -> None:
    from tilert.models.glm_5_2_rocm.ops import moe_router
    from tilert.models.glm_5_2_rocm.ops import rmsnorm_head_proj as h1
    from tilert.models.glm_5_2_rocm.ops import rmsnorm_projq_wqb as m1
    from tilert.models.glm_5_2_rocm.ops import rmsnorm_projx_wqakis as s0
    from tilert.models.glm_5_2_rocm.ops import rmsnorm_projx_wqkva, upgate_silu

    g = torch.Generator().manual_seed(0)
    w = torch.randn(2624, 6144, generator=g)
    q8, _ = rmsnorm_projx_wqkva.quantize_fp8_block(w)
    ref = rmsnorm_projx_wqkva.swizzle_weights_contig(q8)
    assert torch.equal(ref, swizzle_fp8_contig8(q8)), "fp8 contig-8 mismatch"
    wr = torch.randn(256, 6144, generator=g).to(torch.bfloat16)
    assert torch.equal(
        moe_router.swizzle_router_bf16(wr), swizzle_bf16_16x32(wr)
    ), "router bf16 mismatch"
    from tilert.models.glm_5_2_rocm.ops import eh_proj_allreduce as t0
    from tilert.models.glm_5_2_rocm.ops import moe_down_allreduce as mdown

    assert torch.equal(
        t0.swizzle_256_bf16(wr).view(torch.uint8), swizzle_bf16_m4(wr)
    ), "router m4 bf16 mismatch"
    wd = torch.randn(6144, 256, generator=g)
    d8, _ = mdown.quantize_fp8_block(wd)
    assert torch.equal(mdown.swizzle_m4(d8), swizzle_fp8_m4(d8)), "down m4 fp8 mismatch"
    from tilert.models.glm_5_2_rocm.ops import unprojo_allreduce as m7

    wo = torch.randn(6144, 2560, generator=g)
    o8, _ = mdown.quantize_fp8_block(wo)
    assert torch.equal(m7.swizzle_v2(o8), swizzle_fp8_v2(o8)), "wo v2 fp8 mismatch"
    wh = torch.randn(19360, 6144, generator=g).to(torch.bfloat16)
    assert torch.equal(h1.swizzle_head_bf16(wh), swizzle_bf16_16x32(wh)), "head bf16 mismatch"
    wis = torch.randn(32, 6144, generator=g).to(torch.bfloat16)
    assert torch.equal(s0.swizzle_wis_bf16(wis), swizzle_wis_8x64(wis)), "wis bf16 mismatch"
    for inter in (256, 1536):
        wu = torch.randn(2 * inter, 6144, generator=g)
        u8, _ = upgate_silu.quantize_fp8_block(wu)
        ref = upgate_silu.swizzle_pair_interleaved(u8, inter)
        got = swizzle_pair_interleaved(u8, inter)
        assert torch.equal(ref, got), f"pair-interleave({inter}) mismatch"
    wq = torch.randn(2560, 2048, generator=g)
    q8b, sref = quantize_fp8_block(wq, BLK, ATTN_SCALE_BLK_M)
    q8m, sm = m1.quantize_fp8_block(wq)
    assert torch.equal(q8b.view(torch.uint8), q8m.view(torch.uint8))
    assert torch.equal(sref, sm)
    print("selftest_swizzles: OK")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Convert HF GLM-5.2-FP8 weights to TileRT TP8 device shards."
    )
    ap.add_argument("--model_dir", type=str, help="HF checkpoint directory")
    ap.add_argument("--save_dir", type=str, help="output directory")
    ap.add_argument("--layers", type=int, default=None, help="convert first N layers")
    ap.add_argument("--device", type=str, default="cpu", help="work device for requantization")
    ap.add_argument(
        "--num_mtp",
        type=int,
        default=None,
        help="convert the MTP module too (default: model_args.num_mtp)",
    )
    ap.add_argument(
        "--augment-attn-tp8",
        action="store_true",
        help="append the attn_tp8 set to an EXISTING conversion in save_dir (additive + idempotent; no full re-conversion)",
    )
    ap.add_argument(
        "--store-dir",
        type=str,
        default=None,
        help="augment only: write the new shards here and symlink them into the rank dirs (for a save_dir on a full filesystem)",
    )
    ap.add_argument(
        "--augment-attn-lossless",
        action="store_true",
        help="replace the re-quantized attention tensors of an EXISTING conversion in save_dir with lossless checkpoint byte slices (in place, idempotent; no full re-conversion)",
    )
    ap.add_argument(
        "--verify-attn-lossless",
        type=str,
        default=None,
        metavar="LAYERS",
        help="comma-separated layers: check the conversion's attention tensors dequantize bit-exactly to the checkpoint's",
    )
    ap.add_argument(
        "--verify-shards",
        type=str,
        default=None,
        metavar="LAYERS",
        help="'all' or comma-separated layers (78 = MTP block, -1 = tail+embed): assert every stored tensor equals the converter's byte slice of the checkpoint (no re-quantization anywhere); needs --model_dir",
    )
    ap.add_argument(
        "--prune-unreferenced",
        action="store_true",
        help="rewrite the shards in save_dir dropping every tensor the index no longer maps there (the superseded re-quantized Wq_b/Wkv_b after --augment-attn-lossless); every kept tensor is hash-verified",
    )
    ap.add_argument(
        "--audit-referenced",
        action="store_true",
        help="fail unless every tensor in every shard of save_dir is the one its index maps there (no stale bytes)",
    )
    ap.add_argument("--dry-run", action="store_true", help="with --prune-unreferenced: report only")
    ap.add_argument(
        "--stamp-provenance",
        action="store_true",
        help="write the provenance / format metadata the loader requires into an older conversion's tilert_meta.json, after binding save_dir to --model_dir by content (tail + embed bit-exact); idempotent",
    )
    ap.add_argument("--selftest", action="store_true", help="run the swizzle selftest")
    cli = ap.parse_args()
    if cli.selftest:
        selftest_swizzles()
    elif cli.stamp_provenance:
        assert cli.model_dir and cli.save_dir, "--model_dir and --save_dir required"
        stamp_provenance(
            model_dir=os.path.expanduser(cli.model_dir), save_dir=os.path.expanduser(cli.save_dir)
        )
    elif cli.verify_shards is not None:
        assert cli.model_dir and cli.save_dir, "--model_dir and --save_dir required"
        verify_shards_against_checkpoint(
            model_dir=os.path.expanduser(cli.model_dir),
            save_dir=os.path.expanduser(cli.save_dir),
            layers=(
                None
                if cli.verify_shards == "all"
                else [int(x) for x in cli.verify_shards.split(",")]
            ),
        )
    elif cli.prune_unreferenced:
        assert cli.save_dir, "--save_dir required"
        prune_unreferenced(os.path.expanduser(cli.save_dir), dry_run=cli.dry_run)
    elif cli.audit_referenced:
        assert cli.save_dir, "--save_dir required"
        audit_referenced(os.path.expanduser(cli.save_dir))
    elif cli.augment_attn_lossless:
        assert cli.model_dir and cli.save_dir, "--model_dir and --save_dir required"
        augment_attn_lossless(
            model_dir=os.path.expanduser(cli.model_dir),
            save_dir=os.path.expanduser(cli.save_dir),
            store_dir=os.path.expanduser(cli.store_dir) if cli.store_dir else None,
        )
    elif cli.verify_attn_lossless is not None:
        assert cli.model_dir and cli.save_dir, "--model_dir and --save_dir required"
        verify_attn_lossless(
            model_dir=os.path.expanduser(cli.model_dir),
            save_dir=os.path.expanduser(cli.save_dir),
            layers=[int(x) for x in cli.verify_attn_lossless.split(",")],
        )
    elif cli.augment_attn_tp8:
        assert cli.model_dir and cli.save_dir, "--model_dir and --save_dir required"
        augment_attn_tp8(
            model_dir=os.path.expanduser(cli.model_dir),
            save_dir=os.path.expanduser(cli.save_dir),
            work_dev=cli.device,
            store_dir=os.path.expanduser(cli.store_dir) if cli.store_dir else None,
        )
    else:
        assert cli.model_dir and cli.save_dir, "--model_dir and --save_dir required"
        Glm52WeightConverter(
            model_dir=os.path.expanduser(cli.model_dir),
            save_dir=os.path.expanduser(cli.save_dir),
            n_layers=cli.layers,
            work_dev=cli.device,
            num_mtp=cli.num_mtp,
            attn_tp8=True,
        ).convert()
