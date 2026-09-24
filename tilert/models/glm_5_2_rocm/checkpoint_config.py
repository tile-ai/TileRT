"""HF ``config.json`` <-> ``ModelArgsGlm52`` cross-check."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from tilert.models.glm_5_2_rocm.model_args import (
    KIND_DENSE,
    KIND_MOE_SHARED,
    ModelArgsGlm52,
    layer_kind,
)

__all__ = [
    "CONVERTER_VERSION",
    "ConfigMismatch",
    "describe_hf_config",
    "load_hf_config",
    "rope_theta_of",
    "sha256_file",
    "validate_hf_config",
]
CONVERTER_VERSION = 3


class ConfigMismatch(ValueError):
    """The checkpoint's config.json contradicts the model args / kernels."""


def load_hf_config(path_or_dir: str) -> dict[str, Any]:
    path = path_or_dir
    if os.path.isdir(path):
        path = os.path.join(path, "config.json")
    with open(path) as f:
        return json.load(f)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rope_theta_of(cfg: dict[str, Any]) -> float | None:
    rp = cfg.get("rope_parameters")
    if isinstance(rp, dict) and "rope_theta" in rp:
        return float(rp["rope_theta"])
    if "rope_theta" in cfg:
        return float(cfg["rope_theta"])
    return None


def _rope_scaling_of(cfg: dict[str, Any]) -> Any:
    rp = cfg.get("rope_parameters")
    if isinstance(rp, dict):
        t = rp.get("rope_type", "default")
        return None if t in (None, "default") else t
    return cfg.get("rope_scaling")


def validate_hf_config(
    cfg: dict[str, Any],
    args: ModelArgsGlm52 | None = None,
    *,
    n_layers: int | None = None,
    num_mtp: int | None = None,
    max_seq_len: int | None = None,
) -> None:
    args = args or ModelArgsGlm52()
    n_layers = args.n_layers if n_layers is None else n_layers
    num_mtp = args.num_mtp if num_mtp is None else num_mtp
    max_seq_len = args.max_seq_len if max_seq_len is None else max_seq_len
    bad: list[str] = []

    def want(key: str, expect: Any) -> None:
        got = cfg.get(key, "<missing>")
        if got != expect:
            bad.append(f"{key}: checkpoint {got!r} != expected {expect!r}")

    want("hidden_size", args.dim)
    want("vocab_size", args.vocab_size)
    want("intermediate_size", args.inter_dim)
    want("moe_intermediate_size", args.moe_inter_dim)
    want("num_attention_heads", args.n_heads)
    want("q_lora_rank", args.q_lora_rank)
    want("kv_lora_rank", args.kv_lora_rank)
    want("qk_nope_head_dim", args.qk_nope_head_dim)
    want("qk_rope_head_dim", args.qk_rope_head_dim)
    want("v_head_dim", args.v_head_dim)
    want("index_topk", args.index_topk)
    want("index_head_dim", args.index_head_dim)
    want("index_n_heads", args.index_n_heads)
    want("n_routed_experts", args.n_routed_experts)
    want("n_shared_experts", args.n_shared_experts)
    want("num_experts_per_tok", args.n_activated_experts)
    want("routed_scaling_factor", args.route_scale)
    want("first_k_dense_replace", args.n_dense_layers)
    want("rms_norm_eps", args.eps)
    want("scoring_func", "sigmoid")
    want("topk_method", "noaux_tc")
    want("norm_topk_prob", True)
    want("n_group", 1)
    want("topk_group", 1)
    if "indexer_rope_interleave" in cfg:
        want("indexer_rope_interleave", True)
    theta = rope_theta_of(cfg)
    if theta is None:
        bad.append("rope_theta: missing (neither rope_parameters.rope_theta nor rope_theta)")
    elif theta != args.rope_theta:
        bad.append(f"rope_theta: checkpoint {theta!r} != expected {args.rope_theta!r}")
    scaling = _rope_scaling_of(cfg)
    if scaling is not None:
        bad.append(f"rope scaling {scaling!r}: the port implements plain RoPE only")
    depth = cfg.get("num_hidden_layers", "<missing>")
    if not isinstance(depth, int) or depth < n_layers:
        bad.append(f"num_hidden_layers: checkpoint {depth!r} < wanted {n_layers}")
    if n_layers == args.n_layers and depth != args.n_layers:
        bad.append(f"num_hidden_layers: checkpoint {depth!r} != {args.n_layers}")
    nextn = cfg.get("num_nextn_predict_layers", 0)
    if num_mtp > 0 and nextn != 1:
        bad.append(
            f"num_nextn_predict_layers: checkpoint {nextn!r}, the port runs exactly one MTP module"
        )
    share = cfg.get("index_share_for_mtp_iteration")
    if num_mtp > 1 and share is not None and (share is not True):
        bad.append(
            f"index_share_for_mtp_iteration: checkpoint {share!r}, chained MTP drafts (num_mtp {num_mtp}) share MTP[0]'s selection"
        )
    max_pos = cfg.get("max_position_embeddings", "<missing>")
    if not isinstance(max_pos, int):
        bad.append(f"max_position_embeddings: {max_pos!r}")
    elif max_seq_len > max_pos:
        bad.append(f"max_seq_len {max_seq_len} > max_position_embeddings {max_pos}")
    itypes = cfg.get("indexer_types")
    freq = cfg.get("index_topk_freq")
    skip_off = cfg.get("index_skip_topk_offset")
    if isinstance(itypes, list):
        n = min(len(itypes), n_layers)
        exp = ["shared" if layer_kind(i) == KIND_MOE_SHARED else "full" for i in range(n)]
        if itypes[:n] != exp:
            bad.append("indexer_types: does not follow the (i-2) % 4 full/shared rule")
    elif freq is not None or skip_off is not None:
        if not (isinstance(freq, int) and isinstance(skip_off, int) and (freq > 0)):
            bad.append(f"index_topk_freq / index_skip_topk_offset: {freq!r} / {skip_off!r}")
        else:
            for i in range(n_layers):
                full = max(i - skip_off + 1, 0) % freq == 0
                if full != (layer_kind(i) != KIND_MOE_SHARED):
                    bad.append(
                        f"index_topk_freq {freq} / index_skip_topk_offset {skip_off}: layer {i} disagrees with the kernels' (i-2) % 4 full/shared rule"
                    )
                    break
    else:
        bad.append(
            "indexer layer pattern: neither indexer_types nor index_topk_freq/index_skip_topk_offset present"
        )
    mtypes = cfg.get("mlp_layer_types")
    moe_freq = cfg.get("moe_layer_freq")
    if isinstance(mtypes, list):
        n = min(len(mtypes), n_layers)
        exp = ["dense" if layer_kind(i) == KIND_DENSE else "sparse" for i in range(n)]
        if mtypes[:n] != exp:
            bad.append("mlp_layer_types: does not follow the 3-dense-then-MoE rule")
    elif moe_freq is not None:
        if moe_freq != 1:
            bad.append(f"moe_layer_freq: checkpoint {moe_freq!r} != 1 (every layer)")
    else:
        bad.append("mlp layer pattern: neither mlp_layer_types nor moe_layer_freq present")
    q = cfg.get("quantization_config")
    if not isinstance(q, dict):
        bad.append("quantization_config: missing (the port expects an fp8 checkpoint)")
    else:
        if q.get("quant_method") != "fp8" or q.get("fmt") != "e4m3":
            qm, fmt = (q.get("quant_method"), q.get("fmt"))
            bad.append(f"quantization_config: {qm!r}/{fmt!r} != fp8/e4m3")
        if list(q.get("weight_block_size", [])) != [128, 128]:
            bad.append(f"weight_block_size: {q.get('weight_block_size')!r} != [128, 128]")
    if bad:
        raise ConfigMismatch(
            "checkpoint config.json does not match ModelArgsGlm52 / the kernels:\n  "
            + "\n  ".join(bad)
        )


def describe_hf_config(cfg: dict[str, Any]) -> dict[str, Any]:
    q = cfg.get("quantization_config") or {}
    return {
        "hf_model_type": cfg.get("model_type"),
        "hf_architectures": cfg.get("architectures"),
        "num_hidden_layers": cfg.get("num_hidden_layers"),
        "num_nextn_predict_layers": cfg.get("num_nextn_predict_layers"),
        "max_position_embeddings": cfg.get("max_position_embeddings"),
        "rope_theta": rope_theta_of(cfg),
        "index_topk": cfg.get("index_topk"),
        "index_share_for_mtp_iteration": cfg.get("index_share_for_mtp_iteration"),
        "quant": {
            "quant_method": q.get("quant_method"),
            "fmt": q.get("fmt"),
            "weight_block_size": q.get("weight_block_size"),
        },
        "eos_token_id": cfg.get("eos_token_id"),
    }
