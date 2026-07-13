"""Model arguments and hyperparameters."""

from dataclasses import dataclass
from typing import Literal

__all__ = [
    "ModelArgs",
]


@dataclass
class ModelArgs:
    """Data class for defining model arguments and hyperparameters."""

    arch_name = "deepseek_v3_2"

    max_batch_size: int = 1
    max_seq_len: int = 160 * 1024
    dtype: Literal["bf16", "fp8"] = "fp8"
    scale_fmt: str | None = None
    fp8_kv_cache: bool = False

    vocab_size: int = 129280
    dim: int = 7168
    inter_dim: int = 18432
    moe_inter_dim: int = 2048
    n_layers: int = 61
    n_dense_layers: int = 3
    n_heads: int = 128

    n_routed_experts: int = 256
    n_shared_experts: int = 1
    n_activated_experts: int = 8
    n_expert_groups: int = 8
    n_limited_groups: int = 4
    score_func: Literal["softmax", "sigmoid", "sqrtsoftplus"] = "softmax"
    route_scale: float = 2.5

    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128

    original_seq_len: int | None = 4096
    rope_theta: float = 10000.0
    rope_factor: float | None = 40
    beta_fast: int | None = 32
    beta_slow: int | None = 1
    mscale: float = 1.0

    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 2048

    kv_cache_pad: int = 8

    block_size: int = 128

    eps: float = 1e-6
