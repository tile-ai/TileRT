"""Model arguments and hyperparameters."""

from dataclasses import dataclass
from typing import Literal

from tilert.models.glm_5._dsa_v32.model_args import ModelArgs

__all__ = [
    "ModelArgsGLM5",
]


@dataclass
class ModelArgsGLM5(ModelArgs):
    """Data class for defining model arguments and hyperparameters."""

    arch_name = "glm_5"

    max_batch_size: int = 1
    max_seq_len: int = 202752
    dtype: Literal["bf16", "fp8"] = "fp8"
    scale_fmt: str | None = None

    vocab_size: int = 154880
    dim: int = 6144
    inter_dim: int = 12288
    moe_inter_dim: int = 2048
    n_layers: int = 78
    n_dense_layers: int = 3
    n_heads: int = 64

    n_routed_experts: int = 256
    n_shared_experts: int = 1
    n_activated_experts: int = 8
    n_expert_groups: int = 1
    n_limited_groups: int = 1
    score_func: Literal["softmax", "sigmoid"] = "sigmoid"
    route_scale: float = 2.5

    q_lora_rank: int = 2048
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 192
    qk_rope_head_dim: int = 64
    v_head_dim: int = 256

    original_seq_len: int | None = None
    rope_theta: float = 1000000.0
    rope_factor: float | None = None
    beta_fast: int | None = None
    beta_slow: int | None = None
    mscale: float = 1.0

    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 2048

    block_size: int = 128

    eps: float = 1e-5
