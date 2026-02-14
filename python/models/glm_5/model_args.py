"""Model arguments and hyperparameters."""

from dataclasses import dataclass
from typing import Literal

from tilert.models.deepseek_v3_2.model_args import ModelArgs

__all__ = [
    "ModelArgsGLM5",
]


@dataclass
class ModelArgsGLM5(ModelArgs):
    """
    Data class for defining model arguments and hyperparameters.

    Attributes:
        arch_name (str): Architecture name.
        max_batch_size (int): Maximum batch size.
        max_seq_len (int): Maximum sequence length.
        dtype (Literal["bf16", "fp8"]): Data type for computations.
        scale_fmt (Optional[str]): Format for quantization scale.
        vocab_size (int): Vocabulary size.
        dim (int): Model dimension.
        inter_dim (int): Intermediate dimension for MLP layers.
        moe_inter_dim (int): Intermediate dimension for MoE layers.
        n_layers (int): Number of transformer layers.
        n_dense_layers (int): Number of dense layers in the model.
        n_heads (int): Number of attention heads.
        n_routed_experts (int): Number of routed experts for MoE layers.
        n_shared_experts (int): Number of shared experts for MoE layers.
        n_activated_experts (int): Number of activated experts in MoE layers.
        n_expert_groups (int): Number of expert groups.
        n_limited_groups (int): Number of limited groups for MoE routing.
        score_func (Literal["softmax", "sigmoid"]): Scoring function for MoE routing.
        route_scale (float): Scaling factor for routing scores.
        q_lora_rank (int): LoRA rank for query projections.
        kv_lora_rank (int): LoRA rank for key-value projections.
        qk_nope_head_dim (int): Dimension for query-key projections without positional embeddings.
        qk_rope_head_dim (int): Dimension for query-key projections with rotary embeddings.
        v_head_dim (int): Dimension for value projections.
        original_seq_len (Optional[int]): Original sequence length.
        rope_theta (float): Base for rotary positional encoding.
        rope_factor (Optional[float]): Scaling factor for extended sequence lengths.
        beta_fast (Optional[int]): Fast beta correction factor.
        beta_slow (Optional[int]): Slow beta correction factor.
        mscale (float): Scaling factor for extended attention.
        index_head_dim (int): Dimension for index head.
        index_topk (int): Top-k for index head.
    """

    arch_name = "glm_5"

    max_batch_size: int = 1  # NOTE: the current implementation only supports a batch size being 1
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

    # moe
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    n_activated_experts: int = 8
    # n_expert_groups: int = 8
    # n_limited_groups: int = 4
    score_func: Literal["softmax", "sigmoid"] = "softmax"
    route_scale: float = 2.5

    # mla
    q_lora_rank: int = 2048
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 192
    qk_rope_head_dim: int = 64
    v_head_dim: int = 256

    # yarn
    original_seq_len: int | None = None
    rope_theta: float = 1000000.0
    rope_factor: float | None = None
    beta_fast: int | None = None
    beta_slow: int | None = None
    mscale: float = 1.0

    # index
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 2048

    # quant
    block_size: int = 128

    eps: float = 1e-5
