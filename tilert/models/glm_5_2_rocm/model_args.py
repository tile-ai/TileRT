"""Model arguments for GLM-5.2 (TP8 decode on 8x MI350X)."""

from dataclasses import dataclass

__all__ = ["ModelArgsGlm52", "layer_kind", "full_layer_ordinals"]
KIND_DENSE = 0
KIND_MOE_FULL = 1
KIND_MOE_SHARED = 2


def layer_kind(i: int) -> int:
    if i < 3:
        return KIND_DENSE
    return KIND_MOE_FULL if (i - 2) % 4 == 0 else KIND_MOE_SHARED


def full_layer_ordinals(n_layers: int) -> list[int]:
    return [i for i in range(n_layers) if layer_kind(i) != KIND_MOE_SHARED]


@dataclass
class ModelArgsGlm52:
    """GLM-5.2 model arguments (decode bring-up scope: batch=1, seq=1)."""

    arch_name = "glm_5_2"
    max_batch_size: int = 1
    max_seq_len: int = 4096
    vocab_size: int = 154880
    dim: int = 6144
    inter_dim: int = 12288
    moe_inter_dim: int = 2048
    n_layers: int = 78
    n_dense_layers: int = 3
    n_heads: int = 64
    q_lora_rank: int = 2048
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 192
    qk_rope_head_dim: int = 64
    v_head_dim: int = 256
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 2048
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    n_activated_experts: int = 8
    route_scale: float = 2.5
    rope_theta: float = 8000000.0
    eps: float = 1e-05
    num_devices: int = 8
    local_heads: int = 10
    num_mtp: int = 0

    @property
    def vocab_shard(self) -> int:
        return self.vocab_size // self.num_devices

    @property
    def dense_inter_shard(self) -> int:
        return self.inter_dim // self.num_devices

    @property
    def moe_inter_shard(self) -> int:
        return self.moe_inter_dim // self.num_devices
