"""QKV Rope operation module.

Unified for deepseek_v3_2 (n_local_heads=16) and glm_5 (n_local_heads=8).
Dispatches by q_pe.shape[2]: 16 -> qkv_rope_op, 8 -> qkv_rope_glm5_op.
"""

from dataclasses import dataclass

import torch

from tilert.models.base import TileRTModule
from tilert.models.deepseek_v3_2.model_args import ModelArgs
from tilert.models.utils import apply_rotary_emb
from tilert.profiler.utils import parse_profile_log_tensor
from tilert.utils import get_profile_log_tensor

__all__ = [
    "qkv_rope",
    "QKVRoPE",
    "QKVRoPERefWeightsAlias",
    "QKVRoPETilertWeightsAlias",
]


def qkv_rope(
    pe_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    rope_freqs: torch.Tensor,
    cur_pos: torch.Tensor,
    profile_logs: torch.Tensor,
) -> None:
    """
    Perform QKV Rope operation.

    Unified for deepseek_v3_2 (16 heads) and glm_5 (8 heads). Dispatches by
    pe_cache (q_pe) shape[2]: 16 -> qkv_rope_op, 8 -> qkv_rope_glm5_op.

    Args:
        pe_cache: Q PE tensor (bsz, seq, n_local_heads, qk_rope_head_dim).
        kv_cache: K PE cache (bsz, seq, qk_rope_head_dim).
        rope_freqs: Rope frequencies tensor.
        cur_pos: Current position tensor.
        profile_logs: Profile logs tensor.
    """
    n_local_heads = pe_cache.shape[2]
    qk_rope_head_dim = pe_cache.shape[3]
    if qk_rope_head_dim != 64:
        raise ValueError(f"Unsupported qk_rope_head_dim: {qk_rope_head_dim}")

    if n_local_heads == 16:
        torch.ops.tilert.qkv_rope_op(pe_cache, kv_cache, rope_freqs, cur_pos, profile_logs)
    elif n_local_heads == 8:
        torch.ops.tilert.qkv_rope_glm5_op(pe_cache, kv_cache, rope_freqs, cur_pos, profile_logs)
    else:
        raise ValueError(
            f"Unsupported n_local_heads: {n_local_heads}. "
            "QKVRoPE supports n_local_heads=16 (deepseek_v3_2) or 8 (glm_5)."
        )


@dataclass
class QKVRoPERefWeightsAlias:
    """Reference weights alias for QKVRoPE (no weights)."""

    @property
    def ref_tensor_alias(self) -> list[str]:
        return []

    def __call__(self) -> list[str]:
        return self.ref_tensor_alias


@dataclass
class QKVRoPETilertWeightsAlias:
    """TileRT weights alias for QKVRoPE (no weights)."""

    @property
    def tilert_tensor_alias(self) -> list[str]:
        return []

    def __call__(self) -> list[str]:
        return self.tilert_tensor_alias


class QKVRoPE(TileRTModule):
    """QKV RoPE module. Unified for deepseek_v3_2 and glm_5."""

    def __init__(
        self,
        model_args: ModelArgs,
        num_devices: int = 1,
        device_id: int = 0,
        layer_idx: int = 0,
        ref_weights_alias: QKVRoPERefWeightsAlias | None = None,
    ) -> None:
        super().__init__(
            self.__class__.__name__,
            model_args=model_args,
            num_devices=num_devices,
            device_id=device_id,
            layer_idx=layer_idx,
        )
        self.tilert_weights_alias = QKVRoPETilertWeightsAlias()
        self.ref_weights_alias = (
            ref_weights_alias if ref_weights_alias is not None else QKVRoPERefWeightsAlias()
        )
        self.n_local_heads = model_args.n_heads // num_devices
        self.qk_rope_head_dim = model_args.qk_rope_head_dim
        self.profile_logs: torch.Tensor | None = None

    def get_weights_list(self) -> list[torch.Tensor]:
        return []

    def device_sharding(self, weights_map: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        del weights_map
        return {}

    def init_reference_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        del state_dict
        pass

    def init_tilert_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        del state_dict
        pass

    def init_random_weights(self) -> None:
        pass

    def init_tilert_vars(self, batch_size: int, seq_len: int) -> None:
        del batch_size, seq_len
        self.profile_logs = get_profile_log_tensor()
        self.is_var_init = True

    def golden_forward(
        self,
        q_pe: torch.Tensor,
        pe_cache: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        bsz: int,
        seqlen: int,
    ) -> torch.Tensor:
        end_pos = start_pos + seqlen

        k_pe = pe_cache[:bsz, start_pos:end_pos]
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)
        pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)

        return apply_rotary_emb(q_pe, freqs_cis)

    def tilert_forward(
        self,
        q_pe: torch.Tensor,
        pe_cache: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        bsz: int,
        seqlen: int,
    ) -> torch.Tensor:
        assert self.profile_logs is not None
        end_pos = start_pos + seqlen

        q_pe_rope = q_pe.clone()
        rope_freqs = torch.view_as_real(freqs_cis).reshape(*freqs_cis.shape[:-1], -1)
        cur_pos = torch.tensor([start_pos], dtype=torch.int32)

        qkv_rope(
            q_pe_rope, pe_cache[:bsz, start_pos:end_pos], rope_freqs, cur_pos, self.profile_logs
        )
        if self.flag_enable_profiling_log:
            parse_profile_log_tensor(
                self.profile_logs, self.get_profile_log_path(), [(self.op_name, 0.0)]
            )

        return q_pe_rope

    def __call__(
        self,
        q_pe: torch.Tensor,
        pe_cache: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        bsz: int,
        seqlen: int,
    ) -> torch.Tensor:
        if self.flag_enable_tilert:
            return self.tilert_forward(q_pe, pe_cache, start_pos, freqs_cis, bsz, seqlen)
        return self.golden_forward(q_pe, pe_cache, start_pos, freqs_cis, bsz, seqlen)
