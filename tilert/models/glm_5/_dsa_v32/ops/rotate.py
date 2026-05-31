"""Rotate(hadamard transform) operation module."""

from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn.functional as F

from tilert.models.base import TileRTModule
from tilert.models.glm_5._dsa_v32.model_args import ModelArgs
from tilert.models.utils import apply_rotary_emb
from tilert.utils import get_profile_log_tensor

try:
    from fast_hadamard_transform import hadamard_transform

    def rotate_activation(x: torch.Tensor) -> torch.Tensor:
        assert x.dtype == torch.bfloat16
        hidden_size = x.size(-1)
        return hadamard_transform(x, scale=hidden_size**-0.5)

except ImportError:
    print(
        "Cannot import hadamard_transform, fallback to scipy.linalg.hadamard."
        "please install fast_hadamard_transform for correct performance."
    )
    import math

    from scipy.linalg import hadamard

    def hadamard_transform_ref(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        x_shape = x.shape
        dim = x.shape[-1]
        x = x.reshape(-1, dim)
        log_dim = math.ceil(math.log2(dim))
        dim_padded = 2**log_dim
        if dim != dim_padded:
            x = F.pad(x, (0, dim_padded - dim))
        out = F.linear(
            x,
            torch.tensor(hadamard(dim_padded, dtype=float), dtype=x.dtype, device=x.device),
        )
        out = out * scale
        return out[..., :dim].reshape(*x_shape)

    def rotate_activation(x: torch.Tensor) -> torch.Tensor:
        assert x.dtype == torch.bfloat16
        hidden_size = x.size(-1)
        return hadamard_transform_ref(x, scale=hidden_size**-0.5)


__all__ = [
    "rotate",
    "rotate_activation",
    "Rotate",
    "RotateRefWeightsAlias",
    "RotateTilertWeightsAlias",
]


def rotate(
    input_raw: torch.Tensor,
    output_raw: torch.Tensor,
    freqs_cis_raw: torch.Tensor,
    profile_logs: torch.Tensor,
    model_arch: str,
    compute_kernel_type: str = "general",
) -> None:
    """
    Rotate (hadamard transform) operation.

    Args:
        input_raw (torch.Tensor): The input tensor [..., head, 128].
        output_raw (torch.Tensor): The output tensor where the result will be stored.
        freqs_cis_raw (torch.Tensor): The frequency tensor.
        profile_logs (torch.Tensor): Tensor for storing profiling logs.
        model_arch: Model architecture string.
        compute_kernel_type: Compute kernel type string.

    Returns:
        None
    """
    torch.ops.tilert.rotate_op(
        input_raw,
        output_raw,
        freqs_cis_raw,
        model_arch,
        compute_kernel_type,
        profile_logs,
    )


@dataclass
class RotateRefWeightsAlias:
    """Reference weights alias for Rotate (no weights)."""

    @property
    def ref_tensor_alias(self) -> list[str]:
        return []

    def __call__(self) -> list[str]:
        return self.ref_tensor_alias


@dataclass
class RotateTilertWeightsAlias:
    """TileRT weights alias for Rotate (no weights)."""

    @property
    def tilert_tensor_alias(self) -> list[str]:
        return []

    def __call__(self) -> list[str]:
        return self.tilert_tensor_alias


class RotateAlgorithm(Enum):
    """Rotate algorithm."""

    GENERAL = "general"


class Rotate(TileRTModule):
    """Rotate module: RoPE on first qk_rope_head_dim dims + hadamard transform.

    Unified for deepseek_v3_2 (index_n_heads=64) and glm_5 (index_n_heads=32).
    No weights; uses model_args for dimensions.
    """

    _SUPPORTED_ALGORITHMS = {
        "deepseek_v3_2": [RotateAlgorithm.GENERAL],
        "glm_5": [RotateAlgorithm.GENERAL],
    }

    def __init__(
        self,
        model_args: ModelArgs,
        num_devices: int = 1,
        device_id: int = 0,
        ref_weights_alias: RotateRefWeightsAlias | None = None,
    ):
        super().__init__(
            self.__class__.__name__,
            model_args=model_args,
            num_devices=num_devices,
            device_id=device_id,
        )
        self.tilert_weights_alias = RotateTilertWeightsAlias()
        self.ref_weights_alias = (
            ref_weights_alias if ref_weights_alias is not None else RotateRefWeightsAlias()
        )

        self.qk_rope_head_dim = model_args.qk_rope_head_dim
        self.index_n_heads = model_args.index_n_heads
        self.index_head_dim = model_args.index_head_dim

        self.output: torch.Tensor | None = None
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
        self.output = torch.zeros(
            (batch_size, seq_len, self.index_n_heads, self.index_head_dim),
            dtype=torch.bfloat16,
        )
        self.profile_logs = get_profile_log_tensor()
        self.is_init = True

    def golden_forward(
        self,
        idx_q: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        q_pe_idx, q_nope_idx = torch.split(
            idx_q,
            [self.qk_rope_head_dim, self.index_head_dim - self.qk_rope_head_dim],
            dim=-1,
        )
        q_pe_idx = apply_rotary_emb(q_pe_idx, freqs_cis)
        idx_q = torch.cat([q_pe_idx, q_nope_idx], dim=-1)
        return rotate_activation(idx_q)

    def tilert_forward(self, idx_q: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        assert self.output is not None
        assert self.profile_logs is not None
        freqs_cis_real = torch.view_as_real(freqs_cis).reshape(*freqs_cis.shape[:-1], -1)
        rotate(
            idx_q,
            self.output,
            freqs_cis_real,
            self.profile_logs,
            model_arch=self.model_args.arch_name,
        )
        return self.output
