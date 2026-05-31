"""ProjxWis operation module."""

from dataclasses import dataclass
from enum import Enum

import torch

from tilert.models.base import TileRTModule
from tilert.models.common import init_func
from tilert.models.deepseek_v3_2.model_args import ModelArgs
from tilert.utils import get_profile_log_tensor

__all__ = [
    "projx_wis",
    "ProjxWis",
    "ProjxWisRefWeightsAlias",
    "ProjxWisTilertWeightsAlias",
]


def projx_wis(
    x_in: torch.Tensor,
    w: torch.Tensor,
    output: torch.Tensor,
    compute_kernel_type: str,
    profile_logs: torch.Tensor,
    model_arch: str,
) -> None:
    """
    Define the ProjxWis operation.

    Args:
        x_in: Input tensor.
        w: Weight tensor.
        output: Output tensor.
        compute_kernel_type: Compute kernel type ("bf16" or "bf16mma").
        profile_logs: Profile logs tensor.
        model_arch: Model architecture ("deepseek_v3_2" or "glm_5").
    """
    torch.ops.tilert.proj_w_op(x_in, w, output, model_arch, compute_kernel_type, profile_logs)


@dataclass
class ProjxWisRefWeightsAlias:
    """Reference weights alias for ProjxWis."""

    w_weights = "self_attn.indexer.weights_proj.weight"

    @property
    def ref_tensor_alias(self) -> list[str]:
        return [self.w_weights]

    def __call__(self) -> list[str]:
        return self.ref_tensor_alias


@dataclass
class ProjxWisTilertWeightsAlias:
    """TileRT weights alias for ProjxWis."""

    w_weights = "id_score_weights"

    @property
    def tilert_tensor_alias(self) -> list[str]:
        return [self.w_weights]

    def __call__(self) -> list[str]:
        return self.tilert_tensor_alias


class ProjxWisAlgorithm(Enum):
    """ProjxWis algorithm."""

    BF16 = "bf16"
    BF16MMA = "bf16mma"


class ProjxWis(TileRTModule):
    """ProjxWis module: linear projection for indexer score weights."""

    _SUPPORTED_ALGORITHMS = {
        "deepseek_v3_2": [ProjxWisAlgorithm.BF16, ProjxWisAlgorithm.BF16MMA],
        "glm_5": [ProjxWisAlgorithm.BF16, ProjxWisAlgorithm.BF16MMA],
    }

    _HMMA_CONFIGS = {
        7168: (4, 16, 7),
        6144: (2, 16, 6),
    }

    def __init__(
        self,
        model_args: ModelArgs,
        num_devices: int,
        device_id: int = 0,
        ref_weights_alias: ProjxWisRefWeightsAlias | None = None,
        compute_kernel_type: str | None = None,
    ):
        super().__init__(
            self.__class__.__name__,
            model_args=model_args,
            num_devices=num_devices,
            device_id=device_id,
        )

        self.tilert_weights_alias = ProjxWisTilertWeightsAlias()
        self.ref_weights_alias = (
            ref_weights_alias if ref_weights_alias is not None else ProjxWisRefWeightsAlias()
        )

        self.ref_tensor_alias = self.ref_weights_alias.ref_tensor_alias

        self.ref_w: torch.Tensor | None = None
        self.tilert_w: torch.Tensor | None = None
        self.output: torch.Tensor | None = None
        self.profile_logs: torch.Tensor | None = None

        self.dim = model_args.dim
        self.index_n_heads = model_args.index_n_heads

        if compute_kernel_type is not None:
            self.compute_kernel_type = compute_kernel_type
        else:
            self.compute_kernel_type = "bf16"

    @staticmethod
    def _swizzle_mma_16x16(mat_in: torch.Tensor) -> torch.Tensor:
        """Swizzle a 16x16 BF16 tile for the MMA kernel."""
        assert mat_in.shape[-2] == 16 and mat_in.shape[-1] == 16
        pre_shape = mat_in.shape[:-2]
        mat_in = mat_in.reshape(*pre_shape, 2, 8, 2, 4, 2).transpose(-4, -3).transpose(-5, -4)
        return mat_in.reshape(*pre_shape, 2 * 2, 8 * 4, 2).transpose(-3, -2)

    @staticmethod
    def _to_hmma_layout(
        w_orig: torch.Tensor, n_ctas: int, rows_per_cta: int, x_dim: int, num_pages: int
    ) -> torch.Tensor:
        """Convert [output_dim, x_dim] BF16 weights to the MMA layout."""
        cols_per_page = x_dim // num_pages
        n_k_tiles = cols_per_page // 16
        w = w_orig.reshape(n_ctas, rows_per_cta, num_pages, cols_per_page)
        w = w.transpose(1, 2)
        n_row_tiles = rows_per_cta // 16
        w = w.reshape(n_ctas, num_pages, n_row_tiles, 16, n_k_tiles, 16).transpose(-3, -2)
        w = ProjxWis._swizzle_mma_16x16(w)
        return w.reshape(n_ctas, -1).contiguous()

    @property
    def tilert_tensor_alias(self) -> list[str]:
        return self.tilert_weights_alias.tilert_tensor_alias

    def get_weights_list(self) -> list[torch.Tensor]:
        return [self.tilert_w]

    def device_sharding(self, weights_map: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Device sharding: replicate weight for each device.

        Args:
            weights_map: Map from ref weight alias to tensor.

        Returns:
            Map from tilert weight alias to (num_devices, ...) tensors.
        """
        w = weights_map[self.ref_weights_alias.w_weights]
        if self.compute_kernel_type == "bf16mma":
            n_ctas, rows_per_cta, num_pages = self._HMMA_CONFIGS[self.dim]
            w_hmma = self._to_hmma_layout(w, n_ctas, rows_per_cta, self.dim, num_pages)
            w_out = w_hmma[None, ...].repeat(self.num_devices, 1, 1)
        else:
            w_out = w[None, ...].repeat(self.num_devices, 1, 1)
        return {self.tilert_weights_alias.w_weights: w_out}

    def init_reference_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        w = state_dict[self.ref_weights_alias.w_weights]
        self.ref_w = w.detach().clone().to(torch.bfloat16)
        self.is_ref_weights_init = True

    def init_tilert_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.tilert_w = state_dict[self.tilert_weights_alias.w_weights].detach().clone()
        self.is_tilert_weights_init = True

    def init_random_weights(self) -> None:
        ref_w = init_func(torch.empty(self.index_n_heads, self.dim, dtype=torch.bfloat16))
        ref_state_dict = dict(zip(self.ref_weights_alias(), [ref_w]))
        self.init_reference_weights(ref_state_dict)
        sharded = self.device_sharding(ref_state_dict)
        self.init_tilert_weights({k: v[self.device_id] for k, v in sharded.items()})

    def init_tilert_vars(self, batch_size: int, seq_len: int) -> None:
        self.output = torch.zeros((batch_size, seq_len, self.index_n_heads), dtype=torch.bfloat16)
        self.profile_logs = get_profile_log_tensor()
        self.is_var_init = True

    def golden_forward(self, x_norm: torch.Tensor) -> torch.Tensor:
        assert self.ref_w is not None
        return torch.nn.functional.linear(x_norm, self.ref_w)

    def tilert_forward(self, x_norm: torch.Tensor) -> torch.Tensor:
        assert self.tilert_w is not None
        assert self.output is not None
        assert self.profile_logs is not None
        projx_wis(
            x_norm,
            self.tilert_w,
            self.output,
            self.compute_kernel_type,
            self.profile_logs,
            model_arch=self.model_args.arch_name,
        )
        return self.output
