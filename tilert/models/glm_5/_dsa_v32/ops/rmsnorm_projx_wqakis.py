"""RMSNormProjxWqakis operation module."""

from enum import Enum

import torch

from tilert.models.base import TileRTModule, TilertWeightsConverter
from tilert.models.common import weight_dequant
from tilert.models.glm_5._dsa_v32.model_args import ModelArgs
from tilert.models.glm_5._dsa_v32.ops.projx_wis import projx_wis
from tilert.models.glm_5._dsa_v32.ops.projx_wqaki import (
    ProjxWqakiWeightsConverter,
    projx_wqaki,
)
from tilert.models.glm_5._dsa_v32.ops.rmsnorm_quant import rmsnorm_quant
from tilert.utils import get_profile_log_tensor

__all__ = [
    "RMSNormProjxWqakis",
]


class RMSNormProjxWqakisWeightsConverter(TilertWeightsConverter):
    """Weight converter for RMSNormProjxWqakis."""

    def __init__(self, model_args: ModelArgs, num_devices: int):
        super().__init__(model_args, num_devices)

    def convert_to_decoupled(
        self, weights: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert weights to decoupled FP8 MMA format.

        Args:
            weights: [gamma, wq_a, wq_a_scale, wki, wki_scale, wis, wis_scale]

        Returns:
            (wqaki_packed, wis_bf16, gamma)
        """
        arch_name = self.model_args.arch_name
        x_rmsnorm_gamma, wq_a, wq_a_scale, wki, wki_scale, wis, _wis_scale = weights

        if arch_name == "deepseek_v3_2":
            wqaki_packed = ProjxWqakiWeightsConverter.convert_dsv32(
                wq_a, wq_a_scale, wki, wki_scale
            )
        elif arch_name == "glm_5":
            wqaki_packed = ProjxWqakiWeightsConverter.convert_glm5_68cta(
                wq_a, wq_a_scale, wki, wki_scale
            )
        else:
            raise ValueError(f"Unsupported architecture: {arch_name}")

        wis_bf16 = wis.to(torch.bfloat16)
        return wqaki_packed, wis_bf16, x_rmsnorm_gamma.float()


class RMSNormProjxWqakisRefWeightsAlias:
    """Reference weight aliases for RMSNormProjxWqakis."""

    x_rmsnorm_gamma = "input_layernorm.weight"
    q_a_weights = "self_attn.q_a_proj.weight"
    q_a_scales = "self_attn.q_a_proj.weight_scale_inv"
    wk_weights = "self_attn.indexer.wk.weight"
    wk_scales = "self_attn.indexer.wk.weight_scale_inv"
    wis_weights = "self_attn.indexer.weights_proj.weight"

    @property
    def ref_tensor_alias(self) -> list[str]:
        return [
            self.x_rmsnorm_gamma,
            self.q_a_weights,
            self.q_a_scales,
            self.wk_weights,
            self.wk_scales,
            self.wis_weights,
        ]

    def __call__(self) -> list[str]:
        return self.ref_tensor_alias


class RMSNormProjxWqakisTilertWeightsAlias:
    """Tilert weight aliases for RMSNormProjxWqakis."""

    x_rmsnorm_gamma = "x_rmsnorm_gamma"
    q_a_weights = "q_a_weights"
    q_a_scales = "q_a_scales"
    wk_weights = "wk_weights"
    wk_scales = "wk_scales"
    wis_weights = "wis_weights"
    wis_scales = "wis_scales"

    @property
    def tilert_tensor_alias(self) -> list[str]:
        return [
            self.x_rmsnorm_gamma,
            self.q_a_weights,
            self.q_a_scales,
            self.wk_weights,
            self.wk_scales,
            self.wis_weights,
            self.wis_scales,
        ]

    def __call__(self) -> list[str]:
        return self.tilert_tensor_alias


class RMSNormProjxWqakisAlgorithm(Enum):
    """RMSNormProjxWqakis algorithm."""

    FP8MMA = "fp8mma"


class RMSNormProjxWqakis(TileRTModule):
    """Decoupled RMSNorm + GEMV(W_q_a, W_ki, W_is)."""

    _SUPPORTED_ALGORITHMS = {
        "deepseek_v3_2": [RMSNormProjxWqakisAlgorithm.FP8MMA],
        "glm_5": [RMSNormProjxWqakisAlgorithm.FP8MMA],
    }

    def __init__(
        self,
        model_args: ModelArgs,
        num_devices: int,
        device_id: int,
        ref_weights_alias: RMSNormProjxWqakisRefWeightsAlias | None = None,
    ):
        super().__init__(
            self.__class__.__name__,
            model_args=model_args,
            num_devices=num_devices,
            device_id=device_id,
        )

        self.tilert_weights_alias = RMSNormProjxWqakisTilertWeightsAlias()
        self.ref_weights_alias = (
            ref_weights_alias
            if ref_weights_alias is not None
            else RMSNormProjxWqakisRefWeightsAlias()
        )

        self.arch_name = self.model_args.arch_name
        self.dim = self.model_args.dim
        self.q_lora_rank = self.model_args.q_lora_rank
        self.idx_head_dim = self.model_args.index_head_dim
        self.idx_score_dim = self.model_args.index_n_heads
        self.block_size = self.model_args.block_size
        self.eps = self.model_args.eps

        self.ref_norm_gamma: torch.Tensor | None = None
        self.ref_wq_a: torch.Tensor | None = None
        self.ref_wki: torch.Tensor | None = None
        self.ref_wis: torch.Tensor | None = None

        self.tilert_norm_gamma: torch.Tensor | None = None
        self.tilert_wqakis: torch.Tensor | None = None
        self.tilert_wis: torch.Tensor | None = None

        self.q_out: torch.Tensor | None = None
        self.ki_out: torch.Tensor | None = None
        self.idx_scores_out: torch.Tensor | None = None
        self.x_rmsnorm_out: torch.Tensor | None = None
        self.x_rmsnorm_quant_out: torch.Tensor | None = None
        self.x_rmsnorm_quant_scale_out: torch.Tensor | None = None
        self.profile_logs: torch.Tensor | None = None
        self.is_init = False

        if self.arch_name == "glm_5":
            self.compute_kernel_type = "fp8mma_68cta"
        else:
            self.compute_kernel_type = "fp8mma"

        self.tilert_tensor_alias: list[str] = [
            "x_rmsnorm_gamma",
            "qakis_weights",
            "qakis_scales",
        ]

    def get_weights_list(self) -> list[torch.Tensor]:
        return [self.tilert_norm_gamma, self.tilert_wqakis, self.tilert_wis]

    def device_sharding(self, weights_map: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Repeat weights for device sharding."""
        input_layernorm_weight = (
            weights_map[self.ref_weights_alias.x_rmsnorm_gamma][None, ...]
            .float()
            .repeat(self.num_devices, 1)
        )
        q_a_proj_weight = weights_map[self.ref_weights_alias.q_a_weights][None, ...].repeat(
            self.num_devices, 1, 1
        )
        q_a_proj_weight_scale = weights_map[self.ref_weights_alias.q_a_scales][None, ...].repeat(
            self.num_devices, 1, 1
        )
        wk_weight = weights_map[self.ref_weights_alias.wk_weights][None, ...].repeat(
            self.num_devices, 1, 1
        )
        wk_weight_scale = weights_map[self.ref_weights_alias.wk_scales][None, ...].repeat(
            self.num_devices, 1, 1
        )
        wis_weight = weights_map[self.ref_weights_alias.wis_weights][None, ...].repeat(
            self.num_devices, 1, 1
        )
        is_n_rows = weights_map[self.ref_weights_alias.wis_weights].shape[0]
        is_scale_rows = (is_n_rows + self.block_size - 1) // self.block_size
        is_scale_cols = self.dim // self.block_size
        wis_weight_scale = torch.ones(
            self.num_devices, is_scale_rows, is_scale_cols, dtype=torch.bfloat16
        )
        return {
            self.tilert_weights_alias.x_rmsnorm_gamma: input_layernorm_weight,
            self.tilert_weights_alias.q_a_weights: q_a_proj_weight,
            self.tilert_weights_alias.q_a_scales: q_a_proj_weight_scale,
            self.tilert_weights_alias.wk_weights: wk_weight,
            self.tilert_weights_alias.wk_scales: wk_weight_scale,
            self.tilert_weights_alias.wis_weights: wis_weight,
            self.tilert_weights_alias.wis_scales: wis_weight_scale,
        }

    def init_reference_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        aliases = self.ref_weights_alias()
        self.ref_norm_gamma = state_dict[aliases[0]]
        self.ref_wq_a = weight_dequant(state_dict[aliases[1]], state_dict[aliases[2]])
        self.ref_wki = weight_dequant(state_dict[aliases[3]], state_dict[aliases[4]])
        self.ref_wis = state_dict[aliases[5]].to(torch.bfloat16)

        assert self.ref_norm_gamma.shape[-1] == self.dim
        assert self.ref_wq_a.shape == (self.q_lora_rank, self.dim)
        assert self.ref_wki.shape == (self.idx_head_dim, self.dim)
        assert self.ref_wis.shape == (self.idx_score_dim, self.dim)

    def init_tilert_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        tilert_aliases = self.tilert_weights_alias()
        weights_list = [state_dict[alias] for alias in tilert_aliases]
        converter = RMSNormProjxWqakisWeightsConverter(self.model_args, self.num_devices)
        result = converter.convert_to_decoupled(weights_list)
        self.tilert_wqakis, self.tilert_wis, self.tilert_norm_gamma = result

    def init_tilert_vars(self, batch_size: int, seq_len: int) -> None:
        self.q_out = torch.zeros((batch_size, seq_len, self.q_lora_rank), dtype=torch.bfloat16)
        self.ki_out = torch.zeros((batch_size, seq_len, self.idx_head_dim), dtype=torch.bfloat16)
        self.idx_scores_out = torch.zeros(
            (batch_size, seq_len, self.idx_score_dim), dtype=torch.bfloat16
        )
        self.x_rmsnorm_out = torch.zeros((batch_size, seq_len, self.dim), dtype=torch.bfloat16)
        self.x_rmsnorm_quant_out = torch.zeros(
            (batch_size, seq_len, self.dim), dtype=torch.float8_e4m3fn
        )
        self.x_rmsnorm_quant_scale_out = torch.zeros(
            (batch_size, seq_len, self.dim // self.block_size), dtype=torch.float32
        )
        self.profile_logs = get_profile_log_tensor()
        self.is_init = True

    def init_random_weights(self) -> None:
        bs = self.block_size
        dim_scale_dim = self.dim // bs
        q_scale_dim = (self.q_lora_rank + bs - 1) // bs
        ki_scale_dim = (self.idx_head_dim + bs - 1) // bs
        scale_dtype = torch.float32 if self.arch_name == "glm_5" else torch.bfloat16

        tensor_list = [
            torch.randn(self.dim, dtype=torch.float32),
            torch.randn(self.q_lora_rank, self.dim, dtype=torch.bfloat16).to(torch.float8_e4m3fn),
            torch.randn(q_scale_dim, dim_scale_dim, dtype=scale_dtype),
            torch.randn(self.idx_head_dim, self.dim, dtype=torch.bfloat16).to(torch.float8_e4m3fn),
            torch.randn(ki_scale_dim, dim_scale_dim, dtype=scale_dtype),
            torch.randn(self.idx_score_dim, self.dim, dtype=torch.bfloat16),
        ]
        ref_state_dict = dict(zip(self.ref_weights_alias(), tensor_list))
        self.init_reference_weights(ref_state_dict)
        self.init_tilert_weights(
            {_k: _v[self.device_id] for _k, _v in self.device_sharding(ref_state_dict).items()}
        )

    def golden_forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pure PyTorch reference: RMSNorm -> q, ki, idx_scores."""
        assert self.ref_norm_gamma is not None
        assert self.ref_wq_a is not None
        assert self.ref_wki is not None
        assert self.ref_wis is not None

        x_rmsnorm = torch.nn.functional.rms_norm(
            x.float(), [x.size(-1)], self.ref_norm_gamma, self.eps
        )
        q_out = torch.matmul(x_rmsnorm.float(), self.ref_wq_a.transpose(0, 1).float())
        ki_out = torch.matmul(x_rmsnorm.float(), self.ref_wki.transpose(0, 1).float())
        idx_scores_out = torch.matmul(x_rmsnorm.float(), self.ref_wis.transpose(0, 1).float())
        return (
            q_out.to(torch.bfloat16),
            ki_out.to(torch.bfloat16),
            idx_scores_out.to(torch.bfloat16),
        )

    def tilert_forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run RMSNorm + ProjXWqaki + ProjXWis via TileRT CUDA kernels."""
        rmsnorm_quant(
            x.to(torch.bfloat16),
            self.tilert_norm_gamma,
            self.x_rmsnorm_out,
            self.x_rmsnorm_quant_out,
            self.x_rmsnorm_quant_scale_out,
            self.profile_logs,
            model_arch=self.model_args.arch_name,
        )
        projx_wqaki(
            self.x_rmsnorm_quant_out,
            self.x_rmsnorm_quant_scale_out,
            self.tilert_wqakis,
            self.q_out,
            self.ki_out,
            self.profile_logs,
            self.compute_kernel_type,
            model_arch=self.model_args.arch_name,
        )
        wis_compute_kernel_type = "bf16"
        projx_wis(
            self.x_rmsnorm_out,
            self.tilert_wis,
            self.idx_scores_out,
            wis_compute_kernel_type,
            self.profile_logs,
            model_arch=self.model_args.arch_name,
        )

        return self.q_out, self.ki_out, self.idx_scores_out

    def __call__(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.golden_forward(x)
