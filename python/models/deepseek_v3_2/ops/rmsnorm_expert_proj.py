"""RMSNormExpertProj operation module."""

from dataclasses import dataclass

import torch
from torch import nn

from tilert.models.base import TileRTModule
from tilert.models.common import RMSNorm, init_func, linear
from tilert.models.deepseek_v3_2.model_args import ModelArgs
from tilert.profiler.utils import parse_profile_log_tensor
from tilert.utils import get_profile_log_tensor

__all__ = [
    "RMSNormExpertProj",
    "RMSNormExpertProjRefWeightsAlias",
    "RMSNormExpertProjTilertWeightsAlias",
]


@dataclass
class RMSNormExpertProjRefWeightsAlias:
    """Reference weights alias for RMSNormExpertProj."""

    post_attention_layernorm_weight = "post_attention_layernorm.weight"
    mlp_gate_weight = "mlp.gate.weight"

    def __call__(self) -> list[str]:
        return [self.post_attention_layernorm_weight, self.mlp_gate_weight]


@dataclass
class RMSNormExpertProjTilertWeightsAlias:
    """TileRT weights alias for RMSNormExpertProj."""

    unproj_o_gamma = "unproj_o_gamma"
    exp_proj_weights = "exp_proj_weights"

    def __call__(self) -> list[str]:
        return [self.unproj_o_gamma, self.exp_proj_weights]


class RMSNormExpertProj(TileRTModule):
    """RMS Norm followed by expert projection."""

    def __init__(
        self,
        model_args: ModelArgs,
        num_devices: int,
        device_id: int = 0,
        ref_weights_alias: RMSNormExpertProjRefWeightsAlias | None = None,
        tilert_weights_alias: RMSNormExpertProjTilertWeightsAlias | None = None,
    ):
        super().__init__(
            type(self).__name__,
            model_args=model_args,
            num_devices=num_devices,
            device_id=device_id,
        )
        self.dim = model_args.dim
        self.eps = model_args.eps

        self.ref_weights_alias = (
            ref_weights_alias
            if ref_weights_alias is not None
            else RMSNormExpertProjRefWeightsAlias()
        )
        self.tilert_weights_alias = (
            tilert_weights_alias
            if tilert_weights_alias is not None
            else RMSNormExpertProjTilertWeightsAlias()
        )

        self.is_ref_weights_init = False
        self.is_tilert_weights_init = False

        self.ref_rmsnorm: RMSNorm | None = None
        self.ref_proj_weight: torch.Tensor | None = None
        self.proj_weight = nn.Parameter(
            init_func(torch.empty(model_args.n_routed_experts, model_args.dim))
        )
        self.n_routed_experts = model_args.n_routed_experts

        self.tilert_proj_weight: torch.Tensor | None = None
        self.tilert_rms_norm_weight: torch.Tensor | None = None

        self.profile_logs = get_profile_log_tensor()

    def get_weights_list(self) -> list[torch.Tensor]:
        return [self.tilert_rms_norm_weight, self.tilert_proj_weight]

    def device_sharding(
        self, rms_norm_weight: torch.Tensor, proj_weight: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return rms_norm_weight.float().contiguous(), proj_weight.contiguous()

    def init_reference_weights(
        self, state_dict: dict[str, torch.Tensor], device_id: int | None = None
    ) -> None:
        del device_id
        self.ref_rmsnorm = RMSNorm(self.dim, self.eps)
        self.ref_rmsnorm.weight.data = state_dict[
            self.ref_weights_alias.post_attention_layernorm_weight
        ]
        self.ref_proj_weight = state_dict[self.ref_weights_alias.mlp_gate_weight]
        self.is_ref_weights_init = True

    def init_tilert_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.tilert_proj_weight = (
            state_dict[self.tilert_weights_alias.exp_proj_weights].detach().clone()
        )
        self.tilert_rms_norm_weight = (
            state_dict[self.tilert_weights_alias.unproj_o_gamma].detach().clone()
        )
        self.is_tilert_weights_init = True

    def init_random_weights(self) -> None:
        proj_weight = torch.randn(self.n_routed_experts, self.dim)
        rms_norm_weight = torch.randn(self.dim, dtype=torch.float32)
        ref_state_dict = dict(
            zip(
                self.ref_weights_alias(),
                [rms_norm_weight, proj_weight],
            )
        )
        self.init_reference_weights(ref_state_dict)
        assert self.ref_rmsnorm is not None and self.ref_proj_weight is not None
        sharded_weights = self.device_sharding(self.ref_rmsnorm.weight, self.ref_proj_weight)
        self.init_tilert_weights(dict(zip(self.tilert_weights_alias(), sharded_weights)))

    def golden_forward(
        self, x_in: torch.Tensor, residual: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.is_ref_weights_init, "Reference weights must be initialized before forward pass"
        assert self.ref_rmsnorm is not None and self.ref_proj_weight is not None
        norm_x = self.ref_rmsnorm(x_in, residual)
        scores = linear(norm_x.view(-1, self.dim).float(), self.ref_proj_weight.float())
        return norm_x, scores

    def tilert_forward(self, x_in: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.is_tilert_weights_init, "Tilert weights must be initialized before forward pass"
        assert self.tilert_rms_norm_weight is not None and self.tilert_proj_weight is not None
        x_in = x_in.to(torch.bfloat16)
        hidden_out = torch.zeros_like(x_in)
        scores_out = torch.zeros(
            (x_in.shape[0], x_in.shape[1], self.n_routed_experts), dtype=torch.float32
        )
        torch.ops.tilert.rmsnorm_expert_proj_op(
            x_in,
            self.tilert_rms_norm_weight,
            self.tilert_proj_weight,
            scores_out,
            hidden_out,
            self.profile_logs,
        )
        if self.flag_enable_profiling_log:
            parse_profile_log_tensor(
                self.profile_logs, self.get_profile_log_path(), [(self.op_name, 0.0)]
            )
        return hidden_out, scores_out

    def __call__(self, x_in: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.tilert_forward(x_in)
