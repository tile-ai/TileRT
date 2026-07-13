"""RMSNormHeadProj operation module."""

from dataclasses import dataclass
from enum import Enum

import torch

from tilert.models.base import TileRTModule, TilertWeightsConverter
from tilert.models.glm_5._dsa_v32.model_args import ModelArgs
from tilert.utils import get_profile_log_tensor

__all__ = [
    "rmsnorm_head_proj",
    "RMSNormHeadProj",
    "RMSNormHeadProjTilertWeightsAlias",
]


def rmsnorm_head_proj(
    hidden_in: torch.Tensor,
    gamma_in: torch.Tensor,
    weight_in: torch.Tensor,
    hidden_rmsnorm_out: torch.Tensor,
    logits_out: torch.Tensor,
    profile_logs: torch.Tensor,
    model_arch: str,
    compute_kernel_type: str = "general",
) -> None:
    torch.ops.tilert.rmsnorm_head_proj_op(
        hidden_in,
        gamma_in,
        weight_in,
        hidden_rmsnorm_out,
        logits_out,
        model_arch,
        compute_kernel_type,
        profile_logs,
    )


class RMSNormHeadProjAlgorithm(Enum):
    """RMSNormHeadProj algorithm"""

    GENERAL = "general"


class RMSNormHeadProjWeightsConverter(TilertWeightsConverter):
    """RMSNormHeadProj weights converter"""

    @staticmethod
    def tilert_to_tilert_native_bf16_warp_gemv(
        tilert_weight_in: torch.Tensor,
    ) -> torch.Tensor:
        """Convert TILERT weights to TILERT native bf16 warp gemv weights."""
        weights = tilert_weight_in.reshape(1010, 16, 7, 1024)
        weights = weights.transpose(1, 2).reshape(7070, 16, 1024)
        return weights.contiguous()

    def convert_to_general(
        self, weights_list: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        args = self.model_args
        assert args.arch_name == "deepseek_v3_2" or args.arch_name == "glm_5"

        with torch.inference_mode():
            rmsnorm_gamma, mat_in = weights_list
            if args.arch_name == "glm_5":
                from tilert.models.glm_5._dsa_v32.ops.head_proj_w16a16_hmma import (
                    swizzle_head_proj_weight_bf16mma,
                )

                weights = swizzle_head_proj_weight_bf16mma(mat_in.contiguous())
                return rmsnorm_gamma.float(), weights
            logits_dim = mat_in.shape[-2]
            dim = mat_in.shape[-1]
            num_steps = dim // 1024
            assert dim % 1024 == 0
            weights = mat_in.reshape(logits_dim // 16, 16, num_steps, 1024)
            weights = weights.transpose(1, 2).reshape(logits_dim // 16 * num_steps, 16, 1024)
            return rmsnorm_gamma.float(), weights


@dataclass
class RMSNormHeadProjTilertWeightsAlias:
    """TileRT weights alias for RMSNormHeadProj."""

    model_norm_weight = "model.norm.weight"
    lm_head_weight = "lm_head.weight"

    @property
    def tilert_tensor_alias(self) -> list[str]:
        return [self.model_norm_weight, self.lm_head_weight]

    def __call__(self) -> list[str]:
        return self.tilert_tensor_alias


class RMSNormHeadProj(TileRTModule):
    """RMSNormHeadProj module"""

    _SUPPORTED_ALGORITHMS = {
        "deepseek_v3_2": [RMSNormHeadProjAlgorithm.GENERAL],
        "glm_5": [RMSNormHeadProjAlgorithm.GENERAL],
    }

    def __init__(
        self,
        model_args: ModelArgs,
        device_id: int,
        num_devices: int,
        algorithm: RMSNormHeadProjAlgorithm = RMSNormHeadProjAlgorithm.GENERAL,
    ):
        super().__init__(
            self.__class__.__name__,
            model_args=model_args,
            device_id=device_id,
            num_devices=num_devices,
        )

        self.arch_name = self.model_args.arch_name
        self.dim = self.model_args.dim
        self.logits_dim = self.model_args.vocab_size
        self.algorithm = algorithm
        self.eps = self.model_args.eps

        self.ref_rmsnorm_gamma: torch.Tensor | None = None
        self.ref_head_proj: torch.Tensor | None = None

        self.tilert_rmsnorm_gamma: torch.Tensor | None = None
        self.tilert_head_proj: torch.Tensor | None = None

        self.hidden_rmsnorm_out: torch.Tensor | None = None
        self.hidden_out: torch.Tensor | None = None

        self.profile_logs: torch.Tensor | None = None
        self.is_init = False

        self.tilert_weights_alias = RMSNormHeadProjTilertWeightsAlias()

        self.ref_tensor_alias: list[str] = [
            "model.norm.weight",
            "lm_head.weight",
        ]

    @property
    def tilert_tensor_alias(self) -> list[str]:
        return self.tilert_weights_alias()

    def get_weights_list(self) -> list[torch.Tensor]:
        """Get the weights list."""
        return [self.tilert_rmsnorm_gamma, self.tilert_head_proj]

    def device_sharding(
        self,
        weights_dict: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rmsnorm_gamma_key = "model.norm.weight"
        head_proj_key = "lm_head.weight"
        rmsnorm_gamma = weights_dict[rmsnorm_gamma_key][None, ...]
        rmsnorm_gamma = rmsnorm_gamma.repeat(self.num_devices, 1)
        head_proj = weights_dict[head_proj_key]

        head_proj = head_proj.reshape(self.num_devices, -1, self.dim)
        return rmsnorm_gamma.contiguous(), head_proj.contiguous()

    def init_reference_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        sharded_list = self.device_sharding(state_dict)

        gamma, head_proj = sharded_list[0][self.device_id], sharded_list[1][self.device_id]
        self.ref_rmsnorm_gamma = gamma
        self.ref_head_proj = head_proj

    def init_tilert_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        assert self.algorithm is not None
        self.tilert_rmsnorm_gamma, self.tilert_head_proj = RMSNormHeadProjWeightsConverter(
            self.model_args, self.num_devices
        ).dispatch(self.algorithm, [state_dict[alias] for alias in self.tilert_weights_alias()])

    def init_tilert_vars(self, batch_size: int, seq_len: int) -> None:
        self.hidden_rmsnorm_out = torch.zeros(
            (batch_size, seq_len, self.dim),
            dtype=torch.bfloat16,
            device=f"cuda:{self.device_id}",
        )
        self.hidden_out = torch.zeros(
            (batch_size, seq_len, self.logits_dim // self.num_devices),
            dtype=torch.float32,
            device=f"cuda:{self.device_id}",
        )
        self.profile_logs = get_profile_log_tensor(device=f"cuda:{self.device_id}")
        self.is_init = True

    def init_random_weights(self, device_id: int | None = None) -> None:
        if device_id is None:
            device_id = self.device_id
        rmsnorm_gamma = torch.randn(self.dim, dtype=torch.float32, device=f"cuda:{device_id}")
        head_proj = torch.randn(
            self.logits_dim, self.dim, dtype=torch.bfloat16, device=f"cuda:{device_id}"
        )

        tensor_list = [
            rmsnorm_gamma,
            head_proj,
        ]
        state_dict = dict(zip(self.ref_tensor_alias, tensor_list))

        self.init_reference_weights(state_dict)
        sharded_list = self.device_sharding(state_dict)
        sharded_state_dict = {
            alias: sharded_list[i][self.device_id]
            for i, alias in enumerate(self.tilert_weights_alias())
        }
        self.init_tilert_weights(sharded_state_dict)

    def golden_forward(
        self,
        hidden_in: torch.Tensor,
    ) -> torch.Tensor:
        assert self.ref_rmsnorm_gamma is not None
        assert self.ref_head_proj is not None
        bsz = hidden_in.shape[0]
        assert bsz == 1
        hidden_rmsnorm = torch.nn.functional.rms_norm(
            hidden_in.float(), [hidden_in.size(-1)], self.ref_rmsnorm_gamma, self.eps
        )
        return hidden_rmsnorm.float() @ self.ref_head_proj.T.float()

    def tilert_forward(
        self,
        hidden_in: torch.Tensor,
    ) -> torch.Tensor:
        assert self.hidden_out is not None

        rmsnorm_head_proj(
            hidden_in,
            self.tilert_rmsnorm_gamma,
            self.tilert_head_proj,
            self.hidden_rmsnorm_out,
            self.hidden_out,
            self.profile_logs,
            model_arch=self.model_args.arch_name,
        )
        return self.hidden_out

    def __call__(
        self,
        hidden_in: torch.Tensor,
    ) -> torch.Tensor:
        return self.golden_forward(hidden_in)
