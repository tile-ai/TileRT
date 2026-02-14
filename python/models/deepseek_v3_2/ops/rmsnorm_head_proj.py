"""RMSNormHeadProj operation module."""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import torch

from tilert.models.base import TileRTModule, TilertWeightsConverter
from tilert.models.deepseek_v3_2.model_args import ModelArgs
from tilert.utils import get_profile_log_tensor

__all__ = [
    "rmsnorm_head_proj",
    "rmsnorm_head_proj_glm5",
    "RMSNormHeadProj",
    "RMSNormHeadProjTilertWeightsAlias",
]


def rmsnorm_head_proj(
    hidden_in: torch.Tensor,
    gamma_in: torch.Tensor,
    weight_in: torch.Tensor,
    logits_out: torch.Tensor,
    profile_logs: torch.Tensor,
) -> None:
    """RMS Norm Head Projection operation."""
    torch.ops.tilert.rmsnorm_head_proj_op(
        hidden_in,
        gamma_in,
        weight_in,
        logits_out,
        profile_logs,
    )


def rmsnorm_head_proj_dsv32(
    hidden_in: torch.Tensor,
    gamma_in: torch.Tensor,
    weight_in: torch.Tensor,
    hidden_rmsnorm_out: torch.Tensor,
    logits_out: torch.Tensor,
    profile_logs: torch.Tensor,
) -> None:
    """RMS Norm Head Projection operation."""
    del hidden_rmsnorm_out
    torch.ops.tilert.rmsnorm_head_proj_op(
        hidden_in,
        gamma_in,
        weight_in,
        logits_out,
        profile_logs,
    )


def rmsnorm_head_proj_glm5(
    hidden_in: torch.Tensor,
    gamma_in: torch.Tensor,
    weight_in: torch.Tensor,
    hidden_rmsnorm_out: torch.Tensor,
    logits_out: torch.Tensor,
    profile_logs: torch.Tensor,
) -> None:
    """RMS Norm Head Projection operation."""
    torch.ops.tilert.rmsnorm_head_proj_glm5_op(
        hidden_in,
        gamma_in,
        weight_in,
        hidden_rmsnorm_out,
        logits_out,
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
        """
        Convert the weights to general format.

        Args:
            weights_list: List of weights.

        Returns:
            Tuple of weights.
        """
        args = self.model_args
        assert args.arch_name == "deepseek_v3_2" or args.arch_name == "glm_5"

        with torch.inference_mode():
            rmsnorm_gamma, mat_in = weights_list
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

        # reference weights
        self.ref_rmsnorm_gamma: torch.Tensor | None = None
        self.ref_head_proj: torch.Tensor | None = None

        # tilert weights
        self.tilert_rmsnorm_gamma: torch.Tensor | None = None
        self.tilert_head_proj: torch.Tensor | None = None

        # tilert vars
        self.hidden_rmsnorm_out: torch.Tensor | None = None
        self.hidden_out: torch.Tensor | None = None

        self.profile_logs: torch.Tensor | None = None
        self.is_init = False

        # tilert_funcs
        self.rmsnorm_head_proj_func: Callable | None = None

        if self.arch_name == "deepseek_v3_2":
            self.rmsnorm_head_proj_func = rmsnorm_head_proj_dsv32
        elif self.arch_name == "glm_5":
            self.rmsnorm_head_proj_func = rmsnorm_head_proj_glm5
        else:
            raise ValueError(f"Unsupported architecture: {self.arch_name}")

        self.tilert_weights_alias = RMSNormHeadProjTilertWeightsAlias()

        # reference tensor aliases
        self.ref_tensor_alias: list[str] = [
            "model.norm.weight",
            "lm_head.weight",
        ]

    @property
    def tilert_tensor_alias(self) -> list[str]:
        return self.tilert_weights_alias()

    def get_weights_list(self) -> list[torch.Tensor]:
        """
        Get the weights list.

        Returns:
            List of weights.
        """
        return [self.tilert_rmsnorm_gamma, self.tilert_head_proj]

    def device_sharding(
        self,
        weights_dict: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Device sharding.

        Args:
            weights_dict: Dictionary of weights.
            key_prefix: Key prefix.
        Returns:
            Tuple of weights.
        """
        rmsnorm_gamma_key = "model.norm.weight"
        head_proj_key = "lm_head.weight"
        rmsnorm_gamma = weights_dict[rmsnorm_gamma_key][None, ...]
        # repeat number of devices times
        rmsnorm_gamma = rmsnorm_gamma.repeat(self.num_devices, 1)
        head_proj = weights_dict[head_proj_key]

        head_proj = head_proj.reshape(self.num_devices, -1, self.dim)
        return rmsnorm_gamma.contiguous(), head_proj.contiguous()

    def init_reference_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        """
        Initialize the reference weights.

        Args:
            state_dict: State dictionary.
            device_id: Device ID.
        """
        sharded_list = self.device_sharding(state_dict)

        gamma, head_proj = sharded_list[0][self.device_id], sharded_list[1][self.device_id]
        self.ref_rmsnorm_gamma = gamma
        self.ref_head_proj = head_proj

    def init_tilert_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        """
        Initialize the tilert weights.

        Args:
            state_dict: State dictionary.
        """
        assert self.algorithm is not None
        self.tilert_rmsnorm_gamma, self.tilert_head_proj = RMSNormHeadProjWeightsConverter(
            self.model_args, self.num_devices
        ).dispatch(self.algorithm, [state_dict[alias] for alias in self.tilert_weights_alias()])

    def init_tilert_vars(self, batch_size: int, seq_len: int) -> None:
        """
        Initialize the tilert variables.

        Args:
            batch_size: Batch size.
            seq_len: Sequence length.
        """
        # tilert vars
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

    def init_random_weights(self, device_id: int = 0) -> None:
        """Initialize the random weights."""
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
        """
        Forward pass for the down-project module.

        Args:
            hidden_in: Input hidden.

        Returns:
            Output tensor.
        """
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
        assert self.rmsnorm_head_proj_func is not None
        assert self.hidden_out is not None

        self.rmsnorm_head_proj_func(
            hidden_in,
            self.tilert_rmsnorm_gamma,
            self.tilert_head_proj,
            self.hidden_rmsnorm_out,
            self.hidden_out,
            self.profile_logs,
        )
        return self.hidden_out

    def __call__(
        self,
        hidden_in: torch.Tensor,
    ) -> torch.Tensor:
        return self.golden_forward(hidden_in)
