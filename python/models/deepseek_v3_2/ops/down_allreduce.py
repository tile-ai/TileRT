"""DownAllreduce operation module."""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

# import torch.nn.functional as F
import torch

from tilert.models.base import TileRTModule
from tilert.models.common import weight_dequant
from tilert.models.deepseek_v3_2.model_args import ModelArgs
from tilert.models.deepseek_v3_2.ops.expert_down_allreduce import (
    ExpertDownAllReduceWeightsConverter,
)
from tilert.utils import get_profile_log_tensor

__all__ = [
    "down_allreduce",
    "down_allreduce_glm5",
    "DownAllReduceAlgorithm",
    "DownAllReduce",
    "DownAllReduceTilertWeightsAlias",
]


def down_allreduce(
    vec_in: torch.Tensor,
    mat_in: torch.Tensor,
    mat_scale: torch.Tensor,
    x_in: torch.Tensor,
    flag: int,
    vec_out: torch.Tensor,
    profile_logs: torch.Tensor,
) -> None:
    """
    Fused operation of down and allreduce.

    Args:
        vec_in: Input tensor.
        mat_in: Input tensor.
        mat_scale: Input tensor.
        x_in: Input tensor.
        flag: Input flag.
        vec_out: Output tensor.
        profile_logs: Profile logs tensor. This is a 1D tensor of shape
                      (num_sms,) to store the profile logs of the down_allreduce
                      operation, where num_sms is the number of SMs on the
                      device.
    """
    torch.ops.tilert.down_allreduce_op(
        vec_in,
        mat_in,
        mat_scale,
        x_in,
        flag,
        vec_out,
        profile_logs,
    )


def down_allreduce_glm5(
    vec_in: torch.Tensor,
    mat_in: torch.Tensor,
    mat_scale: torch.Tensor,
    x_in: torch.Tensor,
    flag: int,
    vec_out: torch.Tensor,
    profile_logs: torch.Tensor,
) -> None:
    """
    Fused operation of down and allreduce.

    Args:
        vec_in: Input tensor.
        mat_in: Input tensor.
        mat_scale: Input tensor.
        x_in: Input tensor.
        flag: Input flag.
        vec_out: Output tensor.
        profile_logs: Profile logs tensor. This is a 1D tensor of shape
                      (num_sms,) to store the profile logs of the down_allreduce
                      operation, where num_sms is the number of SMs on the
                      device.
    """
    torch.ops.tilert.down_allreduce_glm5_op(
        vec_in,
        mat_in,
        mat_scale,
        x_in,
        flag,
        vec_out,
        profile_logs,
    )


class DownAllReduceAlgorithm(Enum):
    """DownAllReduce algorithm"""

    GENERAL = "general"


DownAllReduceWeightsConverter = ExpertDownAllReduceWeightsConverter


@dataclass
class DownAllReduceTilertWeightsAlias:
    """TileRT weights alias for DownAllReduce."""

    down_weights = "down_weights"
    down_scales = "down_scales"

    @property
    def tilert_tensor_alias(self) -> list[str]:
        return [self.down_weights, self.down_scales]

    def __call__(self) -> list[str]:
        return self.tilert_tensor_alias


class DownAllReduce(TileRTModule):
    """DownAllReduce module"""

    def __init__(
        self,
        model_args: ModelArgs,
        device_id: int,
        num_devices: int,
        algorithm: DownAllReduceAlgorithm = DownAllReduceAlgorithm.GENERAL,
    ):
        super().__init__(
            self.__class__.__name__,
            device_id=device_id,
            model_args=model_args,
            num_devices=num_devices,
        )

        self.arch_name = self.model_args.arch_name
        self.dim = self.model_args.dim

        self.inter_dim = self.model_args.inter_dim
        self.moe_inter_dim = self.model_args.moe_inter_dim
        self.moe_inter_dim_per_device = self.moe_inter_dim // self.num_devices
        self.inter_dim_per_device = self.inter_dim // self.num_devices
        # effective number of experts
        self.n_experts: int = self.inter_dim_per_device // self.moe_inter_dim_per_device
        self.block_size = self.model_args.block_size
        self.dim_scale_dim = self.dim // self.block_size
        self.in_scale_dim = self.inter_dim // self.block_size
        self.moe_inter_scale_dim_per_device = self.moe_inter_dim_per_device // self.block_size
        self.algorithm = algorithm

        # reference weights
        self.ref_down: torch.Tensor | None = None

        # tilert weights
        self.tilert_weights: torch.Tensor | None = None
        self.tilert_scales: torch.Tensor | None = None

        # tilert vars
        self.hidden_out: torch.Tensor | None = None

        self.profile_logs: torch.Tensor | None = None
        self.is_init = False

        # tilert_funcs
        self.down_allreduce_func: Callable | None = None

        if self.arch_name == "deepseek_v3_2":
            self.down_allreduce_func = down_allreduce
        elif self.arch_name == "glm_5":
            self.down_allreduce_func = down_allreduce_glm5
        else:
            raise ValueError(f"Unsupported architecture: {self.arch_name}")

        self.tilert_weights_alias = DownAllReduceTilertWeightsAlias()

        # for device sharding, corresponding to the output of device_sharding
        # and input of tilert_forward
        self.tensor_alias: list[str] = [
            "down_weights",
            "down_scales",
        ]

        # reference tensor aliases
        self.ref_tensor_alias: list[str] = [
            "mlp.down_proj.weight",
            "mlp.down_proj.weight_scale_inv",
        ]

    @property
    def tilert_tensor_alias(self) -> list[str]:
        return self.tilert_weights_alias.tilert_tensor_alias

    def get_weights_list(self) -> list[torch.Tensor]:
        """
        Get the weights list.

        Returns:
            List of weights.
        """
        return [self.tilert_weights, self.tilert_scales]

    def device_sharding(
        self,
        weights_dict: dict[str, torch.Tensor],
        key_prefix: str,  # e.g. model.layers.{layer_id}.mlp
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Device sharding.

        Args:
            weights_dict: Dictionary of weights.
            key_prefix: Key prefix.
        Returns:
            Tuple of weights.
        """
        down_proj_weight_key = f"{key_prefix}.down_proj.weight"
        down_proj_scale_key = f"{key_prefix}.down_proj.weight_scale_inv"
        down_proj_weight = weights_dict[down_proj_weight_key]
        down_proj_scale = weights_dict[down_proj_scale_key]
        # To match the old convertcode
        down_proj_weight = down_proj_weight.reshape(
            self.dim, self.n_experts, self.num_devices, self.moe_inter_dim_per_device
        )
        down_proj_weight_splited = torch.split(down_proj_weight, 1, dim=2)

        down_proj_weight_splited = [
            down_proj_weight_splited[i]
            .reshape(self.dim, self.n_experts, self.moe_inter_dim_per_device)
            .transpose(0, 1)
            .contiguous()
            for i in range(self.num_devices)
        ]

        down_proj_scale = down_proj_scale.reshape(
            self.dim_scale_dim,
            self.n_experts,
            self.num_devices,
            self.moe_inter_scale_dim_per_device,
        )
        down_proj_scale_splited = torch.split(down_proj_scale, 1, dim=2)
        down_proj_scale_splited = [
            down_proj_scale_splited[i]
            .reshape(self.dim_scale_dim, self.n_experts, self.moe_inter_scale_dim_per_device)
            .transpose(0, 1)
            .contiguous()
            for i in range(self.num_devices)
        ]
        down_weights = torch.stack(down_proj_weight_splited, dim=0)
        down_scales = torch.stack(down_proj_scale_splited, dim=0)
        return down_weights.contiguous(), down_scales.contiguous()

    def init_reference_weights(
        self,
        state_dict: dict[str, torch.Tensor],
        key_prefix: str,  # e.g. model.layers.{layer_id}.mlp
        device_id: int = 0,
    ) -> None:
        """
        Initialize the reference weights.

        Args:
            state_dict: State dictionary.
            device_id: Device ID.
        """
        sharded_list = self.device_sharding(state_dict, key_prefix)

        down_weights = sharded_list[0][device_id]
        down_scales = sharded_list[1][device_id]

        down_list = [
            weight_dequant(down_weight, down_scale)
            for down_weight, down_scale in zip(down_weights, down_scales)
        ]
        self.ref_down = torch.stack(down_list, dim=0)

    def init_tilert_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        """
        Initialize the tilert weights.

        Args:
            state_dict: State dictionary.
        """
        assert self.algorithm is not None, "Algorithm is not set"
        self.tilert_weights, self.tilert_scales = DownAllReduceWeightsConverter(
            self.model_args, self.num_devices
        ).dispatch(self.algorithm, [state_dict[alias] for alias in self.tensor_alias])

    def init_tilert_vars(self, batch_size: int, seq_len: int, device_id: int = 0) -> None:
        """
        Initialize the tilert variables.

        Args:
            batch_size: Batch size.
            seq_len: Sequence length.
        """
        # tilert vars
        self.hidden_out = torch.zeros(
            (batch_size, seq_len, self.dim),
            dtype=torch.bfloat16,
            device=f"cuda:{device_id}",
        )
        self.profile_logs = get_profile_log_tensor(device=f"cuda:{device_id}")
        self.is_init = True

    def init_random_weights(self, device_id: int = 0) -> None:
        """Initialize the random weights."""
        scale_dtype = torch.float32 if self.arch_name == "glm_5" else torch.bfloat16
        down_weights = torch.randn(
            self.dim, self.inter_dim, dtype=torch.bfloat16, device=f"cuda:{device_id}"
        ).to(torch.float8_e4m3fn)

        inter_dim_scale_dim = self.inter_dim // self.block_size
        dim_scale_dim = self.dim // self.block_size
        down_scales = torch.randn(
            dim_scale_dim, inter_dim_scale_dim, dtype=scale_dtype, device=f"cuda:{device_id}"
        )
        tensor_list = [
            down_weights,
            down_scales,
        ]
        state_dict = dict(zip(self.ref_tensor_alias, tensor_list))

        self.init_reference_weights(state_dict, "mlp", device_id)
        sharded_list = self.device_sharding(state_dict, "mlp")

        sharded_state_dict = {
            alias: sharded_list[i][device_id] for i, alias in enumerate(self.tensor_alias)
        }
        self.init_tilert_weights(sharded_state_dict)

    def golden_forward(
        self,
        vec_in: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for the down-project module.

        Args:
            vec_in: Input vector.

        Returns:
            Output tensor.
        """
        assert self.ref_down is not None
        bsz = vec_in.shape[0]
        assert bsz == 1
        seq_len = vec_in.shape[1]
        hidden_out_list = []
        for s in range(seq_len):
            hidden_out_w2_list = []
            for i in range(self.n_experts):
                hidden_out_w2_sel = vec_in[0, s, i].float() @ self.ref_down[i].float().T
                hidden_out_w2_list.append(hidden_out_w2_sel)
            hidden_out_w2 = torch.stack(hidden_out_w2_list, dim=0).to(torch.bfloat16)
            hidden_out_w2 = torch.sum(hidden_out_w2, dim=0)
            hidden_out_list.append(hidden_out_w2)
        return torch.stack(hidden_out_list, dim=0)[None, ...]

    def tilert_forward(
        self,
        vec_in: torch.Tensor,
        x_in: torch.Tensor,
        flag: int,
    ) -> torch.Tensor:
        assert self.down_allreduce_func is not None
        assert self.hidden_out is not None
        self.down_allreduce_func(
            vec_in,
            self.tilert_weights,
            self.tilert_scales,
            x_in,
            flag,
            self.hidden_out,
            self.profile_logs,
        )
        return self.hidden_out

    def __call__(
        self,
        x_in: torch.Tensor,
    ) -> torch.Tensor:
        return self.golden_forward(x_in)
