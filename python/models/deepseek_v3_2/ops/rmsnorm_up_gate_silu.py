"""RMSNormUpGateSiLU operation module."""

from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn.functional as F

from tilert.models.base import TileRTModule
from tilert.models.common import weight_dequant
from tilert.models.deepseek_v3_2.model_args import ModelArgs
from tilert.models.deepseek_v3_2.ops.expert_sel_up_gate_silu import (
    ExpertSelectUpGateSiLU,
    ExpertSelectUpGateSiLUWeightsConverter,
)
from tilert.profiler.utils import parse_profile_log_tensor
from tilert.utils import get_profile_log_tensor

__all__ = [
    "RMSNormUpGateSiLUAlgorithm",
    "RMSNormUpGateSiLU",
    "RMSNormUpGateSiLUTilertWeightsAlias",
    "rmsnorm_up_gate_silu",
]


def rmsnorm_up_gate_silu(
    hidden_in: torch.Tensor,
    gamma_in: torch.Tensor,
    weights_in: torch.Tensor,
    hidden_out: torch.Tensor,
    profile_logs: torch.Tensor,
    compute_kernel_type: str = "fp8mma",
) -> None:
    """rmsnorm_up_gate_silu operation."""
    torch.ops.tilert.rmsnorm_up_gate_silu_op(
        hidden_in,
        gamma_in,
        weights_in,
        hidden_out,
        profile_logs,
        compute_kernel_type,
    )


class RMSNormUpGateSiLUAlgorithm(Enum):
    """RMSNormUpGateSiLU algorithm"""

    FP8MMA = "fp8mma"
    FP16MMA = "fp16mma"


RMSNormUpGateSiLUWeightsConverter = ExpertSelectUpGateSiLUWeightsConverter
ExpertSelectUpGateSiLUW = ExpertSelectUpGateSiLUWeightsConverter


@dataclass
class RMSNormUpGateSiLUTilertWeightsAlias:
    """TileRT weights alias for RMSNormUpGateSiLU."""

    unproj_o_gamma = "unproj_o_gamma"
    gate_weights = "gate_weights"
    gate_scales = "gate_scales"
    up_weights = "up_weights"
    up_scales = "up_scales"

    @property
    def tilert_tensor_alias(self) -> list[str]:
        return [
            self.unproj_o_gamma,
            self.gate_weights,
            self.gate_scales,
            self.up_weights,
            self.up_scales,
        ]

    def __call__(self) -> list[str]:
        return self.tilert_tensor_alias


class RMSNormUpGateSiLU(TileRTModule):
    """RMSNormUpGateSiLU module"""

    def __init__(
        self,
        model_args: ModelArgs,
        device_id: int,
        num_devices: int,
        algorithm: RMSNormUpGateSiLUAlgorithm = RMSNormUpGateSiLUAlgorithm.FP8MMA,
    ):
        super().__init__(
            self.__class__.__name__,
            model_args=model_args,
            device_id=device_id,
            num_devices=num_devices,
        )

        self.arch_name = self.model_args.arch_name
        self.dim = self.model_args.dim

        self.inter_dim = self.model_args.inter_dim
        self.moe_inter_dim = self.model_args.moe_inter_dim
        self.moe_inter_dim_per_device = self.moe_inter_dim // self.num_devices
        self.inter_dim_per_device = self.inter_dim // self.num_devices
        # effective number of experts
        self.n_experts = self.inter_dim_per_device // self.moe_inter_dim_per_device
        self.eps = self.model_args.eps

        self.block_size = self.model_args.block_size
        self.algorithm = algorithm

        # reference weights
        self.ref_norm_gamma: torch.Tensor | None = None
        self.ref_gate: torch.Tensor | None = None
        self.ref_up: torch.Tensor | None = None

        # tilert weights
        self.tilert_norm_gamma: torch.Tensor | None = None
        self.tilert_weights: torch.Tensor | None = None
        # for compatibility, to be removed in the future
        self.tilert_scales = torch.zeros(
            9, 4, 64, dtype=torch.bfloat16, device=torch.device("cuda")
        )

        # tilert vars
        self.hidden_out: torch.Tensor | None = None

        self.profile_logs: torch.Tensor | None = None
        self.is_init = False

        # tilert_funcs
        self.rmsnorm_up_gate_silu_func = rmsnorm_up_gate_silu

        self.tilert_weights_alias = RMSNormUpGateSiLUTilertWeightsAlias()

        # reference tensor aliases
        self.ref_tensor_alias: list[str] = [
            "post_attention_layernorm.weight",
            "mlp.gate_proj.weight",
            "mlp.gate_proj.weight_scale_inv",
            "mlp.up_proj.weight",
            "mlp.up_proj.weight_scale_inv",
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
        return [self.tilert_norm_gamma, self.tilert_weights, self.tilert_scales]

    def device_sharding(
        self,
        weights_dict: dict[str, torch.Tensor],
        key_prefix: str,  # e.g. model.layers.{layer_id}.mlp
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Device sharding.

        Args:
            weights_dict: Dictionary of weights.

        Returns:
            Tuple of weights.
        """
        rmsnorm_gamma_key = f"{key_prefix}.post_attention_layernorm.weight"
        if ".mlp" in key_prefix:
            key_prefix_without_mlp = key_prefix.replace(".mlp", "")
            rmsnorm_gamma_key = f"{key_prefix_without_mlp}.post_attention_layernorm.weight"
        elif key_prefix == "mlp":
            rmsnorm_gamma_key = "post_attention_layernorm.weight"
        rmsnorm_gamma = weights_dict[rmsnorm_gamma_key]
        # repeat rmsnorm_gamma for each device
        rmsnorm_gamma = rmsnorm_gamma[None, :].repeat(self.num_devices, 1)

        gate_weights, gate_scales, up_weights, up_scales = (
            ExpertSelectUpGateSiLU.process_gate_up_weights(
                key_prefix,
                weights_dict,
                self.num_devices,
            )
        )
        # Transpose split so to match the old convertcode
        gate_weights = gate_weights.reshape(self.n_experts, self.num_devices, -1, self.dim)
        gate_weights = gate_weights.transpose(0, 1)
        gate_scales = gate_scales.reshape(
            self.n_experts, self.num_devices, -1, self.dim // self.block_size
        )
        gate_scales = gate_scales.transpose(0, 1)
        up_weights = up_weights.reshape(self.n_experts, self.num_devices, -1, self.dim)
        up_weights = up_weights.transpose(0, 1)
        up_scales = up_scales.reshape(
            self.n_experts, self.num_devices, -1, self.dim // self.block_size
        )
        up_scales = up_scales.transpose(0, 1)
        return (
            rmsnorm_gamma.contiguous(),
            gate_weights.contiguous(),
            gate_scales.contiguous(),
            up_weights.contiguous(),
            up_scales.contiguous(),
        )

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

        gamma = sharded_list[0][device_id]
        gate_weights = sharded_list[1][device_id]
        gate_scales = sharded_list[2][device_id]
        up_weights = sharded_list[3][device_id]
        up_scales = sharded_list[4][device_id]
        self.ref_norm_gamma = gamma
        ref_gate_list = [
            weight_dequant(gate_weights, gate_scales)
            for gate_weights, gate_scales in zip(gate_weights, gate_scales)
        ]
        ref_up_list = [
            weight_dequant(up_weights, up_scales)
            for up_weights, up_scales in zip(up_weights, up_scales)
        ]
        self.ref_gate = torch.stack(ref_gate_list, dim=0)
        self.ref_up = torch.stack(ref_up_list, dim=0)

    def init_tilert_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        """
        Initialize the tilert weights.

        Args:
            state_dict: State dictionary.
        """
        assert self.algorithm is not None, "Algorithm is not set"
        self.tilert_norm_gamma, self.tilert_weights = RMSNormUpGateSiLUWeightsConverter(
            self.model_args, self.num_devices
        ).dispatch(self.algorithm, [state_dict[alias] for alias in self.tilert_weights_alias()])

    def init_tilert_vars(self, batch_size: int, seq_len: int, dev_id: int = 0) -> None:
        """
        Initialize the tilert variables.

        Args:
            batch_size: Batch size.
            seq_len: Sequence length.
        """
        # tilert vars
        self.hidden_out = torch.zeros(
            (
                batch_size,
                seq_len,
                self.n_experts,
                self.moe_inter_dim_per_device,
            ),
            dtype=torch.bfloat16,
            device=f"cuda:{dev_id}",
        )

        self.profile_logs = get_profile_log_tensor(device=f"cuda:{dev_id}")
        self.is_init = True

    def init_random_weights(self, dev_id: int = 0) -> None:
        """
        Initialize the random weights.

        Returns:
            None
        """
        gamma = torch.randn(self.dim, dtype=torch.float32, device=f"cuda:{dev_id}")
        gate_weights = torch.randn(
            self.inter_dim, self.dim, dtype=torch.bfloat16, device=f"cuda:{dev_id}"
        ).to(torch.float8_e4m3fn)
        up_weights = torch.randn(
            self.inter_dim, self.dim, dtype=torch.bfloat16, device=f"cuda:{dev_id}"
        ).to(torch.float8_e4m3fn)
        inter_dim_scale_dim = self.inter_dim // self.block_size
        dim_scale_dim = self.dim // self.block_size
        scale_dtype = torch.float32 if self.arch_name == "glm_5" else torch.bfloat16
        gate_scales = torch.randn(
            inter_dim_scale_dim, dim_scale_dim, dtype=scale_dtype, device=f"cuda:{dev_id}"
        )
        up_scales = torch.randn(
            inter_dim_scale_dim, dim_scale_dim, dtype=scale_dtype, device=f"cuda:{dev_id}"
        )
        tensor_list = [
            gamma,
            gate_weights,
            gate_scales,
            up_weights,
            up_scales,
        ]
        state_dict = dict(zip(self.ref_tensor_alias, tensor_list))
        self.init_reference_weights(state_dict, "mlp", dev_id)
        sharded_list = self.device_sharding(state_dict, "mlp")
        sharded_state_dict = {
            alias: sharded_list[i][dev_id] for i, alias in enumerate(self.tilert_weights_alias())
        }
        self.init_tilert_weights(sharded_state_dict)

    def golden_forward(
        self,
        x_in: torch.Tensor,
    ) -> torch.Tensor:
        assert self.ref_gate is not None
        assert self.ref_up is not None
        bsz = x_in.shape[0]
        seq_len = x_in.shape[1]
        assert bsz == 1
        x_in_rmsnorm = torch.nn.functional.rms_norm(
            x_in.float(), [x_in.size(-1)], self.ref_norm_gamma, self.eps
        )
        hidden_out_list = []
        for s in range(seq_len):
            # ref up-gate silu
            hidden_out_w1_list = []
            hidden_out_w3_list = []

            for i in range(self.n_experts):
                hidden_out_w1_sel = x_in_rmsnorm[0, s].float() @ self.ref_gate[i].float().T
                hidden_out_w3_sel = x_in_rmsnorm[0, s].float() @ self.ref_up[i].float().T
                hidden_out_w1_list.append(hidden_out_w1_sel)
                hidden_out_w3_list.append(hidden_out_w3_sel)
            hidden_out_w1 = torch.stack(hidden_out_w1_list, dim=0)
            hidden_out_w3 = torch.stack(hidden_out_w3_list, dim=0)
            hidden_out = F.silu(hidden_out_w1.float()) * hidden_out_w3.float()
            hidden_out = hidden_out.to(torch.bfloat16)
            hidden_out_list.append(hidden_out)
        hidden_out = torch.stack(hidden_out_list, dim=0)
        hidden_out = hidden_out[None, ...]
        return hidden_out

    def tilert_forward(
        self,
        x_in: torch.Tensor,
    ) -> torch.Tensor:
        assert self.rmsnorm_up_gate_silu_func is not None
        assert self.algorithm is not None, "Algorithm is not set"
        self.rmsnorm_up_gate_silu_func(
            x_in,
            self.tilert_norm_gamma,
            self.tilert_weights,
            self.hidden_out,
            self.profile_logs,
            self.algorithm.value,
        )
        if self.flag_enable_profiling_log:
            parse_profile_log_tensor(
                self.profile_logs, self.get_profile_log_path(), [(self.op_name, 0.0)]
            )
        return self.hidden_out

    def __call__(
        self,
        x_in: torch.Tensor,
    ) -> torch.Tensor:
        return self.golden_forward(x_in)
