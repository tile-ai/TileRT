"""ExpertSelectUpGateSiLU operation module."""

from dataclasses import dataclass
from enum import Enum

import numpy as np

# from typing import Any
import torch
import torch.nn.functional as F

from tilert.models.base import TileRTModule, TilertWeightsConverter
from tilert.models.common import weight_dequant
from tilert.models.deepseek_v3_2.model_args import ModelArgs
from tilert.profiler.utils import parse_profile_log_tensor
from tilert.utils import get_profile_log_tensor

__all__ = [
    "ExpertSelectUpGateSiLUAlgorithm",
    "ExpertSelectUpGateSiLU",
    "ExpertSelectUpGateSiLURefWeightsAlias",
    "ExpertSelectUpGateSiLUTilertWeightsAlias",
    "expert_select_up_gate_silu",
]


def expert_select_up_gate_silu(
    hidden_in: torch.Tensor,
    scores_in: torch.Tensor,
    bias_in: torch.Tensor,
    experts_weights_in: torch.Tensor,
    hidden_out: torch.Tensor,
    expert_probs_out: torch.Tensor,
    expert_indices_out: torch.Tensor,
    profile_logs: torch.Tensor,
    algorithm: str = "fp8mma",
) -> None:
    """Expert SelectUpGateSiLU operation."""
    args_list = [
        hidden_in,
        scores_in,
        bias_in,
        experts_weights_in,
        hidden_out,
        expert_probs_out,
        expert_indices_out,
        profile_logs,
        algorithm,
    ]
    torch.ops.tilert.expert_select_up_gate_silu_op(*args_list)


@dataclass
class ExpertSelectUpGateSiLURefWeightsAlias:
    """Reference weights alias for ExpertSelectUpGateSiLU."""

    key_prefix: str = "mlp"
    n_routed_experts: int = 256

    @property
    def ref_tensor_alias(self) -> list[str]:
        n = self.n_routed_experts
        return (
            [f"{self.key_prefix}.gate.e_score_correction_bias"]
            + [f"{self.key_prefix}.shared_experts.gate_proj.weight"]
            + [f"{self.key_prefix}.experts.{i}.gate_proj.weight" for i in range(n)]
            + [f"{self.key_prefix}.shared_experts.up_proj.weight"]
            + [f"{self.key_prefix}.experts.{i}.up_proj.weight" for i in range(n)]
            + [f"{self.key_prefix}.shared_experts.gate_proj.weight_scale_inv"]
            + [f"{self.key_prefix}.experts.{i}.gate_proj.weight_scale_inv" for i in range(n)]
            + [f"{self.key_prefix}.shared_experts.up_proj.weight_scale_inv"]
            + [f"{self.key_prefix}.experts.{i}.up_proj.weight_scale_inv" for i in range(n)]
        )

    def __call__(self) -> list[str]:
        return self.ref_tensor_alias


@dataclass
class ExpertSelectUpGateSiLUTilertWeightsAlias:
    """TileRT weights alias for ExpertSelectUpGateSiLU."""

    exp_bias = "exp_bias"
    exp_gate_weights = "exp_gate_weights"
    exp_gate_scales = "exp_gate_scales"
    exp_up_weights = "exp_up_weights"
    exp_up_scales = "exp_up_scales"

    @property
    def tilert_tensor_alias(self) -> list[str]:
        return [
            self.exp_bias,
            self.exp_gate_weights,
            self.exp_gate_scales,
            self.exp_up_weights,
            self.exp_up_scales,
        ]

    def __call__(self) -> list[str]:
        return self.tilert_tensor_alias


class ExpertSelectUpGateSiLUAlgorithm(Enum):
    """ExpertSelectUpGateSiLU algorithm"""

    FP8MMA = "fp8mma"
    FP16MMA = "fp16mma"


class ExpertSelectUpGateSiLUWeightsConverter(TilertWeightsConverter):
    """ExpertSelectUpGateSiLU weights converter"""

    @staticmethod
    def _swizzle_qmma_16x32(mat_in: torch.Tensor) -> torch.Tensor:
        assert mat_in.shape[-2] == 16 and mat_in.shape[-1] == 32
        assert mat_in.dtype == torch.float8_e4m3fn
        # PTX isa fig.88
        pre_shape = mat_in.shape[:-2]
        mat_in = mat_in.reshape(*pre_shape, 2, 8, 2, 4, 4).transpose(-4, -3).transpose(-5, -4)
        return mat_in.reshape(*pre_shape, 2 * 2, 8 * 4, 4).transpose(-3, -2)

    @staticmethod
    def _swizzle_mma_16x32(mat_in: torch.Tensor) -> torch.Tensor:
        assert mat_in.shape[-2] == 16 and mat_in.shape[-1] == 32
        # PTX isa fig.88
        pre_shape = mat_in.shape[:-2]
        mat_in = mat_in.reshape(*pre_shape, 2, 8, 2, 4, 4).transpose(-4, -3).transpose(-5, -4)
        return mat_in.reshape(*pre_shape, 2 * 2, 8 * 4, 4).transpose(-3, -2)

    @staticmethod
    def _swizzle_mma_16x16(mat_in: torch.Tensor) -> torch.Tensor:
        assert mat_in.shape[-2] == 16 and mat_in.shape[-1] == 16
        # PTX isa fig.88
        pre_shape = mat_in.shape[:-2]
        mat_in = mat_in.reshape(*pre_shape, 2, 8, 2, 4, 2).transpose(-4, -3).transpose(-5, -4)
        return mat_in.reshape(*pre_shape, 2 * 2, 8 * 4, 2).transpose(-3, -2)

    @staticmethod
    def tilert_to_tilert_144sm(
        mat_in: torch.Tensor, mat_scale_in: torch.Tensor, mma_type: str | None = None
    ) -> torch.Tensor:
        """
        Convert tilert weights and scales to tilert_144sm input format.

        Args:
            mat_in: tilert weights
            mat_scale_in: tilert scales
            mma_type: MMA type, None,"16x32" or "16x16"
        Returns:
            tilert_144sm weights and scales
        """
        exp_num = mat_in.shape[0]
        assert mat_in.shape == (exp_num, 512, 7168)
        assert mat_scale_in.shape == (exp_num, 4, 64)
        weights_trt = mat_in.reshape(exp_num, 128, 4, 7168)
        weights_w1 = weights_trt[:, :, :2].reshape(exp_num, 256, 7168)
        weights_w3 = weights_trt[:, :, 2:].reshape(exp_num, 256, 7168)
        # to 16x1024 blocks
        weights_w1 = weights_w1.reshape(exp_num, 16, 16, 7, 1024).transpose(2, 3)
        weights_w3 = weights_w3.reshape(exp_num, 16, 16, 7, 1024).transpose(2, 3)
        if mma_type == "16x32":
            weights_w1 = weights_w1.reshape(exp_num, 16, 7, 16, 32, 32).transpose(3, 4)
            weights_w1 = ExpertSelectUpGateSiLUWeightsConverter._swizzle_mma_16x32(weights_w1)
            weights_w1 = weights_w1.reshape(exp_num, 16, 7, 16, 1024)
            weights_w3 = weights_w3.reshape(exp_num, 16, 7, 16, 32, 32).transpose(3, 4)
            weights_w3 = ExpertSelectUpGateSiLUWeightsConverter._swizzle_mma_16x32(weights_w3)
            weights_w3 = weights_w3.reshape(exp_num, 16, 7, 16, 1024)
        elif mma_type == "16x16":
            weights_w1 = weights_w1.reshape(exp_num, 16, 7, 16, 64, 16).transpose(3, 4)
            weights_w1 = ExpertSelectUpGateSiLUWeightsConverter._swizzle_mma_16x16(weights_w1)
            weights_w1 = weights_w1.reshape(exp_num, 16, 7, 16, 1024)
            weights_w3 = weights_w3.reshape(exp_num, 16, 7, 16, 64, 16).transpose(3, 4)
            weights_w3 = ExpertSelectUpGateSiLUWeightsConverter._swizzle_mma_16x16(weights_w3)
            weights_w3 = weights_w3.reshape(exp_num, 16, 7, 16, 1024)

        weights = torch.cat([weights_w1, weights_w3], dim=3)
        assert weights.shape == (exp_num, 16, 7, 32, 1024)
        weights = weights.reshape(exp_num, 16, 7, 32 * 1024)

        # For scales, first unswizzle
        scales_unswizzled = torch.zeros(exp_num, 4, 56)
        for i in range(64):
            if ((i % 8) * 8 + i // 8) < 56:
                scales_unswizzled[..., ((i % 8) * 8 + i // 8)] = mat_scale_in[..., i]
        scales_unswizzled = scales_unswizzled.reshape(exp_num, 2, 2, 56)

        scales_w1 = scales_unswizzled[:, :, :1].repeat(1, 1, 8, 1).reshape(exp_num, 16, 1, 7, 8)
        scales_w1 = scales_w1.transpose(2, 3)
        scales_w3 = scales_unswizzled[:, :, 1:].repeat(1, 1, 8, 1).reshape(exp_num, 16, 1, 7, 8)
        scales_w3 = scales_w3.transpose(2, 3)
        scales = torch.cat([scales_w1, scales_w3], dim=3)
        assert scales.shape == (exp_num, 16, 7, 2, 8)
        scales = (
            scales.reshape(exp_num, 16, 7, 2 * 8).to(torch.bfloat16).view(dtype=torch.float8_e4m3fn)
        )
        weights_and_scales = torch.zeros(
            exp_num, 16, 7, 32 * 1024 + 128, dtype=torch.float8_e4m3fn, device=mat_in.device
        )
        weights_and_scales[:, :, :, : 32 * 1024].copy_(weights)
        weights_and_scales[:, :, :, 32 * 1024 : 32 * 1024 + 32].copy_(scales)
        return weights_and_scales

    @staticmethod
    def tilert_to_tilert_144sm_mma(
        mat_in: torch.Tensor, mat_scale_in: torch.Tensor, mma_type: str = "16x32"
    ) -> torch.Tensor:
        """
        Convert tilert weights and scales to tilert_144sm_mma input format.

        Args:
            mat_in: tilert weights
            mat_scale_in: tilert scales
        Returns:
            tilert_144sm weights and scales
        """
        return ExpertSelectUpGateSiLUWeightsConverter.tilert_to_tilert_144sm(
            mat_in, mat_scale_in, mma_type
        )

    def convert_to_mma(
        self, weights_list: list[torch.Tensor], algorithm: str = "fp8mma"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert the weights to mma format.

        Args:
            weights: List of weights.

        Returns:
            Tuple of weights.
        """
        args = self.model_args
        dim = args.dim
        pages = dim // 1024  # 6 for GLM5, 7 for DS v3.2
        dim_scale_dim = dim // args.block_size
        with torch.inference_mode():
            # w1: gate, w3: up
            bias_or_gamma, weights_w1, scales_w1, weights_w3, scales_w3 = weights_list
            exp_num = weights_w1.shape[0]
            # to 16x1024 blocks
            weights_w1 = weights_w1.reshape(exp_num, 16, 16, pages, 1024).transpose(2, 3)
            weights_w3 = weights_w3.reshape(exp_num, 16, 16, pages, 1024).transpose(2, 3)
            # to 16x32 blocks and swizzle
            if algorithm == "fp8mma":
                weights_w1 = weights_w1.reshape(exp_num, 16, pages, 16, 32, 32).transpose(3, 4)
                weights_w1 = self._swizzle_qmma_16x32(weights_w1)
                weights_w1 = weights_w1.reshape(exp_num, 16, pages, 16, 1024)
                weights_w3 = weights_w3.reshape(exp_num, 16, pages, 16, 32, 32).transpose(3, 4)
                weights_w3 = self._swizzle_qmma_16x32(weights_w3)
                weights_w3 = weights_w3.reshape(exp_num, 16, pages, 16, 1024)
            elif algorithm == "fp16mma":
                weights_w1 = weights_w1.reshape(exp_num, 16, pages, 16, 64, 16).transpose(3, 4)
                weights_w1 = self._swizzle_mma_16x16(weights_w1)
                weights_w1 = weights_w1.reshape(exp_num, 16, pages, 16, 1024)
                weights_w3 = weights_w3.reshape(exp_num, 16, pages, 16, 64, 16).transpose(3, 4)
                weights_w3 = self._swizzle_mma_16x16(weights_w3)
                weights_w3 = weights_w3.reshape(exp_num, 16, pages, 16, 1024)
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
            # concat w1 and w3
            weights: torch.Tensor = torch.cat([weights_w1, weights_w3], dim=3)
            assert weights.shape == (exp_num, 16, pages, 32, 1024)
            weights = weights.reshape(exp_num, 16, pages, 32 * 1024)

            scales_w1 = (
                scales_w1.reshape(exp_num, 2, 1, dim_scale_dim)
                .repeat(1, 1, 8, 1)
                .reshape(exp_num, 16, 1, pages, 8)
            )
            scales_w1 = scales_w1.transpose(2, 3)
            scales_w3 = (
                scales_w3.reshape(exp_num, 2, 1, dim_scale_dim)
                .repeat(1, 1, 8, 1)
                .reshape(exp_num, 16, 1, pages, 8)
            )
            scales_w3 = scales_w3.transpose(2, 3)
            scales = torch.cat([scales_w1, scales_w3], dim=3)
            assert scales.shape == (exp_num, 16, pages, 2, 8)

            if self.model_args.arch_name == "glm_5":
                if scales.dtype != torch.float32:
                    print(
                        "Warning: ExpertSelectUpGateSiLUWeightsConverter: "
                        + f"scales.dtype: {scales.dtype} "
                        + "is not float32, convert to float32."
                    )
                scales = scales.to(torch.float32)
            else:  # DS v3.2, use bfloat16 for scales
                scales = scales.to(torch.bfloat16)

            scales = scales.reshape(exp_num, 16, pages, 2 * 8).view(dtype=torch.float8_e4m3fn)

            weights_and_scales = torch.zeros(
                exp_num,
                16,
                pages,
                32 * 1024 + 128,
                dtype=torch.float8_e4m3fn,
                device=weights_w1.device,
            )
            weights_and_scales[:, :, :, : 32 * 1024].copy_(weights)
            weights_and_scales[:, :, :, 32 * 1024 : 32 * 1024 + scales.shape[-1]].copy_(scales)

            return bias_or_gamma.float(), weights_and_scales.contiguous()

    def convert_to_fp8mma(
        self, weights_list: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert the weights to fp8mma format.

        Args:
            weights: List of weights.

        Returns:
            Tuple of weights.
        """
        return self.convert_to_mma(weights_list, "fp8mma")

    def convert_to_fp16mma(
        self, weights_list: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert the weights to fp16mma format.

        Args:
            weights: List of weights.

        Returns:
            Tuple of weights.
        """
        return self.convert_to_mma(weights_list, "fp16mma")


class ExpertSelectUpGateSiLU(TileRTModule):
    """ExpertSelectUpGateSiLU module"""

    def __init__(
        self,
        model_args: ModelArgs,
        num_devices: int,
        device_id: int = 0,
        ref_weights_alias: ExpertSelectUpGateSiLURefWeightsAlias | None = None,
        tilert_weights_alias: ExpertSelectUpGateSiLUTilertWeightsAlias | None = None,
        algorithm: ExpertSelectUpGateSiLUAlgorithm = ExpertSelectUpGateSiLUAlgorithm.FP8MMA,
    ):
        super().__init__(
            self.__class__.__name__,
            model_args=model_args,
            num_devices=num_devices,
            device_id=device_id,
        )

        self.arch_name = self.model_args.arch_name
        self.dim = self.model_args.dim

        self.n_activated_experts = self.model_args.n_activated_experts
        self.n_routed_experts = self.model_args.n_routed_experts
        self.n_shared_experts = self.model_args.n_shared_experts
        self.moe_inter_dim = self.model_args.moe_inter_dim
        self.n_expert_groups = self.model_args.n_expert_groups
        self.n_limited_groups = self.model_args.n_limited_groups
        self.route_scale = self.model_args.route_scale
        self.block_size = self.model_args.block_size
        self.algorithm = algorithm

        self.tilert_weights_alias = (
            tilert_weights_alias
            if tilert_weights_alias is not None
            else ExpertSelectUpGateSiLUTilertWeightsAlias()
        )
        self.ref_weights_alias = (
            ref_weights_alias
            if ref_weights_alias is not None
            else ExpertSelectUpGateSiLURefWeightsAlias(
                key_prefix="mlp", n_routed_experts=self.n_routed_experts
            )
        )

        # reference weights
        self.ref_bias: torch.Tensor | None = None
        self.ref_gate: torch.Tensor | None = None
        self.ref_up: torch.Tensor | None = None

        # tilert weights
        self.tilert_bias: torch.Tensor | None = None
        self.tilert_weights: torch.Tensor | None = None
        # for compatibility, to be removed in the future
        self.tilert_scales = torch.zeros(1, dtype=torch.bfloat16, device=torch.device("cuda"))

        # tilert vars
        self.hidden_out: torch.Tensor | None = None
        self.expert_probs: torch.Tensor | None = None
        self.expert_indices: torch.Tensor | None = None

        self.profile_logs: torch.Tensor | None = None
        self.is_init = False

        self._tensor_alias = self.tilert_weights_alias()
        self._tilert_tensor_alias = [
            self.tilert_weights_alias.exp_bias,
            "exp_upgate_weights",
            "exp_upgate_scales",
        ]

    @property
    def tensor_alias(self) -> list[str]:
        return self._tensor_alias

    @property
    def tilert_tensor_alias(self) -> list[str]:
        """Output weight names for get_weights_list (backward compat)."""
        return self._tilert_tensor_alias

    def get_weights_list(self) -> list[torch.Tensor]:
        """
        Get the weights list.

        Returns:
            List of weights.
        """
        return [self.tilert_bias, self.tilert_weights, self.tilert_scales]

    @staticmethod
    def process_gate_up_weights(
        key_prefix: str,  # e.g. mlp.shared_experts or mlp.experts.{id}
        weights_hf: dict[str, torch.Tensor],
        num_devices: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        gate_proj_weight_key = f"{key_prefix}.gate_proj.weight"
        gate_proj_scale_key = f"{key_prefix}.gate_proj.weight_scale_inv"
        up_proj_weight_key = f"{key_prefix}.up_proj.weight"
        up_proj_scale_key = f"{key_prefix}.up_proj.weight_scale_inv"

        gate_proj_weight = weights_hf[gate_proj_weight_key]
        gate_proj_scale = weights_hf[gate_proj_scale_key]
        up_proj_weight = weights_hf[up_proj_weight_key]
        up_proj_scale = weights_hf[up_proj_scale_key]
        dim = gate_proj_weight.shape[-1]
        in_dim = gate_proj_weight.shape[-2]
        scale_dim = gate_proj_scale.shape[-1]
        in_scale_dim = gate_proj_scale.shape[-2]
        in_dim_per_device = in_dim // num_devices
        in_scale_dim_per_device = in_scale_dim // num_devices
        gate_proj_weight = gate_proj_weight.reshape(num_devices, 1, in_dim_per_device, dim)
        gate_proj_scale = gate_proj_scale.reshape(
            num_devices, 1, in_scale_dim_per_device, scale_dim
        )
        up_proj_weight = up_proj_weight.reshape(num_devices, 1, in_dim_per_device, dim)
        up_proj_scale = up_proj_scale.reshape(num_devices, 1, in_scale_dim_per_device, scale_dim)
        return gate_proj_weight, gate_proj_scale, up_proj_weight, up_proj_scale

    def device_sharding(self, weights_map: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Device sharding: ref state dict -> tilert sharded tensors (num_devices, ...).

        Args:
            weights_map: State dict keyed by ref_weights_alias().

        Returns:
            Dict keyed by tilert_weights_alias() with (num_devices, ...) tensors.
        """
        ref_alias = self.ref_weights_alias
        key_prefix = ref_alias.key_prefix

        bias_key = f"{key_prefix}.gate.e_score_correction_bias"
        bias = weights_map[bias_key]
        bias = bias[None, :].repeat(self.num_devices, 1)

        gate_weights_list = []
        gate_scales_list = []
        up_weights_list = []
        up_scales_list = []
        assert self.n_shared_experts == 1, "Only one shared expert is supported"
        exp_prefix = f"{key_prefix}.shared_experts"
        gate_weights, gate_scales, up_weights, up_scales = self.process_gate_up_weights(
            exp_prefix, weights_map, self.num_devices
        )
        gate_weights_list.append(gate_weights)
        gate_scales_list.append(gate_scales)
        up_weights_list.append(up_weights)
        up_scales_list.append(up_scales)

        for exp_id in range(self.n_routed_experts):
            exp_prefix = f"{key_prefix}.experts.{exp_id}"
            gate_weights, gate_scales, up_weights, up_scales = self.process_gate_up_weights(
                exp_prefix, weights_map, self.num_devices
            )
            gate_weights_list.append(gate_weights)
            gate_scales_list.append(gate_scales)
            up_weights_list.append(up_weights)
            up_scales_list.append(up_scales)

        gate_weights = torch.cat(gate_weights_list, dim=1)
        gate_scales = torch.cat(gate_scales_list, dim=1)
        up_weights = torch.cat(up_weights_list, dim=1)
        up_scales = torch.cat(up_scales_list, dim=1)
        tilert_alias = self.tilert_weights_alias
        return {
            tilert_alias.exp_bias: bias,
            tilert_alias.exp_gate_weights: gate_weights,
            tilert_alias.exp_gate_scales: gate_scales,
            tilert_alias.exp_up_weights: up_weights,
            tilert_alias.exp_up_scales: up_scales,
        }

    def init_reference_weights(
        self,
        state_dict: dict[str, torch.Tensor],
        device_id: int | None = None,
    ) -> None:
        """
        Initialize the reference weights.

        Args:
            state_dict: State dict keyed by ref_weights_alias().
            device_id: Device ID; defaults to self.device_id.
        """
        did = self.device_id if device_id is None else device_id
        sharded = self.device_sharding(state_dict)

        tilert_alias = self.tilert_weights_alias
        bias = sharded[tilert_alias.exp_bias][did]
        gate_weights = sharded[tilert_alias.exp_gate_weights][did]
        gate_scales = sharded[tilert_alias.exp_gate_scales][did]
        up_weights = sharded[tilert_alias.exp_up_weights][did]
        up_scales = sharded[tilert_alias.exp_up_scales][did]

        self.ref_bias = bias
        ref_gate_list = [
            weight_dequant(gate_weights[i], gate_scales[i]) for i in range(gate_weights.shape[0])
        ]
        ref_up_list = [
            weight_dequant(up_weights[i], up_scales[i]) for i in range(up_weights.shape[0])
        ]
        self.ref_gate = torch.stack(ref_gate_list, dim=0)
        self.ref_up = torch.stack(ref_up_list, dim=0)

    def init_tilert_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        """
        Initialize the tilert weights.

        Args:
            state_dict: State dict keyed by tilert_weights_alias() (per-device).
        """
        assert self.algorithm is not None, "Algorithm is not set"
        weights_list = [state_dict[alias] for alias in self.tilert_weights_alias()]
        self.tilert_bias, self.tilert_weights = ExpertSelectUpGateSiLUWeightsConverter(
            self.model_args, self.num_devices
        ).dispatch(self.algorithm, weights_list)

    def init_tilert_vars(self, batch_size: int, seq_len: int, device: str = "cuda") -> None:
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
                self.n_activated_experts + self.n_shared_experts,
                self.moe_inter_dim // self.num_devices,
            ),
            dtype=torch.bfloat16,
            device=device,
        )
        self.expert_probs = torch.zeros(
            (batch_size, seq_len, self.n_activated_experts),
            dtype=torch.float32,
            device=device,
        )
        self.expert_indices = torch.zeros(
            (batch_size, seq_len, self.n_activated_experts),
            dtype=torch.int32,
            device=device,
        )

        self.profile_logs = get_profile_log_tensor(device=device)
        self.is_init = True

    def init_random_weights(self, device: str = "cuda") -> None:
        """
        Initialize the random weights.

        Returns:
            None
        """
        bias = torch.randn(self.n_routed_experts, dtype=torch.float32, device=device)
        gate_weights = [
            torch.randn(self.moe_inter_dim, self.dim, dtype=torch.bfloat16, device=device).to(
                torch.float8_e4m3fn
            )
            for _ in range(self.n_routed_experts + 1)
        ]
        up_weights = [
            torch.randn(self.moe_inter_dim, self.dim, dtype=torch.bfloat16, device=device).to(
                torch.float8_e4m3fn
            )
            for _ in range(self.n_routed_experts + 1)
        ]
        moe_inter_dim_scale_dim = self.moe_inter_dim // self.block_size
        dim_scale_dim = self.dim // self.block_size
        scale_dtype = torch.float32 if self.arch_name == "glm_5" else torch.bfloat16
        gate_scales = [
            torch.randn(moe_inter_dim_scale_dim, dim_scale_dim, dtype=scale_dtype, device=device)
            for _ in range(self.n_routed_experts + 1)
        ]
        up_scales = [
            torch.randn(moe_inter_dim_scale_dim, dim_scale_dim, dtype=scale_dtype, device=device)
            for _ in range(self.n_routed_experts + 1)
        ]
        tensor_list = [
            bias,
            *gate_weights,
            *up_weights,
            *gate_scales,
            *up_scales,
        ]
        ref_state_dict = dict(zip(self.ref_weights_alias(), tensor_list))
        self.init_reference_weights(ref_state_dict)
        sharded = self.device_sharding(ref_state_dict)
        per_device_state = {k: v[self.device_id] for k, v in sharded.items()}
        self.init_tilert_weights(per_device_state)

    def _ref_expert_select_ds(self, scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flatten_dim = np.prod(scores.size()[:-1])
        scores = scores.sigmoid()
        original_scores = scores
        if self.ref_bias is not None:
            scores = scores + self.ref_bias

        if self.n_expert_groups > 1:
            scores = scores.view(flatten_dim, self.n_expert_groups, -1)
            if self.ref_bias is None:
                group_scores = scores.amax(dim=-1)
            else:
                group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
            indices = group_scores.topk(self.n_limited_groups, dim=-1)[1]
            mask = scores.new_ones(flatten_dim, self.n_expert_groups, dtype=torch.bool).scatter_(
                1, indices, False
            )
            scores = scores.masked_fill_(mask.unsqueeze(-1), float("-inf")).flatten(1)
        indices = torch.topk(scores, self.n_activated_experts, dim=-1)[1]
        indices = indices.view(*original_scores.shape[:-1], self.n_activated_experts)
        weights = original_scores.gather(-1, indices)
        weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
        return weights, indices

    def _ref_expert_select_glm5(self, scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # flatten_dim = np.prod(scores.size()[:-1])
        scores = scores.sigmoid()
        original_scores = scores
        if self.ref_bias is not None:
            scores = scores + self.ref_bias
        indices = torch.topk(scores, self.n_activated_experts, dim=-1)[1]
        indices = indices.view(*original_scores.shape[:-1], self.n_activated_experts)
        weights = original_scores.gather(-1, indices)
        weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
        return weights, indices

    def golden_forward(
        self,
        x_in: torch.Tensor,
        scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.ref_gate is not None
        assert self.ref_up is not None
        bsz = x_in.shape[0]
        seq_len = x_in.shape[1]
        assert bsz == 1
        if self.arch_name == "deepseek_v3_2":
            weights, indices = self._ref_expert_select_ds(scores)
        elif self.arch_name == "glm_5":
            weights, indices = self._ref_expert_select_glm5(scores)
        else:
            raise ValueError(f"Unsupported architecture: {self.arch_name}")
        hidden_out_list = []
        for s in range(seq_len):
            # ref up-gate silu
            hidden_out_w1_list = []
            hidden_out_w3_list = []
            hidden_out_w1_shared = x_in[0, s].float() @ self.ref_gate[0].float().T
            hidden_out_w3_shared = x_in[0, s].float() @ self.ref_up[0].float().T
            hidden_out_w1_list.append(hidden_out_w1_shared)
            hidden_out_w3_list.append(hidden_out_w3_shared)
            ref_gate_sel = self.ref_gate[1:][indices[0, s]]
            ref_up_sel = self.ref_up[1:][indices[0, s]]
            for i in range(self.n_activated_experts):
                hidden_out_w1_sel = x_in[0, s].float() @ ref_gate_sel[i].float().T
                hidden_out_w3_sel = x_in[0, s].float() @ ref_up_sel[i].float().T
                hidden_out_w1_list.append(hidden_out_w1_sel)
                hidden_out_w3_list.append(hidden_out_w3_sel)
            hidden_out_w1 = torch.stack(hidden_out_w1_list, dim=0)
            hidden_out_w3 = torch.stack(hidden_out_w3_list, dim=0)
            hidden_out = F.silu(hidden_out_w1.float()) * hidden_out_w3.float()
            hidden_out = hidden_out.to(torch.bfloat16)
            hidden_out_list.append(hidden_out)
        hidden_out = torch.stack(hidden_out_list, dim=0)
        hidden_out = hidden_out[None, ...]
        return hidden_out, weights, indices

    def tilert_forward(
        self,
        x_in: torch.Tensor,
        scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.algorithm is not None, "Algorithm is not set"
        expert_select_up_gate_silu(
            x_in,
            scores,
            self.tilert_bias,
            self.tilert_weights,
            self.hidden_out,
            self.expert_probs,
            self.expert_indices,
            self.profile_logs,
            self.algorithm.value,
        )
        if self.flag_enable_profiling_log:
            parse_profile_log_tensor(
                self.profile_logs, self.get_profile_log_path(), [(self.op_name, 0.0)]
            )
        return self.hidden_out, self.expert_probs, self.expert_indices
