"""UnprojOAllreduce operation module."""

from dataclasses import dataclass
from enum import Enum

import torch

from tilert.models.base import TileRTModule, TilertWeightsConverter
from tilert.models.common import weight_dequant
from tilert.models.deepseek_v3_2.model_args import ModelArgs
from tilert.profiler.utils import parse_profile_log_tensor
from tilert.utils import get_profile_log_tensor

__all__ = [
    "unproj_o_allreduce",
    "UnProjOAllReduce",
    "UnProjOAllReduceRefWeightsAlias",
    "UnProjOAllReduceTilertWeightsAlias",
]


def unproj_o_allreduce(
    vec_in: torch.Tensor,
    mat_in: torch.Tensor,
    mat_scale: torch.Tensor,
    x_in: torch.Tensor,
    flag: int,
    vec_out: torch.Tensor,
    profile_logs: torch.Tensor,
    algorithm: str = "fp8mma",
) -> None:
    """
    Fused operation of unprojection and allreduce.

    Args:
        vec_in: Input tensor.
        mat_in: Input tensor.
        mat_scale: Input tensor.
        x_in: Input tensor.
        flag: Input flag.
        vec_out: Output tensor.
        profile_logs: Profile logs tensor. This is a 1D tensor of shape
                      (num_sms,) to store the profile logs of the unproj_o_allreduce
                      operation, where num_sms is the number of SMs on the
                      device.
    """
    if vec_out.shape[-1] == 7168:
        assert algorithm == "fp8mma", "Only fp8mma is supported for deepseek_v3_2"
        torch.ops.tilert.unproj_o_allreduce_op(
            vec_in, mat_in, mat_scale, x_in, flag, vec_out, profile_logs
        )

    elif vec_out.shape[-1] == 6144:
        torch.ops.tilert.unproj_o_allreduce_glm5_op(
            vec_in, mat_in, mat_scale, x_in, flag, vec_out, profile_logs, algorithm
        )
    else:
        raise ValueError(f"Unsupported vector dimension: {vec_out.shape[-1]}")


class UnProjOAllReduceAlgorithm(Enum):
    """UnprojOAllReduce algorithm"""

    FP8MMA = "fp8mma"
    FP16MMA = "fp16mma"


@dataclass
class UnProjOAllReduceRefWeightsAlias:
    """Reference weights alias for UnProjOAllReduce."""

    o_proj_weight = "self_attn.o_proj.weight"
    o_proj_scale_inv = "self_attn.o_proj.weight_scale_inv"

    @property
    def ref_tensor_alias(self) -> list[str]:
        return [self.o_proj_weight, self.o_proj_scale_inv]

    def __call__(self) -> list[str]:
        return self.ref_tensor_alias


@dataclass
class UnProjOAllReduceTilertWeightsAlias:
    """TileRT weights alias for UnProjOAllReduce."""

    unproj_weights = "unproj_weights"
    unproj_scales = "unproj_scales"

    @property
    def tilert_tensor_alias(self) -> list[str]:
        return [self.unproj_weights, self.unproj_scales]

    def __call__(self) -> list[str]:
        return self.tilert_tensor_alias


class UnProjOAllReduceWeightsConverter(TilertWeightsConverter):
    """UnProjOAllReduce weights converter"""

    @staticmethod
    def _swizzle_qmma_16x32(mat_in: torch.Tensor) -> torch.Tensor:
        assert mat_in.shape[-2] == 16 and mat_in.shape[-1] == 32
        assert mat_in.dtype == torch.float8_e4m3fn
        # PTX isa fig.88
        pre_shape = mat_in.shape[:-2]
        mat_in = mat_in.reshape(*pre_shape, 2, 8, 2, 4, 4).transpose(-4, -3).transpose(-5, -4)
        return mat_in.reshape(*pre_shape, 2 * 2, 8 * 4, 4).transpose(-3, -2)

    def convert_to_fp8mma(
        self, weights_list: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert the weights to fp8mma format.

        Args:
            weights_list: List of weights.

        Returns:
            Tuple of weights.
        """
        args = self.model_args
        assert args.arch_name == "deepseek_v3_2" or args.arch_name == "glm_5"
        arch_name = args.arch_name
        dim = args.dim
        num_sms = 128
        if arch_name == "deepseek_v3_2":
            num_sms = 112
        dim_per_sm = dim // num_sms
        dim_scale_dim = dim // args.block_size

        with torch.inference_mode():
            mat_in, scales_trt = weights_list
            vec_dim = mat_in.shape[-1]  # 2048 for both deepseek_v3_2 and glm_5
            assert scales_trt.shape == (dim // args.block_size, vec_dim // args.block_size)

            weights_trt = mat_in.reshape(num_sms, dim_per_sm, vec_dim)
            # dim_per_stage is 512
            stages = vec_dim // 512
            weights_trt = weights_trt.reshape(num_sms, dim_per_sm, stages, 512).transpose(1, 2)

            weights_trt = weights_trt.reshape(
                num_sms, stages, dim_per_sm // 16, 16, 16, 32
            ).transpose(-2, -3)
            weights_trt = self._swizzle_qmma_16x32(weights_trt)
            weights_trt = weights_trt.reshape(num_sms, stages, -1)

            if arch_name == "glm_5":
                if scales_trt.dtype != torch.float32:
                    print(
                        "Warning: UnProjOAllReduceWeightsConverter: "
                        + f"scales_trt.dtype: {scales_trt.dtype} "
                        + "is not float32, convert to float32."
                    )
                scales_trt = scales_trt.to(torch.float32)
                # repeat 8 times
                scales_trt = (
                    scales_trt.reshape((dim_scale_dim, 1, -1)).repeat(1, 8, 1).reshape(num_sms, -1)
                )
            else:  # DS v3.2, use bfloat16 for scales
                scales_trt = scales_trt.to(torch.bfloat16)

            return weights_trt.contiguous(), scales_trt.contiguous()

    @staticmethod
    def _swizzle_mma_16x16(mat_in: torch.Tensor) -> torch.Tensor:
        assert mat_in.shape[-2] == 16 and mat_in.shape[-1] == 16
        # PTX isa fig.88
        pre_shape = mat_in.shape[:-2]
        mat_in = mat_in.reshape(*pre_shape, 2, 8, 2, 4, 2).transpose(-4, -3).transpose(-5, -4)
        return mat_in.reshape(*pre_shape, 2 * 2, 8 * 4, 2).transpose(-3, -2)

    def convert_to_fp16mma(
        self,
        weights_list: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert common weights to TileRT FP16 MMA layout."""
        assert self.model_args.arch_name == "glm_5", "Only GLM-5 supports FP16 MMA"

        with torch.inference_mode():
            mat, scales = weights_list
            if scales.dtype != torch.float32:
                print(
                    "Warning: UnProjOAllReduceWeightsConverter: "
                    + f"scales.dtype: {scales.dtype} "
                    + "is not float32, convert to float32."
                )
                scales = scales.to(torch.float32)

            sms = 128  # use 128 sms for glm_5
            pages = 4
            scales = scales.reshape(6144 // 128, 1, 2048 // 128)
            scales = scales.repeat(1, 8, 1)
            scales = scales.reshape(128, 3, 4, 4).transpose(1, 2)
            # to 128, 4, 12x4
            scales = scales.reshape(128, 4, 12).view(torch.float8_e4m3fn)

            mat = (
                mat.reshape(128, 48, 2048)
                .reshape(128, 3, 16, 4, 512)
                .transpose(2, 3)
                .reshape(128, 3, 4, 16, 32, 16)
                .transpose(3, 4)
                .reshape(128, 3, 4, 32, 16, 16)
            )
            mat = UnProjOAllReduceWeightsConverter._swizzle_mma_16x16(mat)
            mat = mat.transpose(1, 2).reshape(128, 4, -1)

            scales_padding = torch.zeros(
                sms,
                pages,
                128 - scales.shape[-1],
                dtype=torch.float8_e4m3fn,
                device=mat.device,
            )  # append 128-byte aligned scale: (128, 4, 24704) for glm_5
            mat_full = torch.cat([mat, scales, scales_padding], dim=-1).contiguous()
            dummy_scales = torch.zeros(1, dtype=torch.float32, device=mat.device)
            return mat_full, dummy_scales


class UnProjOAllReduce(TileRTModule):
    """UnProjOAllReduce module"""

    def __init__(
        self,
        model_args: ModelArgs,
        num_devices: int,
        device_id: int = 0,
        ref_weights_alias: UnProjOAllReduceRefWeightsAlias | None = None,
        tilert_weights_alias: UnProjOAllReduceTilertWeightsAlias | None = None,
        algorithm: UnProjOAllReduceAlgorithm = UnProjOAllReduceAlgorithm.FP8MMA,
    ):
        super().__init__(
            self.__class__.__name__,
            model_args=model_args,
            num_devices=num_devices,
            device_id=device_id,
        )

        self.tilert_weights_alias = (
            tilert_weights_alias
            if tilert_weights_alias is not None
            else UnProjOAllReduceTilertWeightsAlias()
        )
        self.ref_weights_alias = (
            ref_weights_alias
            if ref_weights_alias is not None
            else UnProjOAllReduceRefWeightsAlias()
        )

        self.arch_name = self.model_args.arch_name
        self.dim = self.model_args.dim
        self.n_heads = self.model_args.n_heads
        self.head_dim = self.model_args.v_head_dim

        self.block_size = self.model_args.block_size
        self.algorithm: UnProjOAllReduceAlgorithm = algorithm

        # reference weights
        self.ref_unproj_o: torch.Tensor | None = None

        # tilert weights
        self.tilert_weights: torch.Tensor | None = None
        self.tilert_scales: torch.Tensor | None = None

        # tilert vars
        self.hidden_out: torch.Tensor | None = None

        self.profile_logs: torch.Tensor | None = None
        self.is_var_init = False

    def get_weights_list(self) -> list[torch.Tensor]:
        """
        Get the weights list.

        Returns:
            List of weights.
        """
        return [self.tilert_weights, self.tilert_scales]

    def device_sharding(self, weights_map: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Device sharding.

        Args:
            weights_map: Map from ref weight alias to tensor (full model).

        Returns:
            Map from tilert weight alias to (num_devices, ...) tensors.
        """
        unproj_o_weight = weights_map[self.ref_weights_alias.o_proj_weight]
        unproj_o_scale = weights_map[self.ref_weights_alias.o_proj_scale_inv]
        unproj_o_weight = unproj_o_weight.reshape(self.dim, self.num_devices, -1)
        unproj_o_weight = unproj_o_weight.transpose(0, 1)
        unproj_o_scale = unproj_o_scale.reshape(self.dim // self.block_size, self.num_devices, -1)
        unproj_o_scale = unproj_o_scale.transpose(0, 1)
        return {
            self.tilert_weights_alias.unproj_weights: unproj_o_weight.contiguous(),
            self.tilert_weights_alias.unproj_scales: unproj_o_scale.contiguous(),
        }

    def init_reference_weights(
        self,
        state_dict: dict[str, torch.Tensor],
        device_id: int | None = None,
    ) -> None:
        """
        Initialize the reference weights.

        Args:
            state_dict: State dictionary keyed by ref weight alias (full model).
            device_id: Device ID for this shard; defaults to self.device_id.
        """
        did = self.device_id if device_id is None else device_id
        sharded = self.device_sharding(state_dict)
        weights = sharded[self.tilert_weights_alias.unproj_weights][did]
        scales = sharded[self.tilert_weights_alias.unproj_scales][did]
        self.ref_unproj_o = weight_dequant(weights, scales)

    def init_tilert_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        """
        Initialize the tilert weights.

        Args:
            state_dict: State dictionary keyed by tilert weight alias (per-device).
        """
        assert self.algorithm is not None, "Algorithm is not set"
        self.tilert_weights, self.tilert_scales = UnProjOAllReduceWeightsConverter(
            self.model_args, self.num_devices
        ).dispatch(
            self.algorithm,
            [state_dict[alias] for alias in self.tilert_weights_alias()],
        )

    def init_tilert_vars(self, batch_size: int, seq_len: int) -> None:
        """
        Initialize the tilert variables.

        Args:
            batch_size: Batch size.
            seq_len: Sequence length.
        """
        self.hidden_out = torch.zeros(
            (batch_size, seq_len, self.dim),
            dtype=torch.bfloat16,
            device=f"cuda:{self.device_id}",
        )
        self.profile_logs = get_profile_log_tensor(device=f"cuda:{self.device_id}")
        self.is_var_init = True

    def init_random_weights(self) -> None:
        """Initialize the random weights."""
        unproj_o_weights = torch.randn(
            self.dim,
            self.n_heads * self.head_dim,
            dtype=torch.bfloat16,
            device=f"cuda:{self.device_id}",
        ).to(torch.float8_e4m3fn)

        head_scale_dim = self.head_dim // self.block_size
        dim_scale_dim = self.dim // self.block_size
        scale_dtype = torch.float32 if self.arch_name == "glm_5" else torch.bfloat16
        unproj_o_scales = torch.randn(
            dim_scale_dim,
            self.n_heads * head_scale_dim,
            dtype=scale_dtype,
            device=f"cuda:{self.device_id}",
        )
        ref_state_dict = {
            self.ref_weights_alias.o_proj_weight: unproj_o_weights,
            self.ref_weights_alias.o_proj_scale_inv: unproj_o_scales,
        }

        self.init_reference_weights(ref_state_dict)
        sharded = self.device_sharding(ref_state_dict)
        per_device_state = {k: v[self.device_id] for k, v in sharded.items()}
        self.init_tilert_weights(per_device_state)

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
        assert self.ref_unproj_o is not None
        bsz = vec_in.shape[0]
        seq_len = vec_in.shape[1]
        assert bsz == 1
        res = vec_in.reshape(bsz, seq_len, -1).float() @ self.ref_unproj_o.T.float()
        return res.to(torch.bfloat16)

    def tilert_forward(
        self,
        vec_in: torch.Tensor,
        x_in: torch.Tensor,
        flag: int,
    ) -> torch.Tensor:
        assert self.hidden_out is not None
        assert self.profile_logs is not None
        assert self.algorithm is not None
        unproj_o_allreduce(
            vec_in,
            self.tilert_weights,
            self.tilert_scales,
            x_in,
            flag,
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
        vec_in: torch.Tensor,
    ) -> torch.Tensor:
        return self.golden_forward(vec_in)
