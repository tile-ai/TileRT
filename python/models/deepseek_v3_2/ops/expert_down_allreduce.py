from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import torch

from tilert.models.base import TileRTModule, TilertWeightsConverter
from tilert.models.common import weight_dequant
from tilert.models.deepseek_v3_2.model_args import ModelArgs
from tilert.utils import get_profile_log_tensor

__all__ = [
    "expert_down_allreduce",
    "expert_down_allreduce_glm5",
    "ExpertDownAllReduceAlgorithm",
    "ExpertDownAllReduce",
    "ExpertDownAllReduceTilertWeightsAlias",
]


VALID_SEQ_LENS = (1, 2, 4)


def expert_down_allreduce(
    vec_in: torch.Tensor,
    mat_in: torch.Tensor,
    mat_scale: torch.Tensor,
    indices: torch.Tensor,
    scores: torch.Tensor,
    x_in: torch.Tensor,
    flag: int,
    vec_out: torch.Tensor,
    profile_logs: torch.Tensor,
) -> None:
    """
    Fused expert down + allreduce (deepseek_v3_2).

    Args:
        vec_in: [1, seq_len, n_experts, 256], bfloat16.
        mat_in: [n_experts, 6144, 256], float8_e4m3fn.
        mat_scale: [n_experts, 1024, 2], bfloat16.
        indices: [1, seq_len, 8], int32.
        scores: [1, seq_len, 8], float32.
        x_in: [1, seq_len, 6144], bfloat16.
        flag: User flag.
        vec_out: [1, seq_len, 6144], bfloat16 (output).
        profile_logs: 1D tensor (num_sms,) for profile logs.
    """
    torch.ops.tilert.expert_down_allreduce_op(
        vec_in, mat_in, mat_scale, indices, scores, x_in, flag, vec_out, profile_logs
    )


def expert_down_allreduce_glm5(
    vec_in: torch.Tensor,
    mat_in: torch.Tensor,
    mat_scale: torch.Tensor,
    indices: torch.Tensor,
    scores: torch.Tensor,
    x_in: torch.Tensor,
    flag: int,
    vec_out: torch.Tensor,
    profile_logs: torch.Tensor,
) -> None:
    """
    Fused expert down + allreduce (glm_5).

    Args:
        vec_in: [1, seq_len, n_experts, 256], bfloat16.
        mat_in: [n_experts, 6144, 256], float8_e4m3fn.
        mat_scale: [n_experts, 1024, 2], bfloat16.
        indices: [1, seq_len, 8], int32.
        scores: [1, seq_len, 8], float32.
        x_in: [1, seq_len, 6144], bfloat16.
        flag: User flag.
        vec_out: [1, seq_len, 6144], bfloat16 (output).
        profile_logs: 1D tensor (num_sms,) for profile logs.
    """
    torch.ops.tilert.expert_down_allreduce_glm5_op(
        vec_in, mat_in, mat_scale, indices, scores, x_in, flag, vec_out, profile_logs
    )


class ExpertDownAllReduceAlgorithm(Enum):
    """ExpertDownAllReduce algorithm."""

    GENERAL = "general"


class ExpertDownAllReduceWeightsConverter(TilertWeightsConverter):
    """ExpertDownAllReduce weights converter."""

    @staticmethod
    def _swizzle_qmma_16x32(mat_in: torch.Tensor) -> torch.Tensor:
        assert mat_in.shape[-2] == 16 and mat_in.shape[-1] == 32
        assert mat_in.dtype == torch.float8_e4m3fn
        pre_shape = mat_in.shape[:-2]
        mat_in = mat_in.reshape(*pre_shape, 2, 8, 2, 4, 4).transpose(-4, -3).transpose(-5, -4)
        return mat_in.reshape(*pre_shape, 2 * 2, 8 * 4, 4).transpose(-3, -2)

    @staticmethod
    def _swizzle_qmma_8x32(mat_in: torch.Tensor) -> torch.Tensor:
        assert mat_in.shape[-2] == 8 and mat_in.shape[-1] == 32
        pre_shape = mat_in.shape[:-2]
        return mat_in.reshape(*pre_shape, 8, 2, 4, 4).transpose(-2, -3).contiguous()

    def convert_to_general(
        self, weights_list: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert weights to general (tilert) format."""
        args = self.model_args
        assert args.arch_name in ("deepseek_v3_2", "glm_5")
        arch_name = args.arch_name
        dim = args.dim
        num_sms = 128
        dim_per_sm = dim // num_sms
        dim_scale_dim = dim // args.block_size

        with torch.inference_mode():
            mat_in, scale_in = weights_list
            exp_num = mat_in.shape[0]
            mat_in_s = mat_in.reshape(exp_num, num_sms, dim_per_sm, 256)
            mat_in_0 = mat_in_s[:, :, :16].reshape(exp_num, num_sms, 16, 8, 32).transpose(2, 3)
            mat_in_0 = self._swizzle_qmma_16x32(mat_in_0).reshape(exp_num, 128, -1)
            mat_in_1 = mat_in_s[:, :, 16:32].reshape(exp_num, num_sms, 16, 8, 32).transpose(2, 3)
            mat_in_1 = self._swizzle_qmma_16x32(mat_in_1).reshape(exp_num, 128, -1)
            mat_in_2 = mat_in_s[:, :, 32:48].reshape(exp_num, num_sms, 16, 8, 32).transpose(2, 3)
            mat_in_2 = self._swizzle_qmma_16x32(mat_in_2).reshape(exp_num, 128, -1)
            mat_in_swizzled = torch.cat([mat_in_0, mat_in_1, mat_in_2], dim=2)
            if arch_name == "deepseek_v3_2":
                mat_in_3 = mat_in_s[:, :, 48:56].reshape(exp_num, num_sms, 8, 8, 32).transpose(2, 3)
                mat_in_3 = self._swizzle_qmma_8x32(mat_in_3).reshape(exp_num, 128, -1)
                mat_in_swizzled = torch.cat([mat_in_0, mat_in_1, mat_in_2, mat_in_3], dim=2)
            mat_in_swizzled = mat_in_swizzled.reshape(exp_num, dim, 256)

            mat_scale_tilert = (
                scale_in.reshape(exp_num, dim_scale_dim, 1, 2)
                .repeat(1, 1, 16, 1)
                .reshape(exp_num, num_sms, -1)
            )
            padding_zeros = torch.zeros(
                (exp_num, num_sms, 16 - mat_scale_tilert.shape[-1]),
                dtype=scale_in.dtype,
                device=scale_in.device,
            )
            mat_scale_tilert = torch.cat([mat_scale_tilert, padding_zeros], dim=2)
            mat_scale_tilert = mat_scale_tilert.reshape(exp_num, 1024, 2)
            if arch_name == "glm_5":
                if mat_scale_tilert.dtype != torch.float32:
                    print(
                        "Warning: ExpertDownAllReduceWeightsConverter: "
                        + f"mat_scale_tilert.dtype: {mat_scale_tilert.dtype} "
                        + "is not float32, convert to float32."
                    )
                mat_scale_tilert = mat_scale_tilert.to(torch.float32)
            else:  # DS v3.2, use bfloat16 for mat_scale_tilert
                mat_scale_tilert = mat_scale_tilert.to(torch.bfloat16)
            return mat_in_swizzled.contiguous(), mat_scale_tilert.contiguous()


@dataclass
class ExpertDownAllReduceTilertWeightsAlias:
    """TileRT weights alias for ExpertDownAllReduce."""

    exp_down_weights = "exp_down_weights"
    exp_down_scales = "exp_down_scales"

    @property
    def tilert_tensor_alias(self) -> list[str]:
        return [self.exp_down_weights, self.exp_down_scales]

    def __call__(self) -> list[str]:
        return self.tilert_tensor_alias


class ExpertDownAllReduce(TileRTModule):
    """ExpertDownAllReduce module."""

    def __init__(
        self,
        model_args: ModelArgs,
        device_id: int,
        num_devices: int,
        algorithm: ExpertDownAllReduceAlgorithm = ExpertDownAllReduceAlgorithm.GENERAL,
    ):
        super().__init__(
            self.__class__.__name__,
            model_args=model_args,
            device_id=device_id,
            num_devices=num_devices,
        )
        self.arch_name = self.model_args.arch_name
        self.dim = self.model_args.dim
        self.n_activated_experts: int = self.model_args.n_activated_experts
        self.n_routed_experts: int = self.model_args.n_routed_experts
        self.n_shared_experts: int = self.model_args.n_shared_experts
        self.moe_inter_dim = self.model_args.moe_inter_dim
        self.block_size = self.model_args.block_size
        self.algorithm = algorithm

        self.ref_down: torch.Tensor | None = None
        self.tilert_weights: torch.Tensor | None = None
        self.tilert_scales: torch.Tensor | None = None
        self.hidden_out: torch.Tensor | None = None
        self.profile_logs: torch.Tensor | None = None
        self.is_init = False
        self.exp_down_allreduce_func: Callable | None = None

        if self.arch_name == "deepseek_v3_2":
            self.exp_down_allreduce_func = expert_down_allreduce
        elif self.arch_name == "glm_5":
            self.exp_down_allreduce_func = expert_down_allreduce_glm5
        else:
            raise ValueError(f"Unsupported architecture: {self.arch_name}")

        self.tilert_weights_alias = ExpertDownAllReduceTilertWeightsAlias()
        self.tensor_alias = ["exp_down_weights", "exp_down_scales"]
        self.ref_tensor_alias = (
            ["mlp.shared_experts.down_proj.weight"]
            + [f"mlp.experts.{i}.down_proj.weight" for i in range(self.n_routed_experts)]
            + ["mlp.shared_experts.down_proj.weight_scale_inv"]
            + [f"mlp.experts.{i}.down_proj.weight_scale_inv" for i in range(self.n_routed_experts)]
        )

    @property
    def tilert_tensor_alias(self) -> list[str]:
        return self.tilert_weights_alias.tilert_tensor_alias

    def get_weights_list(self) -> list[torch.Tensor]:
        return [self.tilert_weights, self.tilert_scales]

    @staticmethod
    def process_down_weights(
        key_prefix: str,
        weights_hf: dict[str, torch.Tensor],
        num_devices: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        down_proj_weight_key = f"{key_prefix}.down_proj.weight"
        down_proj_scale_key = f"{key_prefix}.down_proj.weight_scale_inv"
        down_proj_weight = weights_hf[down_proj_weight_key]
        down_proj_scale = weights_hf[down_proj_scale_key]

        dim = down_proj_weight.shape[-2]
        dim_scale_dim = down_proj_scale.shape[-2]
        moe_inter_dim = down_proj_weight.shape[-1]
        in_scale_dim = down_proj_scale.shape[-1]
        moe_inter_dim_per_device = moe_inter_dim // num_devices
        in_scale_dim_per_device = in_scale_dim // num_devices

        down_proj_weight = down_proj_weight.reshape(dim, num_devices, moe_inter_dim_per_device)
        down_proj_weight = down_proj_weight.transpose(0, 1).reshape(
            num_devices, 1, dim, moe_inter_dim_per_device
        )
        down_proj_scale = down_proj_scale.reshape(
            dim_scale_dim, num_devices, in_scale_dim_per_device
        )
        down_proj_scale = down_proj_scale.transpose(0, 1).reshape(
            num_devices, 1, dim_scale_dim, in_scale_dim_per_device
        )
        return down_proj_weight, down_proj_scale

    def device_sharding(
        self,
        weights_dict: dict[str, torch.Tensor],
        key_prefix: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.n_shared_experts == 1, "Only one shared expert is supported"
        down_weights_list = []
        down_scales_list = []
        exp_prefix = f"{key_prefix}.shared_experts"
        down_weights, down_scales = self.process_down_weights(
            exp_prefix, weights_dict, self.num_devices
        )
        down_weights_list.append(down_weights)
        down_scales_list.append(down_scales)
        for exp_id in range(self.n_routed_experts):
            exp_prefix = f"{key_prefix}.experts.{exp_id}"
            down_weights, down_scales = self.process_down_weights(
                exp_prefix, weights_dict, self.num_devices
            )
            down_weights_list.append(down_weights)
            down_scales_list.append(down_scales)
        down_weights = torch.cat(down_weights_list, dim=1)
        down_scales = torch.cat(down_scales_list, dim=1)
        return down_weights.contiguous(), down_scales.contiguous()

    def init_reference_weights(
        self,
        state_dict: dict[str, torch.Tensor],
        key_prefix: str,
        device_id: int = 0,
    ) -> None:
        sharded_list = self.device_sharding(state_dict, key_prefix)
        down_weights = sharded_list[0][device_id]
        down_scales = sharded_list[1][device_id]

        down_list = [
            weight_dequant(down_weight, down_scale)
            for down_weight, down_scale in zip(down_weights, down_scales)
        ]
        self.ref_down = torch.stack(down_list, dim=0)

    def init_tilert_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        assert self.algorithm is not None, "Algorithm is not set"
        self.tilert_weights, self.tilert_scales = ExpertDownAllReduceWeightsConverter(
            self.model_args, self.num_devices
        ).dispatch(self.algorithm, [state_dict[alias] for alias in self.tensor_alias])

    def init_tilert_vars(self, batch_size: int, seq_len: int, device_id: int = 0) -> None:
        self.hidden_out = torch.zeros(
            (batch_size, seq_len, self.dim),
            dtype=torch.bfloat16,
            device=f"cuda:{device_id}",
        )
        self.profile_logs = get_profile_log_tensor(device=f"cuda:{device_id}")
        self.is_init = True

    def init_random_weights(self, device_id: int = 0) -> None:
        down_weights = [
            torch.randn(
                self.dim, self.moe_inter_dim, dtype=torch.bfloat16, device=f"cuda:{device_id}"
            ).to(torch.float8_e4m3fn)
            for _ in range(self.n_routed_experts + 1)
        ]
        dim_scale_dim = self.dim // self.block_size
        moe_inter_dim_scale_dim = self.moe_inter_dim // self.block_size
        scale_dtype = torch.float32 if self.arch_name == "glm_5" else torch.bfloat16
        down_scales = [
            torch.randn(
                dim_scale_dim,
                moe_inter_dim_scale_dim,
                dtype=scale_dtype,
                device=f"cuda:{device_id}",
            )
            for _ in range(self.n_routed_experts + 1)
        ]
        state_dict = dict(
            zip(
                self.ref_tensor_alias,
                [*down_weights, *down_scales],
            )
        )
        self.init_reference_weights(state_dict, "mlp", device_id)
        sharded_list = self.device_sharding(state_dict, "mlp")
        sharded_state_dict = {
            alias: sharded_list[i][device_id] for i, alias in enumerate(self.tensor_alias)
        }
        self.init_tilert_weights(sharded_state_dict)

    def golden_forward(
        self,
        vec_in: torch.Tensor,
        indices: torch.Tensor,
        scores: torch.Tensor,
    ) -> torch.Tensor:
        assert self.ref_down is not None
        assert vec_in.dim() == 4 and vec_in.size(0) == 1
        seq_len = vec_in.shape[1]
        hidden_out_list = []
        for s in range(seq_len):
            hidden_out_w2_list = []
            hidden_out_w2_shared = vec_in[0, s, 0].float() @ self.ref_down[0].float().T
            hidden_out_w2_list.append(hidden_out_w2_shared)
            ref_down_sel = self.ref_down[1:][indices[0, s]]
            for i in range(self.n_activated_experts):
                hidden_out_w2_sel = vec_in[0, s, i + 1].float() @ ref_down_sel[i].float().T
                hidden_out_w2_list.append(hidden_out_w2_sel * scores[0, s, i])
            hidden_out_w2 = torch.stack(hidden_out_w2_list, dim=0).to(torch.bfloat16)
            hidden_out_w2 = torch.sum(hidden_out_w2, dim=0)
            hidden_out_list.append(hidden_out_w2)
        hidden_out = torch.stack(hidden_out_list, dim=0)
        return hidden_out[None, ...]

    def tilert_forward(
        self,
        vec_in: torch.Tensor,
        indices: torch.Tensor,
        scores: torch.Tensor,
        x_in: torch.Tensor,
        flag: int,
    ) -> torch.Tensor:
        assert self.exp_down_allreduce_func is not None
        assert self.hidden_out is not None
        self.exp_down_allreduce_func(
            vec_in,
            self.tilert_weights,
            self.tilert_scales,
            indices,
            scores,
            x_in,
            flag,
            self.hidden_out,
            self.profile_logs,
        )
        return self.hidden_out

    def __call__(
        self,
        x_in: torch.Tensor,
        indices: torch.Tensor,
        scores: torch.Tensor,
    ) -> torch.Tensor:
        return self.golden_forward(x_in, indices, scores)
