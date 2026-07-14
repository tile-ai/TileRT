"""ExpertDownAllreduce operation module."""

from dataclasses import dataclass
from enum import Enum

import torch

from tilert.models.base import TileRTModule, TilertWeightsConverter
from tilert.models.common import weight_dequant
from tilert.models.glm_5._dsa_v32.model_args import ModelArgs
from tilert.utils import get_profile_log_tensor

__all__ = [
    "expert_down_allreduce",
    "ExpertDownAllReduceAlgorithm",
    "ExpertDownAllReduce",
    "ExpertDownAllReduceTilertWeightsAlias",
]


def expert_down_allreduce(
    vec_in: torch.Tensor,
    mat_in: torch.Tensor,
    mat_scale: torch.Tensor,
    indices: torch.Tensor,
    scores: torch.Tensor,
    x_in: torch.Tensor,
    flag: int,
    vec_out: torch.Tensor,
    model_arch: str,
    compute_kernel_type: str = "bf16",
    profile_logs: torch.Tensor | None = None,
) -> None:
    """Fused expert down + allreduce (unified for DSv32 and GLM5)."""
    torch.ops.tilert.expert_down_allreduce_op(
        vec_in,
        mat_in,
        mat_scale,
        indices,
        scores,
        x_in,
        flag,
        vec_out,
        model_arch,
        compute_kernel_type,
        profile_logs,
    )


class ExpertDownAllReduceAlgorithm(Enum):
    """ExpertDownAllReduce algorithm."""

    GENERAL = "general"
    BF16MMA = "bf16mma"
    GLM5_FP4_HMMA = "glm5_fp4_hmma"


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

    @staticmethod
    def _swizzle_bf16mma_full_16x32(mat_in: torch.Tensor) -> torch.Tensor:
        assert mat_in.shape[-2] == 16 and mat_in.shape[-1] == 32
        assert mat_in.dtype == torch.float8_e4m3fn
        pre = mat_in.shape[:-2]
        mat = mat_in.reshape(*pre, 2, 8, 2, 2, 4, 2)
        n = len(pre)
        mat = mat.permute(*range(n), 1 + n, 4 + n, 2 + n, 3 + n, 0 + n, 5 + n)
        return mat.reshape(*pre, 32, 16).contiguous()

    @staticmethod
    def _swizzle_bf16mma_partial_8x32(mat_in: torch.Tensor) -> torch.Tensor:
        assert mat_in.shape[-2] == 8 and mat_in.shape[-1] == 32
        assert mat_in.dtype == torch.float8_e4m3fn
        pre = mat_in.shape[:-2]
        mat = mat_in.reshape(*pre, 8, 2, 2, 4, 2)
        n = len(pre)
        mat = mat.permute(*range(n), 0 + n, 3 + n, 1 + n, 2 + n, 4 + n)
        return mat.reshape(*pre, 32, 8).contiguous()

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
        expert_dim = args.moe_inter_dim // 8
        k_chunks = expert_dim // 32
        scale_cols = expert_dim // args.block_size

        with torch.inference_mode():
            mat_in, scale_in = weights_list
            exp_num = mat_in.shape[0]
            mat_in_s = mat_in.reshape(exp_num, num_sms, dim_per_sm, expert_dim)
            mat_in_0 = (
                mat_in_s[:, :, :16].reshape(exp_num, num_sms, 16, k_chunks, 32).transpose(2, 3)
            )
            mat_in_0 = self._swizzle_qmma_16x32(mat_in_0).reshape(exp_num, 128, -1)
            mat_in_1 = (
                mat_in_s[:, :, 16:32].reshape(exp_num, num_sms, 16, k_chunks, 32).transpose(2, 3)
            )
            mat_in_1 = self._swizzle_qmma_16x32(mat_in_1).reshape(exp_num, 128, -1)
            mat_in_2 = (
                mat_in_s[:, :, 32:48].reshape(exp_num, num_sms, 16, k_chunks, 32).transpose(2, 3)
            )
            mat_in_2 = self._swizzle_qmma_16x32(mat_in_2).reshape(exp_num, 128, -1)
            mats_to_cat = [mat_in_0, mat_in_1, mat_in_2]
            if arch_name == "deepseek_v3_2":
                mat_in_3 = (
                    mat_in_s[:, :, 48:56].reshape(exp_num, num_sms, 8, k_chunks, 32).transpose(2, 3)
                )
                mat_in_3 = self._swizzle_qmma_8x32(mat_in_3).reshape(exp_num, 128, -1)
                mats_to_cat.append(mat_in_3)
            mat_in_swizzled = torch.cat(mats_to_cat, dim=2)
            mat_in_swizzled = mat_in_swizzled.reshape(exp_num, dim, expert_dim)

            mat_scale_tilert = (
                scale_in.reshape(exp_num, dim_scale_dim, 1, scale_cols)
                .repeat(1, 1, 16, 1)
                .reshape(exp_num, num_sms, -1)
            )
            target_cols_per_sm = 1024 * scale_cols // num_sms
            pad_amount = target_cols_per_sm - mat_scale_tilert.shape[-1]
            if pad_amount > 0:
                padding_zeros = torch.zeros(
                    (exp_num, num_sms, pad_amount),
                    dtype=scale_in.dtype,
                    device=scale_in.device,
                )
                mat_scale_tilert = torch.cat([mat_scale_tilert, padding_zeros], dim=2)
            mat_scale_tilert = mat_scale_tilert.reshape(exp_num, 1024, scale_cols)
            if arch_name == "glm_5":
                if mat_scale_tilert.dtype != torch.float32:
                    print(
                        "Warning: ExpertDownAllReduceWeightsConverter: "
                        + f"mat_scale_tilert.dtype: {mat_scale_tilert.dtype} "
                        + "is not float32, convert to float32."
                    )
                mat_scale_tilert = mat_scale_tilert.to(torch.float32)
            else:
                mat_scale_tilert = mat_scale_tilert.to(torch.bfloat16)
            return mat_in_swizzled.contiguous(), mat_scale_tilert.contiguous()

    def convert_to_bf16mma(
        self, weights_list: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        args = self.model_args
        assert args.arch_name in (
            "deepseek_v3_2",
            "glm_5",
        ), "BF16 MMA wired for DSv32 / GLM5 only."
        dim = args.dim
        num_sms = 128
        dim_per_sm = dim // num_sms
        expert_dim = args.moe_inter_dim // 8
        k_chunks = expert_dim // 32
        scale_cols = expert_dim // args.block_size
        n_full_tiles = dim_per_sm // 16
        remainder_rows = dim_per_sm % 16
        full_rows = n_full_tiles * 16

        with torch.inference_mode():
            mat_in, scale_in = weights_list
            exp_num = mat_in.shape[0]
            mat_per_cta = mat_in.reshape(exp_num, num_sms, dim_per_sm, expert_dim)

            full_part = mat_per_cta[:, :, :full_rows, :]
            full_tiles = full_part.reshape(
                exp_num, num_sms, n_full_tiles, 16, k_chunks, 32
            ).transpose(3, 4)
            full_swizzled = self._swizzle_bf16mma_full_16x32(full_tiles)
            full_swizzled = full_swizzled.reshape(
                exp_num, num_sms, n_full_tiles * k_chunks * 32 * 16
            )

            mats = [full_swizzled]
            if remainder_rows > 0:
                partial_part = mat_per_cta[:, :, full_rows:, :]
                partial_tiles = partial_part.reshape(
                    exp_num, num_sms, 1, remainder_rows, k_chunks, 32
                ).transpose(3, 4)
                partial_swizzled = self._swizzle_bf16mma_partial_8x32(partial_tiles)
                partial_swizzled = partial_swizzled.reshape(
                    exp_num, num_sms, k_chunks * 32 * remainder_rows
                )
                mats.append(partial_swizzled)

            mat_swizzled = torch.cat(mats, dim=2) if len(mats) > 1 else mats[0]
            mat_swizzled = mat_swizzled.reshape(exp_num, dim, expert_dim)

            mat_scale_tilert = (
                scale_in.reshape(exp_num, dim // args.block_size, 1, scale_cols)
                .repeat(1, 1, 16, 1)
                .reshape(exp_num, num_sms, -1)
            )
            target_cols_per_sm = 1024 * scale_cols // num_sms
            pad_amount = target_cols_per_sm - mat_scale_tilert.shape[-1]
            if pad_amount > 0:
                padding_zeros = torch.zeros(
                    (exp_num, num_sms, pad_amount),
                    dtype=scale_in.dtype,
                    device=scale_in.device,
                )
                mat_scale_tilert = torch.cat([mat_scale_tilert, padding_zeros], dim=2)
            mat_scale_tilert = mat_scale_tilert.reshape(exp_num, 1024, scale_cols)
            if args.arch_name == "glm_5":
                mat_scale_tilert = mat_scale_tilert.to(torch.float32)
            else:
                mat_scale_tilert = mat_scale_tilert.to(torch.bfloat16)

            return mat_swizzled.contiguous(), mat_scale_tilert.contiguous()

    def convert_to_glm5_fp4_hmma(
        self, weights_list: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from tilert.models.common_mxfp4 import (
            _unpack_fp4_nibbles_last,
            build_down_weights_mma_natural,
        )

        assert (
            len(weights_list) == 2
        ), f"convert_to_glm5_fp4_hmma expects 2 tensors, got {len(weights_list)}"
        down_fp4, down_e8m0 = weights_list
        arch = self.model_args.arch_name
        assert arch == "glm_5", f"GLM5_FP4_HMMA down converter is GLM5-only, got arch={arch}"

        dim = self.model_args.dim
        moe_inter_pd = self.model_args.moe_inter_dim // self.num_devices

        with torch.inference_mode():
            if down_fp4.shape[-1] == moe_inter_pd:
                down_nib = down_fp4.to(torch.uint8).contiguous()
            else:
                assert down_fp4.shape[-1] == moe_inter_pd // 2, (
                    "routed fp4 down last dim must be inter_pd or inter_pd/2; "
                    f"got {down_fp4.shape[-1]} (inter_pd={moe_inter_pd})"
                )
                down_nib = _unpack_fp4_nibbles_last(down_fp4)
            down_e8 = down_e8m0.to(torch.uint8).contiguous()

            n_routed = down_nib.shape[0]
            assert down_nib.shape == (n_routed, dim, moe_inter_pd), (
                f"down_fp4 must be (n_routed, {dim}, {moe_inter_pd}); "
                f"got {tuple(down_nib.shape)}"
            )

            device = down_nib.device
            e_total = n_routed + 1
            u8 = {"dtype": torch.uint8, "device": device}
            full_nib = torch.zeros(e_total, dim, moe_inter_pd, **u8)
            full_e8 = torch.zeros(e_total, dim, moe_inter_pd // 32, **u8)
            full_nib[1:] = down_nib
            full_e8[1:] = down_e8
            down_packed = build_down_weights_mma_natural(full_nib, full_e8, dim, moe_inter_pd)
            dummy = torch.zeros(1, dtype=torch.float32, device=device)
            return down_packed, dummy


@dataclass
class ExpertDownAllReduceTilertWeightsAlias:

    exp_down_weights = "exp_down_weights"
    exp_down_scales = "exp_down_scales"

    @property
    def tilert_tensor_alias(self) -> list[str]:
        return [self.exp_down_weights, self.exp_down_scales]

    def glm5_fp4_tilert_tensor_alias(self) -> list[str]:
        return [self.exp_down_weights, self.exp_down_scales]

    def __call__(self) -> list[str]:
        return self.tilert_tensor_alias


class ExpertDownAllReduce(TileRTModule):
    """ExpertDownAllReduce module."""

    _SUPPORTED_ALGORITHMS = {
        "deepseek_v3_2": [
            ExpertDownAllReduceAlgorithm.GENERAL,
            ExpertDownAllReduceAlgorithm.BF16MMA,
        ],
        "glm_5": [
            ExpertDownAllReduceAlgorithm.GENERAL,
            ExpertDownAllReduceAlgorithm.BF16MMA,
            ExpertDownAllReduceAlgorithm.GLM5_FP4_HMMA,
        ],
    }

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

        if self.arch_name in ("deepseek_v3_2", "glm_5"):
            self.compute_kernel_type = "bf16"
            if algorithm == ExpertDownAllReduceAlgorithm.BF16MMA:
                self.compute_kernel_type = "bf16mma"
        else:
            raise ValueError(f"Unsupported architecture: {self.arch_name}")

        self.model_arch = self.arch_name

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

    def set_algorithm(self, algorithm: Enum) -> None:
        super().set_algorithm(algorithm)
        if algorithm == ExpertDownAllReduceAlgorithm.BF16MMA:
            self.compute_kernel_type = "bf16mma"
        elif algorithm == ExpertDownAllReduceAlgorithm.GENERAL:
            self.compute_kernel_type = "bf16"

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

    @staticmethod
    def _split_last_axis(t: torch.Tensor, num_devices: int) -> torch.Tensor:
        d, inter = t.shape[-2], t.shape[-1]
        assert (
            inter % num_devices == 0
        ), f"down last-axis {inter} not divisible by num_devices {num_devices}"
        return (
            t.reshape(d, num_devices, inter // num_devices)
            .transpose(0, 1)
            .reshape(num_devices, 1, d, inter // num_devices)
        )

    def process_down_weights_fp4(
        self,
        key_prefix: str,
        weights_hf: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_dev = self.num_devices
        down_w = weights_hf[f"{key_prefix}.down_proj.weight"]
        down_s = weights_hf[f"{key_prefix}.down_proj.weight_scale"]
        return self._split_last_axis(down_w, n_dev), self._split_last_axis(down_s, n_dev)

    def device_sharding_fp4(
        self,
        weights_dict: dict[str, torch.Tensor],
        key_prefix: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Shard routed-only down weight + scale across devices."""
        dw, ds = [], []
        for exp_id in range(self.n_routed_experts):
            down_weights, down_scales = self.process_down_weights_fp4(
                f"{key_prefix}.experts.{exp_id}", weights_dict
            )
            dw.append(down_weights)
            ds.append(down_scales)
        return torch.cat(dw, dim=1).contiguous(), torch.cat(ds, dim=1).contiguous()

    def device_sharding(
        self,
        weights_dict: dict[str, torch.Tensor],
        key_prefix: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.algorithm == ExpertDownAllReduceAlgorithm.GLM5_FP4_HMMA:
            return self.device_sharding_fp4(weights_dict, key_prefix)
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
        self.ref_down = torch.stack([t.to(torch.bfloat16) for t in down_list], dim=0)

    def get_tilert_weights_alias(self) -> list[str]:
        """Return the alias list keyed into ``state_dict`` for this op."""
        return list(self.tilert_weights_alias())

    def init_tilert_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        assert self.algorithm is not None, "Algorithm is not set"
        if self.algorithm == ExpertDownAllReduceAlgorithm.GLM5_FP4_HMMA:
            assert (
                self.arch_name == "glm_5"
            ), f"GLM5_FP4_HMMA is GLM5-only, got arch={self.arch_name}"
            converter = ExpertDownAllReduceWeightsConverter(self.model_args, self.num_devices)
            self.tilert_weights, self.tilert_scales = converter.convert_to_glm5_fp4_hmma(
                [state_dict[alias] for alias in self.tensor_alias]
            )
            self.is_tilert_weights_init = True
            return
        aliases = [state_dict[alias] for alias in self.tensor_alias]
        self.tilert_weights, self.tilert_scales = (
            torch.ops.tilert.expert_down_allreduce__convert_weights(
                aliases, self.model_arch, self.compute_kernel_type
            )
        )
        self.is_tilert_weights_init = True

    def init_tilert_vars(self, batch_size: int, seq_len: int, device_id: int = 0) -> None:
        self.hidden_out = torch.zeros(
            (batch_size, seq_len, self.dim),
            dtype=torch.bfloat16,
            device=f"cuda:{device_id}",
        )
        self.profile_logs = get_profile_log_tensor(device=f"cuda:{device_id}")
        self.is_init = True

    def init_random_weights(self, device_id: int | None = None) -> None:
        if device_id is None:
            device_id = self.device_id
        n = self.n_routed_experts + 1
        dev = f"cuda:{device_id}"
        down_weights = list(
            torch.randn(n, self.dim, self.moe_inter_dim, dtype=torch.bfloat16, device=dev)
            .to(torch.float8_e4m3fn)
            .unbind(0)
        )
        dim_scale_dim = self.dim // self.block_size
        moe_inter_dim_scale_dim = self.moe_inter_dim // self.block_size
        scale_dtype = torch.float32 if self.arch_name == "glm_5" else torch.bfloat16
        down_scales = list(
            torch.randn(
                n, dim_scale_dim, moe_inter_dim_scale_dim, dtype=scale_dtype, device=dev
            ).unbind(0)
        )
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
        assert self.hidden_out is not None
        expert_down_allreduce(
            vec_in,
            self.tilert_weights,
            self.tilert_scales,
            indices,
            scores,
            x_in,
            flag,
            self.hidden_out,
            self.model_arch,
            self.compute_kernel_type,
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
