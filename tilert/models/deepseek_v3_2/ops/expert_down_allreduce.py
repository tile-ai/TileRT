"""ExpertDownAllreduce operation module.

Mirrors the C++ registration pattern in register_expert_down_allreduce.cu:
- Op wrappers call torch.ops and optionally validate args (SetExpertDownAllreduceInst).
- Weights converter prepares weights for the kernel.
- Module wires init, device_sharding, golden_forward, and tilert_forward.
"""

from dataclasses import dataclass
from enum import Enum

import torch

from tilert.models.base import TileRTModule, TilertWeightsConverter
from tilert.models.common import weight_dequant
from tilert.models.deepseek_v3_2.model_args import ModelArgs
from tilert.utils import get_profile_log_tensor

__all__ = [
    "expert_down_allreduce",
    "ExpertDownAllReduceAlgorithm",
    "ExpertDownAllReduce",
    "ExpertDownAllReduceTilertWeightsAlias",
]

# -----------------------------------------------------------------------------
# Constants (aligned with run_expert_down_allreduce_op in C++)
# -----------------------------------------------------------------------------

VALID_SEQ_LENS = (1, 2, 4)


# -----------------------------------------------------------------------------
# Op wrapper (unified for both DSv32 and GLM5)
# -----------------------------------------------------------------------------


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
    model_arch: str,
    compute_kernel_type: str = "bf16",
) -> None:
    """
    Fused expert down + allreduce (unified for DSv32 and GLM5).

    Args:
        vec_in: [1, seq_len, n_experts, 256], bfloat16.
        mat_in: [n_experts, dim, 256], float8_e4m3fn.
        mat_scale: [n_experts, 1024, 2], bfloat16 (DSv32) or float32 (GLM5).
        indices: [1, seq_len, 8], int32.
        scores: [1, seq_len, 8], float32.
        x_in: [1, seq_len, dim], bfloat16.
        flag: User flag.
        vec_out: [1, seq_len, dim], bfloat16 (output).
        profile_logs: 1D tensor (num_sms,) for profile logs.
        compute_kernel_type: "bf16" for both DSv32 and GLM5.
    """
    torch.ops.tilert.expert_down_allreduce_op(
        vec_in,
        mat_in,
        mat_scale,
        indices,
        scores,
        x_in,
        flag,
        vec_out,
        profile_logs,
        model_arch,
        compute_kernel_type,
    )


# -----------------------------------------------------------------------------
# Algorithm & Weights converter
# -----------------------------------------------------------------------------


class ExpertDownAllReduceAlgorithm(Enum):
    """ExpertDownAllReduce algorithm."""

    GENERAL = "general"
    # DSv32-only BF16 m16n8k16 MMA path (CUDA 13.2+).  Same FP8 weights +
    # FP32-or-BF16 group scales as GENERAL, but the per-CTA matrix uses a
    # lane-major BF16-MMA-friendly swizzle (3 full M=16 tiles + 1 partial
    # M=8 tile, each warp-K-chunk packed as 32 lanes × 16/8 bytes).  In the
    # kernel: hidden BF16 stays BF16 (no FP8 quantize), weights cvt FP8->BF16
    # via cvt.rn.bf16x2.e4m3x2 (single SASS F2FP.BF16.E4M3.UNPACK_B), MMA is
    # mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32.  Selected by
    # compute_kernel_type="bf16mma" on the C++ side; module sets that
    # automatically when this algorithm is active.
    BF16MMA = "bf16mma"


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
        """Lane-major BF16 m16n8k16 swizzle of a (16, 32) FP8 sub-block.

        K=32 covers two BF16 m16n8k16 sub-tiles (K=0..15, K=16..31).  Output
        is laid out lane-major so each lane's full uint4 (16 bytes) carries
        BOTH sub-tiles back-to-back in one LDS.128 — saving 1 LDS per warp's
        K=32 chunk vs the older MMA-major layout (which forced two separate
        uint2 LDS).

        Per-lane (g=lid/4, c=lid%4) byte order:
          byte  0..1 : M=g,   K=c*2..c*2+1     (mma#1 a[0])
          byte  2..3 : M=g+8, K=c*2..c*2+1     (mma#1 a[1])
          byte  4..5 : M=g,   K=c*2+8..c*2+9   (mma#1 a[2])
          byte  6..7 : M=g+8, K=c*2+8..c*2+9   (mma#1 a[3])
          byte  8..9 : M=g,   K=c*2+16..c*2+17 (mma#2 a[0])
          byte 10..11: M=g+8, K=c*2+16..c*2+17 (mma#2 a[1])
          byte 12..13: M=g,   K=c*2+24..c*2+25 (mma#2 a[2])
          byte 14..15: M=g+8, K=c*2+24..c*2+25 (mma#2 a[3])

        Output: (..., 32 lanes, 16 bytes/lane) FP8 — 512 bytes total.
        """
        assert mat_in.shape[-2] == 16 and mat_in.shape[-1] == 32
        assert mat_in.dtype == torch.float8_e4m3fn
        pre = mat_in.shape[:-2]
        # M=16 = (M_lh=2, g=8): M = M_lh*8 + g.
        # K=32 = (mma=2, lh=2, c=4, kp=2): K = mma*16 + lh*8 + c*2 + kp.
        mat = mat_in.reshape(*pre, 2, 8, 2, 2, 4, 2)  # (M_lh, g, mma, lh, c, kp)
        # Lane-major output: (g, c, mma, lh, M_lh, kp).
        # Per lane (=g, c): bytes are mma-major within the lane, so MMA#1 8B
        # then MMA#2 8B in registers after a single uint4 LDS.
        n = len(pre)
        mat = mat.permute(*range(n), 1 + n, 4 + n, 2 + n, 3 + n, 0 + n, 5 + n)
        return mat.reshape(*pre, 32, 16).contiguous()

    @staticmethod
    def _swizzle_bf16mma_partial_8x32(mat_in: torch.Tensor) -> torch.Tensor:
        """Lane-major BF16 m16n8k16 swizzle of a (8, 32) FP8 partial sub-block.

        Output mirrors `_swizzle_bf16mma_full_16x32` but with no M=g+8 row
        (M=8 only).  Each lane holds 8 bytes total = 4 bytes mma#1 + 4 bytes
        mma#2 contiguously, so one LDS.64 covers both sub-tiles per K=32.

        Per-lane (g=lid/4, c=lid%4) byte order:
          byte 0..1: M=g, K=c*2..c*2+1     (mma#1 a[0]; a[1] is don't-care)
          byte 2..3: M=g, K=c*2+8..c*2+9   (mma#1 a[2]; a[3] don't-care)
          byte 4..5: M=g, K=c*2+16..c*2+17 (mma#2 a[0]; a[1] don't-care)
          byte 6..7: M=g, K=c*2+24..c*2+25 (mma#2 a[2]; a[3] don't-care)

        Output: (..., 32 lanes, 8 bytes/lane) FP8 — 256 bytes total.
        """
        assert mat_in.shape[-2] == 8 and mat_in.shape[-1] == 32
        assert mat_in.dtype == torch.float8_e4m3fn
        pre = mat_in.shape[:-2]
        # M=8 (g=0..7), K=32 = (mma=2, lh=2, c=4, kp=2).
        mat = mat_in.reshape(*pre, 8, 2, 2, 4, 2)  # (g, mma, lh, c, kp)
        # Lane-major output: (g, c, mma, lh, kp).
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
            else:  # DS v3.2, use bfloat16 for mat_scale_tilert
                mat_scale_tilert = mat_scale_tilert.to(torch.bfloat16)
            return mat_in_swizzled.contiguous(), mat_scale_tilert.contiguous()

    def convert_to_bf16mma(
        self, weights_list: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pack FP8 weights for the BF16 m16n8k16 MMA kernel (DSv32 only).

        Same input as ``convert_to_general`` (FP8 weights + FP32 group scales)
        and same output shapes/dtypes — only the per-CTA M-tile swizzle of the
        weight bytes differs. Scales are unchanged (broadcast-and-pad to
        (n_experts, 1024, scale_cols) BF16).
        """
        args = self.model_args
        assert args.arch_name == "deepseek_v3_2", "BF16 MMA only wired for DSv32."
        dim = args.dim  # 7168
        num_sms = 128
        dim_per_sm = dim // num_sms  # 56
        expert_dim = args.moe_inter_dim // 8  # 256
        k_chunks = expert_dim // 32  # 8
        scale_cols = expert_dim // args.block_size  # 2
        # Layout assumes 3 full M=16 tiles + 1 partial M=8 tile (DSv32 dim_per_sm=56).
        assert dim_per_sm == 56, "BF16 MMA layout currently assumes dim_per_sm=56 (DSv32)."

        with torch.inference_mode():
            mat_in, scale_in = weights_list
            exp_num = mat_in.shape[0]
            mat_per_cta = mat_in.reshape(exp_num, num_sms, dim_per_sm, expert_dim)

            full_part = mat_per_cta[:, :, :48, :]  # (exp, sm, 48, 256)
            partial_part = mat_per_cta[:, :, 48:, :]  # (exp, sm, 8, 256)

            # 3 full M=16 tiles. Lane-major layout: per-warp K=32 chunk holds
            # 32 lanes * 16 bytes contiguously (one uint4 per lane).
            full_tiles = full_part.reshape(exp_num, num_sms, 3, 16, k_chunks, 32)
            full_tiles = full_tiles.transpose(3, 4)  # (exp, sm, 3, k_chunks, 16, 32)
            full_swizzled = self._swizzle_bf16mma_full_16x32(
                full_tiles
            )  # (exp, sm, 3, k_chunks, 32, 16)
            full_swizzled = full_swizzled.reshape(exp_num, num_sms, 3 * k_chunks * 32 * 16)

            # 1 partial M=8 tile. Lane-major: 32 lanes * 8 bytes per K=32 chunk.
            partial_tiles = partial_part.reshape(exp_num, num_sms, 1, 8, k_chunks, 32).transpose(
                3, 4
            )  # (exp, sm, 1, k_chunks, 8, 32)
            partial_swizzled = self._swizzle_bf16mma_partial_8x32(
                partial_tiles
            )  # (exp, sm, 1, k_chunks, 32, 8)
            partial_swizzled = partial_swizzled.reshape(exp_num, num_sms, k_chunks * 32 * 8)

            mat_swizzled = torch.cat([full_swizzled, partial_swizzled], dim=2)
            mat_swizzled = mat_swizzled.reshape(exp_num, dim, expert_dim)

            # Scales: identical to convert_to_general.
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
            mat_scale_tilert = mat_scale_tilert.to(torch.bfloat16)

            return mat_swizzled.contiguous(), mat_scale_tilert.contiguous()


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

    _SUPPORTED_ALGORITHMS = {
        "deepseek_v3_2": [
            ExpertDownAllReduceAlgorithm.GENERAL,
            ExpertDownAllReduceAlgorithm.BF16MMA,
        ],
        "glm_5": [ExpertDownAllReduceAlgorithm.GENERAL],
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
            # BF16MMA algorithm switches the C++ dispatch to kBf16MMA (FP8
            # weights cvt-to-BF16 in-kernel + BF16 m16n8k16 MMA, no input
            # FP8 quantize). The weight converter (convert_to_bf16mma) emits
            # a different per-CTA M-tile swizzle than convert_to_general.
            if algorithm == ExpertDownAllReduceAlgorithm.BF16MMA:
                self.compute_kernel_type = "bf16mma"
        else:
            raise ValueError(f"Unsupported architecture: {self.arch_name}")

        # OpContext dispatch parameters
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
        """Set algorithm and sync compute_kernel_type for BF16MMA dispatch."""
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
        # bf16 cast: matches ref_gate/ref_up in ExpertSelectUpGateSiLU — see
        # comment there for rationale (halves random-weights smoke memory).
        # Cast per-tensor before stacking to avoid a full-size fp32 intermediate.
        self.ref_down = torch.stack([t.to(torch.bfloat16) for t in down_list], dim=0)

    def get_tilert_weights_alias(self) -> list[str]:
        """Return the alias list keyed into ``state_dict`` for this op."""
        return list(self.tilert_weights_alias())

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
            self.profile_logs,
            self.model_arch,
            self.compute_kernel_type,
        )
        return self.hidden_out

    def __call__(
        self,
        x_in: torch.Tensor,
        indices: torch.Tensor,
        scores: torch.Tensor,
    ) -> torch.Tensor:
        return self.golden_forward(x_in, indices, scores)
