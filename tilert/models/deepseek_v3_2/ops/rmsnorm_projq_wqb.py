"""RmsnormProjqWqb operation module."""

import math
from dataclasses import dataclass
from enum import Enum

import torch

from tilert.models.base import TileRTModule, TilertWeightsConverter
from tilert.models.common import weight_dequant
from tilert.models.deepseek_v3_2.model_args import ModelArgs
from tilert.utils import get_profile_log_tensor

__all__ = [
    "RmsnormProjqWqb",
    "RmsnormProjqWqbAlgorithm",
    "RmsnormProjqWqbWeightsConverter",
]


def rmsnorm_projq_wqb_op(
    q: torch.Tensor,
    wq_b: torch.Tensor,
    wq_b_scales: torch.Tensor,
    q_norm_weight: torch.Tensor,
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    profile_logs: torch.Tensor,
    algorithm: str,
    model_arch: str,
) -> None:
    torch.ops.tilert.rmsnorm_proj_qb_op(
        q,
        wq_b,
        wq_b_scales,
        q_norm_weight,
        q_nope,
        q_pe,
        model_arch,
        algorithm,
        profile_logs,
        torch.empty(0, dtype=torch.int64, device=q.device),
    )


class RmsnormProjqWqbAlgorithm(Enum):
    """RmsnormProjqWqb algorithm."""

    FP16MMA = "fp16mma"
    BF16MMA = "bf16mma"


class RmsnormProjqWqbWeightsConverter(TilertWeightsConverter):
    """Weights converter for RmsnormProjqWqb.

    Supports configurations where n_heads is not evenly divisible by
    num_devices; in that case n_local_heads is padded and padded head
    weight rows are zero-filled.
    """

    kBf16NumCtas = 80
    kGemvPageSize = 8

    def __init__(self, model_args: ModelArgs, num_devices: int):
        super().__init__(model_args=model_args, num_devices=num_devices)

        self.proc_groups = 8
        self.repeat = 16

        self.block_size = self.model_args.block_size

        self.qk_nope_head_dim = self.model_args.qk_nope_head_dim
        self.qk_rope_head_dim = self.model_args.qk_rope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim

        self.needs_padding = self.model_args.n_heads % num_devices != 0
        self.n_local_heads = self._compute_n_local_heads(
            self.model_args.n_heads, num_devices, self.qk_head_dim
        )

        self.q_lora_dim = self.model_args.q_lora_rank
        self.q_lora_qdim = self.q_lora_dim // self.block_size

        self.qk_dim = self.qk_head_dim * self.n_local_heads
        self.qk_qdim = self.qk_dim // self.block_size

        assert self.qk_dim % (self.kBf16NumCtas * self.kGemvPageSize) == 0, (
            f"qk_dim ({self.qk_dim}) must be divisible by "
            f"kBf16NumCtas * kGemvPageSize ({self.kBf16NumCtas * self.kGemvPageSize})"
        )
        assert self.qk_dim % self.block_size == 0, (
            f"qk_dim ({self.qk_dim}) must be divisible by block_size ({self.block_size}) "
            f"for scale alignment"
        )

    @classmethod
    def _compute_n_local_heads(cls, n_total_heads: int, num_devices: int, qk_head_dim: int) -> int:
        """Compute padded n_local_heads per device."""
        if n_total_heads % num_devices == 0:
            return n_total_heads // num_devices

        base = math.ceil(n_total_heads / num_devices)
        align_unit = cls.kBf16NumCtas * cls.kGemvPageSize
        g = math.gcd(qk_head_dim, align_unit)
        head_align = align_unit // g
        return math.ceil(base / head_align) * head_align

    @staticmethod
    def _redistribute_heads(
        wq_b_full: torch.Tensor,
        wq_b_scale_full: torch.Tensor,
        n_total_heads: int,
        n_local_heads: int,
        num_devices: int,
        qk_head_dim: int,
        block_size: int,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Redistribute heads across devices with padding.

        Args:
            wq_b_full: [n_total_heads * qk_head_dim, q_lora_dim] full weight.
            wq_b_scale_full: [n_total_heads * qk_head_dim // block_size, q_lora_qdim] full scale.
            n_total_heads: Total number of heads (e.g. 128).
            n_local_heads: Target heads per GPU (padded, e.g. 20).
            num_devices: Number of devices (e.g. 7).
            qk_head_dim: Head dimension (e.g. 192).
            block_size: Quantization block size (e.g. 128).

        Returns:
            Lists of per-device (wq_b, wq_b_scale) with shape
            [n_local_heads * qk_head_dim, q_lora_dim] and
            [n_local_heads * qk_head_dim // block_size, q_lora_qdim].
        """
        total_rows = n_total_heads * qk_head_dim
        rows_per_dev = n_local_heads * qk_head_dim
        scale_rows_per_dev = rows_per_dev // block_size
        total_scale_rows = total_rows // block_size

        q_lora_dim = wq_b_full.shape[-1]
        q_lora_qdim = wq_b_scale_full.shape[-1]

        assert rows_per_dev % block_size == 0, (
            f"n_local_heads * qk_head_dim ({rows_per_dev}) must be "
            f"divisible by block_size ({block_size})"
        )

        wq_b_list = []
        scale_list = []
        for dev in range(num_devices):
            start_row = dev * rows_per_dev
            end_row = min(total_rows, start_row + rows_per_dev)
            real_rows = max(0, end_row - start_row)

            dev_wqb = torch.zeros(
                rows_per_dev, q_lora_dim, dtype=wq_b_full.dtype, device=wq_b_full.device
            )
            if real_rows > 0:
                dev_wqb[:real_rows] = wq_b_full[start_row:end_row]

            start_scale = dev * scale_rows_per_dev
            end_scale = min(total_scale_rows, start_scale + scale_rows_per_dev)
            real_scale_rows = max(0, end_scale - start_scale)

            dev_scale = torch.zeros(
                scale_rows_per_dev,
                q_lora_qdim,
                dtype=wq_b_scale_full.dtype,
                device=wq_b_scale_full.device,
            )
            if real_scale_rows > 0:
                dev_scale[:real_scale_rows] = wq_b_scale_full[start_scale:end_scale]

            wq_b_list.append(dev_wqb)
            scale_list.append(dev_scale)

        return wq_b_list, scale_list

    @staticmethod
    def _swizzle_mma_16x16(mat_in: torch.Tensor) -> torch.Tensor:
        assert mat_in.shape[-2] == 16 and mat_in.shape[-1] == 16
        pre_shape = mat_in.shape[:-2]
        mat_in = mat_in.reshape(*pre_shape, 2, 8, 2, 4, 2).transpose(-4, -3).transpose(-5, -4)
        return mat_in.reshape(*pre_shape, 2 * 2, 8 * 4, 2).transpose(-3, -2)

    @staticmethod
    def _swizzle_mma_16x16_for_pages(
        mat_in: torch.Tensor, q_lora_dim: int, pages: int
    ) -> torch.Tensor:
        """Swizzle 16xK matrix for paged MMA layout, any K divisible by 16."""
        assert mat_in.shape[-2] == 16 and mat_in.shape[-1] == q_lora_dim
        k_per_page = q_lora_dim // pages
        n_k_tiles = k_per_page // 16
        pre_shape = mat_in.shape[:-2]
        mat_in = mat_in.reshape(*pre_shape, 16, pages, k_per_page).transpose(-3, -2)
        mat_in = mat_in.reshape(*pre_shape, pages, 16, n_k_tiles, 16).transpose(-3, -2)
        mat_in = RmsnormProjqWqbWeightsConverter._swizzle_mma_16x16(mat_in)
        return mat_in.contiguous()

    def _common_to_tilert_fp16mma(
        self,
        wq_b: torch.Tensor,
        wq_b_scales_raw: torch.Tensor,
        rmsnorm_gamma: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert common weights to the FP16 MMA layout."""
        pages = 2
        rows_per_cta = 32

        qk_nope_dim = self.n_local_heads * self.qk_nope_head_dim
        qk_pe_dim = self.n_local_heads * self.qk_rope_head_dim
        nope_ctas = qk_nope_dim // rows_per_cta
        pe_ctas = qk_pe_dim // rows_per_cta
        num_ctas = nope_ctas + pe_ctas

        wq_b_scales_f32 = wq_b_scales_raw.to(torch.float32)
        wq_b_scales_f32 = (
            wq_b_scales_f32.reshape(self.qk_qdim, 1, self.q_lora_qdim)
            .repeat(1, self.block_size, 1)
            .reshape(self.qk_dim, self.q_lora_qdim)
        )

        wq_b_scales_f32 = wq_b_scales_f32.reshape(
            self.n_local_heads, self.qk_head_dim, self.q_lora_qdim
        )
        scale_nope = wq_b_scales_f32[:, : self.qk_nope_head_dim, :].reshape(-1, self.q_lora_qdim)
        scale_pe = wq_b_scales_f32[:, self.qk_nope_head_dim :, :].reshape(-1, self.q_lora_qdim)

        scale_nope = scale_nope.reshape(
            nope_ctas, rows_per_cta, pages, self.q_lora_qdim // pages
        ).transpose(1, 2)[:, :, 0, :]
        scale_pe = scale_pe.reshape(
            pe_ctas, rows_per_cta, pages, self.q_lora_qdim // pages
        ).transpose(1, 2)[:, :, 0, :]

        scales = torch.cat([scale_nope, scale_pe], dim=0)
        scales_fp8 = scales.contiguous().view(torch.float8_e4m3fn)

        wq_b = wq_b.reshape(self.n_local_heads, self.qk_head_dim, self.q_lora_dim)
        wq_b_nope = wq_b[:, : self.qk_nope_head_dim, :].reshape(-1, self.q_lora_dim)
        wq_b_pe = wq_b[:, self.qk_nope_head_dim :, :].reshape(-1, self.q_lora_dim)

        wq_b_nope = wq_b_nope.reshape(nope_ctas, rows_per_cta // 16, 16, self.q_lora_dim)
        wq_b_nope = RmsnormProjqWqbWeightsConverter._swizzle_mma_16x16_for_pages(
            wq_b_nope, self.q_lora_dim, pages
        )
        wq_b_nope = (
            wq_b_nope.reshape(nope_ctas, rows_per_cta // 16, pages, 16, -1)
            .transpose(1, 2)
            .reshape(nope_ctas, pages, rows_per_cta, -1)
        )

        wq_b_pe = wq_b_pe.reshape(pe_ctas, rows_per_cta // 16, 16, self.q_lora_dim)
        wq_b_pe = RmsnormProjqWqbWeightsConverter._swizzle_mma_16x16_for_pages(
            wq_b_pe, self.q_lora_dim, pages
        )
        wq_b_pe = (
            wq_b_pe.reshape(pe_ctas, rows_per_cta // 16, pages, 16, -1)
            .transpose(1, 2)
            .reshape(pe_ctas, pages, rows_per_cta, -1)
        )

        weights = torch.cat([wq_b_nope, wq_b_pe], dim=0)
        weights = weights.reshape(num_ctas, pages, -1)

        scale_padding_size = 128 - scales_fp8.shape[-1]
        scale_padding = torch.zeros(
            num_ctas,
            pages,
            scale_padding_size,
            dtype=torch.float8_e4m3fn,
            device=wq_b.device,
        )
        tilert_wqb = torch.cat([weights, scales_fp8, scale_padding], dim=-1).contiguous()

        tilert_wqb_scales = torch.zeros(1, dtype=torch.bfloat16)
        tilert_gamma = rmsnorm_gamma.float().detach().clone()
        return tilert_wqb, tilert_wqb_scales, tilert_gamma

    def convert_to_bf16mma(
        self, weights: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert common-format weights to the BF16 MMA layout."""
        return self.convert_to_fp16mma(weights)

    def convert_to_fp16mma(
        self, weights: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert common-format weights to TileRT FP16 MMA layout."""
        with torch.inference_mode():
            wq_b, wq_b_scale, q_norm_weight = weights
            return self._common_to_tilert_fp16mma(wq_b, wq_b_scale, q_norm_weight)


@dataclass
class RmsnormProjqWqbRefWeightsAlias:
    """Reference weights alias for RmsnormProjqWqb."""

    rmsnorm_gamma = "self_attn.q_a_layernorm.weight"
    wqb_weights = "self_attn.q_b_proj.weight"
    wqb_scales = "self_attn.q_b_proj.weight_scale_inv"

    @property
    def ref_tensor_alias(self) -> list[str]:
        return [
            self.rmsnorm_gamma,
            self.wqb_weights,
            self.wqb_scales,
        ]

    def __call__(self) -> list[str]:
        return self.ref_tensor_alias


@dataclass
class RmsnormProjqWqbTilertWeightsAlias:
    """TileRT weights alias for RmsnormProjqWqb."""

    rmsnorm_gamma = "q_rmsnorm_gamma"
    wqb_weights = "wqb_weights"
    wqb_scales = "wqb_scales"

    @property
    def tilert_tensor_alias(self) -> list[str]:
        return [
            self.rmsnorm_gamma,
            self.wqb_weights,
            self.wqb_scales,
        ]

    def __call__(self) -> list[str]:
        return self.tilert_tensor_alias


class RmsnormProjqWqb(TileRTModule):
    """RmsnormProjqWqb module: RMSNorm + Q projection (wq_b only)."""

    _SUPPORTED_ALGORITHMS = {
        "deepseek_v3_2": [
            RmsnormProjqWqbAlgorithm.FP16MMA,
            RmsnormProjqWqbAlgorithm.BF16MMA,
        ],
        "glm_5": [RmsnormProjqWqbAlgorithm.FP16MMA],
    }

    def __init__(
        self,
        model_args: ModelArgs,
        device_id: int,
        num_devices: int = 7,
        ref_weights_alias: RmsnormProjqWqbRefWeightsAlias | None = None,
    ):
        super().__init__(
            self.__class__.__name__,
            model_args=model_args,
            device_id=device_id,
            num_devices=num_devices,
        )

        self.tilert_weights_alias = RmsnormProjqWqbTilertWeightsAlias()
        self.ref_weights_alias = (
            ref_weights_alias if ref_weights_alias is not None else RmsnormProjqWqbRefWeightsAlias()
        )

        self.n_local_heads = RmsnormProjqWqbWeightsConverter._compute_n_local_heads(
            model_args.n_heads,
            num_devices,
            model_args.qk_nope_head_dim + model_args.qk_rope_head_dim,
        )
        self.q_lora_rank = model_args.q_lora_rank
        self.n_heads = model_args.n_heads
        self.qk_nope_head_dim = model_args.qk_nope_head_dim
        self.qk_rope_head_dim = model_args.qk_rope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.qk_local_dim = self.qk_head_dim * self.n_local_heads

        self.block_size = model_args.block_size
        self.q_lora_qdim = self.q_lora_rank // self.block_size
        self.qk_local_qdim = self.qk_local_dim // self.block_size
        self.eps = model_args.eps

        self.ref_q_norm: torch.Tensor | None = None
        self.ref_wq_b: torch.Tensor | None = None

        self.tilert_wq_b: torch.Tensor | None = None
        self.tilert_wq_b_scales: torch.Tensor | None = None
        self.tilert_q_norm_weight: torch.Tensor | None = None

        self.q_nope: torch.Tensor | None = None
        self.q_pe: torch.Tensor | None = None

        self.profile_logs: torch.Tensor | None = None

    def get_weights_list(self) -> list[torch.Tensor]:
        return [self.tilert_q_norm_weight, self.tilert_wq_b, self.tilert_wq_b_scales]

    def device_sharding(self, weights_map: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Redistribute heads across devices with padding."""
        gamma = weights_map[self.ref_weights_alias.rmsnorm_gamma][None, ...].repeat(
            self.num_devices, 1
        )

        wq_b_full = weights_map[self.ref_weights_alias.wqb_weights]
        wq_b_scale_full = weights_map[self.ref_weights_alias.wqb_scales]

        wq_b_list, scale_list = RmsnormProjqWqbWeightsConverter._redistribute_heads(
            wq_b_full,
            wq_b_scale_full,
            n_total_heads=self.n_heads,
            n_local_heads=self.n_local_heads,
            num_devices=self.num_devices,
            qk_head_dim=self.qk_head_dim,
            block_size=self.block_size,
        )

        sharded_wqb_weights = torch.stack(wq_b_list, dim=0)
        sharded_wqb_scales = torch.stack(scale_list, dim=0)

        return {
            self.tilert_weights_alias.rmsnorm_gamma: gamma,
            self.tilert_weights_alias.wqb_weights: sharded_wqb_weights,
            self.tilert_weights_alias.wqb_scales: sharded_wqb_scales,
        }

    def init_reference_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Initialize reference weights from common-format state dict."""
        self.ref_q_norm = state_dict[self.ref_weights_alias.rmsnorm_gamma]

        wq_b_full = state_dict[self.ref_weights_alias.wqb_weights]
        wq_b_scale_full = state_dict[self.ref_weights_alias.wqb_scales]

        wq_b_bf16_full = weight_dequant(wq_b_full, wq_b_scale_full)

        total_rows = self.n_heads * self.qk_head_dim
        rows_per_dev = self.n_local_heads * self.qk_head_dim
        start_row = self.device_id * rows_per_dev
        end_row = min(total_rows, start_row + rows_per_dev)
        real_rows = max(0, end_row - start_row)

        dev_wqb = torch.zeros(
            rows_per_dev,
            wq_b_bf16_full.shape[-1],
            dtype=wq_b_bf16_full.dtype,
            device=wq_b_bf16_full.device,
        )
        if real_rows > 0:
            dev_wqb[:real_rows] = wq_b_bf16_full[start_row:end_row]

        self.ref_wq_b = dev_wqb.contiguous()

    def init_tilert_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Initialize TileRT weights from common-format state dict."""
        weights = [
            state_dict[self.tilert_weights_alias.wqb_weights],
            state_dict[self.tilert_weights_alias.wqb_scales],
            state_dict[self.tilert_weights_alias.rmsnorm_gamma],
        ]
        assert self.algorithm is not None, "Algorithm is not set"
        self.tilert_wq_b, self.tilert_wq_b_scales, self.tilert_q_norm_weight = (
            RmsnormProjqWqbWeightsConverter(self.model_args, self.num_devices).dispatch(
                self.algorithm, weights
            )
        )

    def init_random_weights(self) -> None:
        """Initialize random reference and TileRT weights for testing."""
        q_norm = torch.randn(self.q_lora_rank, dtype=torch.float32)

        wq_b = torch.randn(self.qk_local_dim, self.q_lora_rank, dtype=torch.bfloat16).to(
            torch.float8_e4m3fn
        )
        scale_dtype = torch.float32 if self.model_args.arch_name == "glm_5" else torch.bfloat16
        wq_b_scale = torch.randn(self.qk_local_qdim, self.q_lora_qdim, dtype=scale_dtype)

        self.ref_q_norm = q_norm
        self.ref_wq_b = weight_dequant(wq_b, wq_b_scale).contiguous()

        assert self.algorithm is not None, "Algorithm is not set"
        weights = [wq_b, wq_b_scale, q_norm]
        self.tilert_wq_b, self.tilert_wq_b_scales, self.tilert_q_norm_weight = (
            RmsnormProjqWqbWeightsConverter(self.model_args, self.num_devices).dispatch(
                self.algorithm, weights
            )
        )

    def init_tilert_vars(self, batch_size: int, seq_len: int) -> None:
        """Allocate TileRT output buffers."""
        self.q_nope = torch.zeros(
            batch_size, seq_len, self.n_local_heads, self.qk_nope_head_dim, dtype=torch.bfloat16
        )
        self.q_pe = torch.zeros(
            batch_size, seq_len, self.n_local_heads, self.qk_rope_head_dim, dtype=torch.bfloat16
        )
        self.profile_logs = get_profile_log_tensor()
        self.is_var_init = True

    def golden_forward(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Reference forward: RMSNorm + linear projection (no iq)."""
        assert self.ref_q_norm is not None
        assert self.ref_wq_b is not None

        bsz, seqlen, _ = q.shape
        if bsz != 1 or seqlen not in [1, 2, 4]:
            raise ValueError(f"Invalid batch size or sequence length: bsz={bsz}, seqlen={seqlen}")

        qr = torch.nn.functional.rms_norm(q.float(), [q.size(-1)], self.ref_q_norm, self.eps).to(
            q.dtype
        )

        q_out = torch.matmul(qr, self.ref_wq_b.T)
        q_out = q_out.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q_out, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        return q_nope, q_pe

    def tilert_forward(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.tilert_wq_b is not None
        assert self.tilert_wq_b_scales is not None
        assert self.tilert_q_norm_weight is not None
        assert self.q_nope is not None
        assert self.q_pe is not None
        assert self.profile_logs is not None

        bsz, seqlen, _ = q.shape
        if bsz != 1 or seqlen not in [1, 2, 4]:
            raise ValueError(f"Invalid batch size or sequence length: bsz={bsz}, seqlen={seqlen}")

        assert self.algorithm is not None, "Algorithm is not set"

        rmsnorm_projq_wqb_op(
            q,
            self.tilert_wq_b,
            self.tilert_wq_b_scales,
            self.tilert_q_norm_weight,
            self.q_nope,
            self.q_pe,
            self.profile_logs,
            self.algorithm.value,
            model_arch=self.model_args.arch_name,
        )

        return self.q_nope, self.q_pe
