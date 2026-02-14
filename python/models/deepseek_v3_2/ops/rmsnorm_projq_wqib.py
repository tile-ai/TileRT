"""RmsnormProjqWqib operation module."""

from dataclasses import dataclass
from enum import Enum

import torch
from einops import rearrange

from tilert.models.base import TileRTModule, TilertWeightsConverter
from tilert.models.common import weight_dequant
from tilert.models.deepseek_v3_2.model_args import ModelArgs
from tilert.models.deepseek_v3_2.ops.expert_sel_up_gate_silu import (
    ExpertSelectUpGateSiLUWeightsConverter as WeightsConverter,
)
from tilert.profiler.utils import parse_profile_log_tensor
from tilert.utils import get_profile_log_tensor

__all__ = [
    "RmsnormProjqWqib",
    "RmsnormProjqWqibAlgorithm",
    "RmsnormProjqWqibWeightsConverter",
]


def rmsnorm_projq_wqib_op(
    q: torch.Tensor,
    wq_b_full: torch.Tensor,
    wq_b_full_scales: torch.Tensor,
    q_norm_weight: torch.Tensor,
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    iq: torch.Tensor,
    profile_logs: torch.Tensor,
    algorithm: str,
) -> None:
    dim = q.shape[-1]
    if dim == 1536:
        impl_func = torch.ops.tilert.rmsnorm_proj_qb_iq_op
    elif dim == 2048:
        impl_func = torch.ops.tilert.rmsnorm_proj_qb_iq_glm5_op
    else:
        raise ValueError(f"Invalid dimension: {dim}")
    impl_func(
        q,
        wq_b_full,
        wq_b_full_scales,
        q_norm_weight,
        q_nope,
        q_pe,
        iq,
        profile_logs,
        algorithm,
    )


class RmsnormProjqWqibAlgorithm(Enum):
    """RmsnormProjqWqib algorithm."""

    BF16 = "bf16"
    FP8 = "fp8"
    FP16MMA = "fp16mma"


class RmsnormProjqWqibWeightsConverter(TilertWeightsConverter):
    """Weights converter: common format to TileRT format."""

    def __init__(self, model_args: ModelArgs, num_devices: int):
        super().__init__(model_args=model_args, num_devices=num_devices)

        self.proc_groups = 8
        self.repeat = 16

        self.block_size = self.model_args.block_size
        self.n_local_heads = self.model_args.n_heads // self.num_devices

        self.q_lora_dim = self.model_args.q_lora_rank
        self.q_lora_qdim = self.q_lora_dim // self.block_size

        self.qk_nope_head_dim = self.model_args.qk_nope_head_dim
        self.qk_rope_head_dim = self.model_args.qk_rope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.qk_dim = self.qk_head_dim * self.n_local_heads
        self.qk_qdim = self.qk_dim // self.block_size

        self.index_n_heads = self.model_args.index_n_heads
        self.index_head_dim = self.index_n_heads * self.model_args.index_head_dim
        self.index_head_qdim = self.index_head_dim // self.block_size

    def _common_to_tilert_bf16(
        self,
        wq_b: torch.Tensor,
        wq_b_scales_raw: torch.Tensor,
        wq_b_iq: torch.Tensor,
        wq_b_iq_scales: torch.Tensor,
        rmsnorm_gamma: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert common weights to TileRT BF16 layout."""
        wq_b = wq_b.reshape(self.n_local_heads, self.qk_head_dim, self.q_lora_dim)
        wq_b_nope = wq_b[:, : self.qk_nope_head_dim, :]
        wq_b_nope = wq_b_nope.reshape(
            self.n_local_heads,
            self.proc_groups,
            self.qk_nope_head_dim // self.proc_groups,
            self.q_lora_dim,
        )
        wq_b_pe = wq_b[:, self.qk_nope_head_dim :, :]
        wq_b_pe = wq_b_pe.reshape(
            self.n_local_heads,
            self.proc_groups,
            self.qk_rope_head_dim // self.proc_groups,
            self.q_lora_dim,
        )
        wq_b = torch.cat([wq_b_nope, wq_b_pe], dim=2)
        wq_b = wq_b.reshape(self.qk_dim, self.q_lora_dim)
        wq_b_full = torch.cat([wq_b, wq_b_iq], dim=0)

        wq_b_scales_iq_raw = wq_b_iq_scales
        wq_b_scales_t16 = (
            wq_b_scales_raw.reshape((self.qk_qdim, 1, self.q_lora_qdim))
            .repeat(1, self.repeat, 1)
            .reshape(self.qk_qdim * self.repeat, self.q_lora_qdim)
        )
        wq_b_scales_t16 = wq_b_scales_t16.reshape(
            self.n_local_heads, self.qk_head_dim // self.proc_groups, self.q_lora_qdim
        )
        wq_b_scales_t16_nope = wq_b_scales_t16[:, : self.qk_nope_head_dim // 8]
        wq_b_scales_t16_pe = wq_b_scales_t16[:, self.qk_nope_head_dim // 8 :]
        wq_b_scales_t16_nope = wq_b_scales_t16_nope.reshape(
            self.n_local_heads,
            self.proc_groups,
            self.qk_nope_head_dim // 8 // self.proc_groups,
            self.q_lora_qdim,
        )
        wq_b_scales_t16_pe = wq_b_scales_t16_pe.reshape(
            self.n_local_heads,
            self.proc_groups,
            self.qk_rope_head_dim // 8 // self.proc_groups,
            self.q_lora_qdim,
        )
        wq_b_scales_t16 = torch.cat([wq_b_scales_t16_nope, wq_b_scales_t16_pe], dim=2)
        wq_b_scales_t16 = wq_b_scales_t16.reshape(-1, self.q_lora_qdim)
        wq_b_scales_full = torch.cat([wq_b_scales_t16, wq_b_scales_iq_raw], dim=0)

        return (
            wq_b_full.detach().clone(),
            wq_b_scales_full.detach().clone(),
            rmsnorm_gamma.float().detach().clone(),
        )

    def _common_to_tilert_fp8(
        self,
        wq_b: torch.Tensor,
        wq_b_scales_raw: torch.Tensor,
        wq_b_iq: torch.Tensor,
        wq_b_iq_scales_raw: torch.Tensor,
        rmsnorm_gamma: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert common weights to TileRT FP8 MMA layout."""
        # Reshape wq_b: simple split of nope and pe, then concatenate
        wq_b = wq_b.reshape(self.n_local_heads, self.qk_head_dim, self.q_lora_dim)
        wq_b_nope = wq_b[:, : self.qk_nope_head_dim, :].reshape(-1, self.q_lora_dim)
        wq_b_pe = wq_b[:, self.qk_nope_head_dim :, :].reshape(-1, self.q_lora_dim)
        wq_b = torch.cat([wq_b_nope, wq_b_pe], dim=0)

        # Process scales: expand and split nope/pe similarly to weights
        m_scale_group = self.block_size // self.repeat
        wq_b_scales_t16 = (
            wq_b_scales_raw.reshape((self.qk_qdim, 1, self.q_lora_qdim))
            .repeat(1, self.repeat, 1)
            .reshape(-1, self.qk_head_dim // m_scale_group, self.q_lora_qdim)
        )

        # Split nope and pe parts
        wq_b_scales_nope = wq_b_scales_t16[:, : self.qk_nope_head_dim // m_scale_group, :].reshape(
            [-1, self.q_lora_qdim]
        )
        wq_b_scales_pe = wq_b_scales_t16[:, self.qk_nope_head_dim // m_scale_group :, :].reshape(
            [-1, self.q_lora_qdim]
        )
        wq_b_scales_t16 = torch.cat([wq_b_scales_nope, wq_b_scales_pe], dim=0)

        # Process wq_b_iq scales
        wq_b_iq_scales_t16 = (
            wq_b_iq_scales_raw.reshape([self.index_head_qdim, 1, self.q_lora_qdim])
            .repeat([1, self.repeat, 1])
            .reshape((-1, self.q_lora_qdim))
        )

        # Concatenate weights and scales
        wq_b_raw = torch.cat([wq_b, wq_b_iq], dim=0)
        page_k = self.q_lora_qdim
        total_out_dim = self.qk_dim + self.index_head_dim
        total_out_qdim = total_out_dim // self.block_size
        wq_b_scales_full = (
            torch.cat(
                [wq_b_scales_t16.to(torch.float32), wq_b_iq_scales_t16.to(torch.float32)], dim=0
            )
            .reshape((total_out_qdim, self.repeat, page_k, self.q_lora_qdim // page_k))
            .permute([0, 2, 1, 3])
            .contiguous()
            .view(torch.float8_e4m3fn)
        )

        wq_b_raw = wq_b_raw.reshape(
            [total_out_qdim, 128 // 16, 16, page_k, self.q_lora_dim // 32 // page_k, 32]
        ).permute([0, 3, 1, 4, 2, 5])
        wq_b_raw = WeightsConverter._swizzle_mma_16x32(wq_b_raw)

        tilert_wq_b_full = torch.cat(
            [
                wq_b_raw.reshape((total_out_qdim, page_k, -1)),
                wq_b_scales_full.reshape([total_out_qdim, page_k, -1]),
            ],
            -1,
        ).contiguous()
        # TODO: use fp32 scale for glm_5
        tilert_wq_b_full_scales = torch.zeros(1, dtype=torch.bfloat16)
        tilert_q_norm_weight = rmsnorm_gamma.float().detach().clone()
        return tilert_wq_b_full, tilert_wq_b_full_scales, tilert_q_norm_weight

    @staticmethod
    def _swizzle_mma_16x16(mat_in: torch.Tensor) -> torch.Tensor:
        assert mat_in.shape[-2] == 16 and mat_in.shape[-1] == 16
        # PTX isa fig.88
        pre_shape = mat_in.shape[:-2]
        mat_in = mat_in.reshape(*pre_shape, 2, 8, 2, 4, 2).transpose(-4, -3).transpose(-5, -4)
        return mat_in.reshape(*pre_shape, 2 * 2, 8 * 4, 2).transpose(-3, -2)

    @staticmethod
    def _swizzle_mma_16x16_for_16x2048_4pages(mat_in: torch.Tensor) -> torch.Tensor:
        assert mat_in.shape[-2] == 16 and mat_in.shape[-1] == 2048
        pre_shape = mat_in.shape[:-2]
        mat_in = mat_in.reshape(*pre_shape, 16, 4, 512).transpose(-3, -2)
        mat_in = mat_in.reshape(*pre_shape, 4, 16, 32, 16).transpose(-3, -2)
        mat_in = RmsnormProjqWqibWeightsConverter._swizzle_mma_16x16(mat_in)
        return mat_in.contiguous()

    def _common_to_tilert_fp16mma(
        self,
        wq_b: torch.Tensor,
        wq_b_scale: torch.Tensor,
        wq_b_iq: torch.Tensor,
        wq_b_iq_scale: torch.Tensor,
        q_norm_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert common weights to TileRT FP16 MMA layout."""
        assert self.model_args.arch_name == "glm_5", "Only GLM-5 supports FP16 MMA"

        if wq_b_scale.dtype != torch.float32:
            print(
                "Warning: RmsnormProjqWqibWeightsConverter: "
                + f"wq_b_scale.dtype: {wq_b_scale.dtype} "
                + "is not float32, convert to float32."
            )
            wq_b_scale = wq_b_scale.to(torch.float32)
        if wq_b_iq_scale.dtype != torch.float32:
            print(
                "Warning: RmsnormProjqWqibWeightsConverter: "
                + f"wq_b_iq_scale.dtype: {wq_b_iq_scale.dtype} "
                + "is not float32, convert to float32."
            )
            wq_b_iq_scale = wq_b_iq_scale.to(torch.float32)

        sms = 128  # use 128 sms for glm_5
        pages = 4
        qk_dim = self.qk_head_dim * self.n_local_heads
        qk_dim_per_sm = qk_dim // sms  # 16 per sm
        qk_nope_dim = self.n_local_heads * self.qk_nope_head_dim
        qk_pe_dim = self.n_local_heads * self.qk_rope_head_dim
        iq_dim_per_sm = self.index_head_dim // sms  # 32 per sm

        wq_b_scale = wq_b_scale.reshape(
            self.n_local_heads, self.qk_head_dim // self.block_size, 1, self.q_lora_qdim
        ).repeat(
            1, 1, self.block_size, 1
        )  # 2048, 2048//128

        wq_b_scale = wq_b_scale.reshape(self.n_local_heads, self.qk_head_dim, -1)
        wq_b_nope_scale = (
            wq_b_scale[:, : self.qk_nope_head_dim, :]
            .reshape(qk_nope_dim // qk_dim_per_sm, qk_dim_per_sm, pages, self.q_lora_qdim // pages)
            .transpose(1, 2)  # (96, 4, 16, 4) for glm_5
        )

        wq_b_pe_scale = (
            wq_b_scale[:, self.qk_nope_head_dim :, :]
            .reshape(qk_pe_dim // qk_dim_per_sm, qk_dim_per_sm, pages, self.q_lora_qdim // pages)
            .transpose(1, 2)  # (32, 4, 16, 4) for glm_5
        )
        wq_b_scale = torch.cat([wq_b_nope_scale, wq_b_pe_scale], dim=0)
        wq_b_scale = wq_b_scale[:, :, 0, :]  # (128, 4, 4) for glm_5

        wq_b_iq_scale = wq_b_iq_scale.reshape(self.index_head_qdim, 1, self.q_lora_qdim).repeat(
            1, self.block_size, 1
        )  # (4096, 16) for glm_5
        wq_b_iq_scale = wq_b_iq_scale.reshape(
            sms, iq_dim_per_sm, pages, self.q_lora_qdim // pages
        ).transpose(1, 2)
        wq_b_iq_scale = wq_b_iq_scale[:, :, 0, :]  # (128, 4, 4) for glm_5

        wq_b_full_scales = (
            torch.cat([wq_b_scale, wq_b_iq_scale], dim=-1).contiguous().view(torch.float8_e4m3fn)
        )  # (128, 4, 8x4) for glm_5

        wq_b = wq_b.reshape(self.n_local_heads, self.qk_head_dim, self.q_lora_dim)
        wq_b_nope = wq_b[:, : self.qk_nope_head_dim, :].reshape(-1, self.q_lora_dim)  # 8x192, 2048
        wq_b_nope = RmsnormProjqWqibWeightsConverter._swizzle_mma_16x16_for_16x2048_4pages(
            wq_b_nope.reshape(qk_nope_dim // qk_dim_per_sm, qk_dim_per_sm, self.q_lora_dim)
        )
        wq_b_nope = wq_b_nope.reshape(qk_nope_dim // qk_dim_per_sm, pages, qk_dim_per_sm, -1)
        # (96, 4, 16, 512) for glm_5

        wq_b_pe = wq_b[:, self.qk_nope_head_dim :, :].reshape(-1, self.q_lora_dim)  # 8x64, 2048
        wq_b_pe = RmsnormProjqWqibWeightsConverter._swizzle_mma_16x16_for_16x2048_4pages(
            wq_b_pe.reshape(qk_pe_dim // qk_dim_per_sm, qk_dim_per_sm, self.q_lora_dim)
        )
        wq_b_pe = wq_b_pe.reshape(qk_pe_dim // qk_dim_per_sm, pages, qk_dim_per_sm, -1)
        # (32, 4, 16, 512) for glm_5
        wq_b = torch.cat([wq_b_nope, wq_b_pe], dim=0)
        # (128, 4, 16, 512) for glm_5

        wq_b_iq = RmsnormProjqWqibWeightsConverter._swizzle_mma_16x16_for_16x2048_4pages(
            wq_b_iq.reshape(sms, 2, iq_dim_per_sm // 2, self.q_lora_dim)
        )
        wq_b_iq = (
            wq_b_iq.reshape(sms, 2, pages, iq_dim_per_sm // 2, -1)
            .transpose(1, 2)
            .reshape(sms, pages, iq_dim_per_sm, -1)
        )
        # (128, 4, 32, 512) for glm_5
        wq_b = torch.cat([wq_b, wq_b_iq], dim=2)
        wq_b = wq_b.reshape(sms, pages, -1)
        # (128, 4, 48*512) for glm_5
        wq_b_scales_padding = torch.zeros(
            sms,
            pages,
            128 - wq_b_full_scales.shape[-1],
            dtype=torch.float8_e4m3fn,
            device=wq_b.device,
        )  # append 128-byte aligned scale: (128, 4, 24704) for glm_5
        tilert_wq_b_full = torch.cat(
            [wq_b, wq_b_full_scales, wq_b_scales_padding], dim=-1
        ).contiguous()
        tilert_wq_b_dummy_scales = torch.zeros(1, dtype=torch.bfloat16)
        tilert_q_norm_weight = q_norm_weight.float().detach().clone()
        return tilert_wq_b_full, tilert_wq_b_dummy_scales, tilert_q_norm_weight

    def convert_to_bf16(
        self, weights: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert common-format weights to TileRT BF16 layout.

        Args:
            weights: [q_norm_weight, wq_b, wq_b_scale, wq_b_iq, wq_b_iq_scale].
        """
        with torch.inference_mode():
            wq_b, wq_b_scale, wq_b_iq, wq_b_iq_scale, q_norm_weight = weights
            if self.model_args.arch_name == "glm_5":
                if wq_b_scale.dtype != torch.float32:
                    print(
                        "Warning: RmsnormProjqWqibWeightsConverter: "
                        + f"wq_b_scale.dtype: {wq_b_scale.dtype} "
                        + "is not float32, convert to float32."
                    )
                wq_b_scales = wq_b_scale.to(torch.float32)
                wq_b_iq_scales = wq_b_iq_scale.to(torch.float32)
                return self._common_to_tilert_bf16(
                    wq_b,
                    wq_b_scales,
                    wq_b_iq,
                    wq_b_iq_scales,
                    q_norm_weight,
                )

            # DS v3.2, use bfloat16 for wq_b_scale and wq_b_iq_scale
            wq_b_scales_bf16 = wq_b_scale.to(torch.bfloat16)
            wq_b_iq_scales_bf16 = wq_b_iq_scale.to(torch.bfloat16)
            return self._common_to_tilert_bf16(
                wq_b,
                wq_b_scales_bf16,
                wq_b_iq,
                wq_b_iq_scales_bf16,
                q_norm_weight,
            )

    def convert_to_fp8(
        self, weights: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert common-format weights to TileRT FP8 MMA layout.

        Args:
            weights: [q_norm_weight, wq_b, wq_b_scale, wq_b_iq, wq_b_iq_scale].
        """
        with torch.inference_mode():
            wq_b, wq_b_scale, wq_b_iq, wq_b_iq_scale, q_norm_weight = weights
            return self._common_to_tilert_fp8(
                wq_b,
                wq_b_scale,
                wq_b_iq,
                wq_b_iq_scale,
                q_norm_weight,
            )

    def convert_to_fp16mma(
        self, weights: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert common-format weights to TileRT FP16 MMA layout.

        Args:
            weights: [q_norm_weight, wq_b, wq_b_scale, wq_b_iq, wq_b_iq_scale].
        """
        with torch.inference_mode():
            wq_b, wq_b_scale, wq_b_iq, wq_b_iq_scale, q_norm_weight = weights
            return self._common_to_tilert_fp16mma(
                wq_b,
                wq_b_scale,
                wq_b_iq,
                wq_b_iq_scale,
                q_norm_weight,
            )


@dataclass
class RmsnormProjqWqibRefWeightsAlias:
    """Reference weights alias for RmsnormProjqWqib."""

    rmsnorm_gamma = "self_attn.q_a_layernorm.weight"
    wqb_weights = "self_attn.q_b_proj.weight"
    wqb_scales = "self_attn.q_b_proj.weight_scale_inv"
    wi_weights = "self_attn.indexer.wq_b.weight"
    wi_scales = "self_attn.indexer.wq_b.weight_scale_inv"

    @property
    def ref_tensor_alias(self) -> list[str]:
        return [
            self.rmsnorm_gamma,
            self.wqb_weights,
            self.wqb_scales,
            self.wi_weights,
            self.wi_scales,
        ]

    def __call__(self) -> list[str]:
        return self.ref_tensor_alias


@dataclass
class RmsnormProjqWqibTilertWeightsAlias:
    """TileRT weights alias for RmsnormProjqWqib."""

    rmsnorm_gamma = "q_rmsnorm_gamma"
    wqb_weights = "wqb_weights"
    wqb_scales = "wqb_scales"
    wi_weights = "wi_weights"
    wi_scales = "wi_scales"

    @property
    def tilert_tensor_alias(self) -> list[str]:
        return [
            self.rmsnorm_gamma,
            self.wqb_weights,
            self.wqb_scales,
            self.wi_weights,
            self.wi_scales,
        ]

    def __call__(self) -> list[str]:
        return self.tilert_tensor_alias


class RmsnormProjqWqib(TileRTModule):
    """RmsnormProjqWqib module: RMSNorm + Q projection (wq_b + wq_b_iq)."""

    def __init__(
        self,
        model_args: ModelArgs,
        device_id: int,
        num_devices: int,
        ref_weights_alias: RmsnormProjqWqibRefWeightsAlias | None = None,
    ):
        super().__init__(
            self.__class__.__name__,
            model_args=model_args,
            device_id=device_id,
            num_devices=num_devices,
        )

        self.tilert_weights_alias = RmsnormProjqWqibTilertWeightsAlias()
        self.ref_weights_alias = (
            ref_weights_alias
            if ref_weights_alias is not None
            else RmsnormProjqWqibRefWeightsAlias()
        )

        self.n_local_heads = model_args.n_heads // num_devices
        self.q_lora_rank = model_args.q_lora_rank
        self.index_n_heads = model_args.index_n_heads
        self.head_dim = model_args.index_head_dim
        self.index_head_dim = model_args.index_n_heads * model_args.index_head_dim
        self.n_heads = model_args.n_heads
        self.qk_head_dim = model_args.qk_nope_head_dim + model_args.qk_rope_head_dim
        self.qk_local_dim = self.qk_head_dim * self.n_local_heads
        self.qk_nope_head_dim = model_args.qk_nope_head_dim
        self.qk_rope_head_dim = model_args.qk_rope_head_dim

        # quantize block size
        self.block_size = model_args.block_size
        self.q_lora_qdim = self.q_lora_rank // self.block_size
        self.qk_local_qdim = self.qk_local_dim // self.block_size
        self.index_head_qdim = self.index_head_dim // self.block_size
        self.eps = model_args.eps

        self.ref_q_norm: torch.Tensor | None = None
        self.ref_wq_b: torch.Tensor | None = None
        self.ref_wq_b_iq: torch.Tensor | None = None

        self.tilert_wq_b_full: torch.Tensor | None = None
        self.tilert_wq_b_full_scales: torch.Tensor | None = None
        self.tilert_q_norm_weight: torch.Tensor | None = None

        self.q_nope: torch.Tensor | None = None
        self.q_pe: torch.Tensor | None = None
        self.iq: torch.Tensor | None = None

        self.profile_logs: torch.Tensor | None = None

    def get_weights_list(self) -> list[torch.Tensor]:
        return [self.tilert_q_norm_weight, self.tilert_wq_b_full, self.tilert_wq_b_full_scales]

    def device_sharding(self, weights_map: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Device sharding."""
        gamma = weights_map[self.ref_weights_alias.rmsnorm_gamma][None, ...].repeat(
            self.num_devices, 1
        )

        sharded_wqb_weights = weights_map[self.ref_weights_alias.wqb_weights].reshape(
            self.num_devices, self.qk_local_dim, self.q_lora_rank
        )
        sharded_wi_weights = weights_map[self.ref_weights_alias.wi_weights][None, ...].repeat(
            self.num_devices, 1, 1
        )

        sharded_wqb_scales = weights_map[self.ref_weights_alias.wqb_scales].reshape(
            self.num_devices, self.qk_local_qdim, self.q_lora_qdim
        )
        sharded_wi_scales = weights_map[self.ref_weights_alias.wi_scales][None, ...].repeat(
            self.num_devices, 1, 1
        )

        return {
            self.tilert_weights_alias.rmsnorm_gamma: gamma,
            self.tilert_weights_alias.wqb_weights: sharded_wqb_weights,
            self.tilert_weights_alias.wqb_scales: sharded_wqb_scales,
            self.tilert_weights_alias.wi_weights: sharded_wi_weights,
            self.tilert_weights_alias.wi_scales: sharded_wi_scales,
        }

    def init_reference_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Initialize reference weights from common-format state dict."""
        self.ref_q_norm = state_dict[self.ref_weights_alias.rmsnorm_gamma]
        qk_local_dim_start = self.qk_local_dim * self.device_id
        qk_local_qdim_start = qk_local_dim_start // self.block_size
        qk_local_dim_end = qk_local_dim_start + self.qk_local_dim
        qk_local_qdim_end = qk_local_dim_end // self.block_size
        wq_b = weight_dequant(
            state_dict[self.ref_weights_alias.wqb_weights][qk_local_dim_start:qk_local_dim_end],
            state_dict[self.ref_weights_alias.wqb_scales][qk_local_qdim_start:qk_local_qdim_end],
        )
        wq_b_iq = weight_dequant(
            state_dict[self.ref_weights_alias.wi_weights],
            state_dict[self.ref_weights_alias.wi_scales],
        )
        self.ref_wq_b = wq_b.contiguous()
        self.ref_wq_b_iq = wq_b_iq.contiguous()

    def init_tilert_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Initialize TileRT weights from common-format state dict."""
        weights = [
            state_dict[_k]
            for _k in [
                self.tilert_weights_alias.wqb_weights,
                self.tilert_weights_alias.wqb_scales,
                self.tilert_weights_alias.wi_weights,
                self.tilert_weights_alias.wi_scales,
                self.tilert_weights_alias.rmsnorm_gamma,
            ]
        ]
        assert self.algorithm is not None, "Algorithm is not set"
        self.tilert_wq_b_full, self.tilert_wq_b_full_scales, self.tilert_q_norm_weight = (
            RmsnormProjqWqibWeightsConverter(self.model_args, self.num_devices).dispatch(
                self.algorithm, weights
            )
        )

    def init_random_weights(self) -> None:
        """Initialize random reference and TileRT weights for testing."""
        q_norm = torch.randn(self.q_lora_rank, dtype=torch.float32)
        wq_b = torch.randn(
            self.num_devices * self.qk_local_dim, self.q_lora_rank, dtype=torch.bfloat16
        ).to(torch.float8_e4m3fn)
        scale_dtype = torch.float32 if self.model_args.arch_name == "glm_5" else torch.bfloat16
        wq_b_scale = torch.randn(
            self.num_devices * self.qk_local_qdim, self.q_lora_qdim, dtype=scale_dtype
        )
        wq_b_iq = torch.randn(self.index_head_dim, self.q_lora_rank, dtype=torch.bfloat16).to(
            torch.float8_e4m3fn
        )
        wq_b_iq_scale = torch.randn(self.index_head_qdim, self.q_lora_qdim, dtype=scale_dtype)
        ref_state = {
            self.ref_weights_alias.rmsnorm_gamma: q_norm,
            self.ref_weights_alias.wqb_weights: wq_b,
            self.ref_weights_alias.wqb_scales: wq_b_scale,
            self.ref_weights_alias.wi_weights: wq_b_iq,
            self.ref_weights_alias.wi_scales: wq_b_iq_scale,
        }

        self.init_reference_weights(ref_state)
        self.init_tilert_weights(
            {_k: _v[self.device_id] for _k, _v in self.device_sharding(ref_state).items()}
        )

    def init_tilert_vars(self, batch_size: int, seq_len: int) -> None:
        """Allocate TileRT output buffers."""
        self.q_nope = torch.zeros(
            batch_size, seq_len, self.n_local_heads, self.qk_nope_head_dim, dtype=torch.bfloat16
        )
        self.q_pe = torch.zeros(
            batch_size, seq_len, self.n_local_heads, self.qk_rope_head_dim, dtype=torch.bfloat16
        )
        self.iq = torch.zeros(
            batch_size, seq_len, self.index_n_heads, self.head_dim, dtype=torch.bfloat16
        )
        self.profile_logs = get_profile_log_tensor()
        self.is_var_init = True

    def golden_forward(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reference forward: RMSNorm + linear projections."""
        assert self.ref_q_norm is not None
        assert self.ref_wq_b is not None
        assert self.ref_wq_b_iq is not None

        bsz, seqlen, _ = q.shape
        if bsz != 1 or seqlen not in [1, 2, 4]:
            raise ValueError(f"Invalid batch size or sequence length: bsz={bsz}, seqlen={seqlen}")

        qr = torch.nn.functional.rms_norm(q.float(), [q.size(-1)], self.ref_q_norm, self.eps).to(
            q.dtype
        )

        q = torch.matmul(qr, self.ref_wq_b.T)
        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_idx = torch.matmul(qr, self.ref_wq_b_iq.T)
        q_idx = rearrange(q_idx, "b s (h d) -> b s h d", d=self.head_dim)
        return q_nope, q_pe, q_idx

    def tilert_forward(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.tilert_wq_b_full is not None
        assert self.tilert_wq_b_full_scales is not None
        assert self.tilert_q_norm_weight is not None
        assert self.q_nope is not None
        assert self.q_pe is not None
        assert self.iq is not None
        assert self.profile_logs is not None

        bsz, seqlen, _ = q.shape
        if bsz != 1 or seqlen not in [1, 2, 4]:
            raise ValueError(f"Invalid batch size or sequence length: bsz={bsz}, seqlen={seqlen}")

        assert self.algorithm is not None, "Algorithm is not set"

        rmsnorm_projq_wqib_op(
            q,
            self.tilert_wq_b_full,
            self.tilert_wq_b_full_scales,
            self.tilert_q_norm_weight,
            self.q_nope,
            self.q_pe,
            self.iq,
            self.profile_logs,
            self.algorithm.value,
        )

        if self.flag_enable_profiling_log:
            torch.cuda.synchronize()
            parse_profile_log_tensor(
                self.profile_logs, self.get_profile_log_path(), [(self.op_name, 0.0)]
            )
        return self.q_nope, self.q_pe, self.iq
