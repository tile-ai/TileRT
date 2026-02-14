"""RMSNormProjxWqkvia operation module."""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

# from typing import Any
import torch

from tilert.models.base import TileRTModule, TilertWeightsConverter
from tilert.models.common import weight_dequant
from tilert.models.deepseek_v3_2.model_args import ModelArgs
from tilert.models.deepseek_v3_2.ops.rmsnorm_quant import rmsnorm_quant
from tilert.profiler.utils import parse_profile_log_tensor
from tilert.utils import get_profile_log_tensor

__all__ = [
    "RMSNormProjQAKVAKIWeightsConverter",
    "RMSNormProjxWqkviaAlgorithm",
    "RMSNormProjxWqkvia",
    "RMSNormProjxWqkviaRefWeightsAlias",
    "RMSNormProjxWqkviaTilertWeightsAlias",
    "rmsnorm_projx_wqkvia",
    "projx_wqkvia",
]


def rmsnorm_projx_wqkvia(
    x_in: torch.Tensor,
    wqkv_a: torch.Tensor,
    wqkv_a_scales: torch.Tensor,
    rmsnorm_gamma: torch.Tensor,
    cur_pos: torch.Tensor,
    q_out: torch.Tensor,
    kv_out: torch.Tensor,
    pe_cache: torch.Tensor,
    ki_out: torch.Tensor,
    x_rmsnorm_out: torch.Tensor,
    profile_logs: torch.Tensor,
) -> None:
    """
    rmsnorm_projx_wqkvia operation.

    Args:
        x_in: Input tensor.
        wqkv_a: QKV weights.
        wqkv_a_scales: QKV scales.
        rmsnorm_gamma: RMSNorm gamma.
        cur_pos: Current position.
        q_out: Q output tensor.
        kv_out: KV output tensor.
        pe_cache: PE cache tensor.
        ki_out: Ki output tensor.
        x_rmsnorm_out: RMSNorm output tensor.
        profile_logs: Profile logs tensor.
    """
    torch.ops.tilert.rmsnorm_proj_qa_kva_ki_op(
        x_in,
        wqkv_a,
        wqkv_a_scales,
        rmsnorm_gamma,
        cur_pos,
        q_out,
        kv_out,
        pe_cache,
        ki_out,
        x_rmsnorm_out,
        profile_logs,
    )


def projx_wqkvia(
    x_quant: torch.Tensor,
    x_scale: torch.Tensor,
    wqkvia: torch.Tensor,
    cur_pos: torch.Tensor,
    out_q: torch.Tensor,
    out_kv: torch.Tensor,
    pe_cache: torch.Tensor,
    out_ki: torch.Tensor,
    profile_logs: torch.Tensor,
) -> None:
    """
    Define the ProjXWQKVIa operation.

    Args:
        x_quant: Input tensor.
        x_scale: Weight tensor.
        wqkvia: Weight tensor.
        cur_pos: Current position tensor.
        out_q: Output tensor.
        out_kv: Output tensor.
        pe_cache: Output tensor.
        out_ki: Output tensor.
        profile_logs: Profile logs tensor.
    """
    dim = x_quant.shape[-1]
    if dim == 6144:
        func_call = torch.ops.tilert.projx_wqkvia_glm5
    elif dim == 7168:
        func_call = torch.ops.tilert.projx_wqkvia_op
    else:
        raise ValueError(f"Unsupported dimension: {dim}")
    func_call(x_quant, x_scale, wqkvia, cur_pos, out_q, out_kv, pe_cache, out_ki, profile_logs)


class RMSNormProjxWqkviaAlgorithm(Enum):
    """RMSNormProjxWqkvia algorithm"""

    GENERAL = "general"  # fused
    DECOUPLED = "decoupled"  # rmsnorm_quant + projx_wqkvia


class RMSNormProjQAKVAKIWeightsConverter:
    """Weights converter class."""

    @staticmethod
    def _swizzle_mma_16x32(mat_in: torch.Tensor) -> torch.Tensor:
        assert mat_in.shape[-2] == 16 and mat_in.shape[-1] == 32
        # PTX isa fig.88
        pre_shape = mat_in.shape[:-2]
        mat_in = mat_in.reshape(*pre_shape, 2, 8, 2, 4, 4).transpose(-4, -3).transpose(-5, -4)
        return mat_in.reshape(*pre_shape, 2 * 2, 8 * 4, 4).transpose(-3, -2)

    @staticmethod
    def tilert_to_common(
        tilert_wqkv_a: torch.Tensor,
        tilert_wqkv_a_scales: torch.Tensor,
        tilert_attn_norm_weight: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Convert tilert weights to common weights.

        Args:
            tilert_wqkv_a: Tilert weight tensor.
            tilert_wqkv_a_scales: Tilert weight scale tensor.
            tilert_attn_norm_weight: Tilert attention norm weight tensor.
        Returns:
            tuple: Common weights.
        """
        wq_a = tilert_wqkv_a[:1536]  # 1536, 7168
        wkv_a = tilert_wqkv_a[1536 : 1536 + 576]  # 576, 7168
        wk = tilert_wqkv_a[1536 + 576 :]  # 128, 7168

        wqkv_a_scales_0 = tilert_wqkv_a_scales[:128, :].reshape(16, 8, 64)
        wqkv_a_scales_0 = wqkv_a_scales_0[:, 0, :].reshape(16, 64)
        wqkv_a_scales_1 = tilert_wqkv_a_scales[128:129, :]  # 1, 64
        wqkv_a_scales_2 = tilert_wqkv_a_scales[129:, :]  # 1, 64
        wqkv_a_scales_swizzled = torch.cat(
            [wqkv_a_scales_0, wqkv_a_scales_1, wqkv_a_scales_2], dim=0
        )
        wqkv_scales = torch.zeros(
            (18, 56), dtype=torch.bfloat16, device=tilert_wqkv_a_scales.device
        )

        for i in range(64):
            if ((i % 8) * 8 + i // 8) < 56:
                wqkv_scales[:, ((i % 8) * 8 + i // 8)] = wqkv_a_scales_swizzled[:, i]
        wq_a_scale = wqkv_scales[:12, :]  # 12, 56
        wkv_a_scale = wqkv_scales[12:17, :]  # 5, 56
        wk_scale = wqkv_scales[17:, :]  # 1, 56

        attn_norm_weight = tilert_attn_norm_weight
        return wq_a, wq_a_scale, wkv_a, wkv_a_scale, wk, wk_scale, attn_norm_weight

    @staticmethod
    def common_to_tilert(
        wq_a: torch.Tensor,
        wq_a_scale: torch.Tensor,
        wkv_a: torch.Tensor,
        wkv_a_scale: torch.Tensor,
        wk: torch.Tensor,
        wk_scale: torch.Tensor,
        attn_norm_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert common weights to tilert weights.

        Args:
            wq_a: Common weight tensor.
            wq_a_scale: Common weight scale tensor.
            wkv_a: Common weight tensor.
            wkv_a_scale: Common weight scale tensor.
            wk: Common weight tensor.
            wk_scale: Common weight scale tensor.
            attn_norm_weight: Common attention norm weight tensor.
        Returns:
            tuple: Tilert weights.
        """
        wqkv_a = torch.cat([wq_a, wkv_a, wk], dim=0)
        wqkv_a_scales_raw = torch.cat([wq_a_scale, wkv_a_scale, wk_scale], dim=0)

        wqkv_a_scales = torch.zeros((18, 64), dtype=torch.bfloat16, device=wq_a_scale.device)
        for i in range(64):
            wqkv_a_scales[:, i] = wqkv_a_scales_raw[:, ((i % 8) * 8 + i // 8) % 56]
            if ((i % 8) * 8 + i // 8) >= 56:
                wqkv_a_scales[:, i] = 0.0
        wqkv_a_scales_0 = wqkv_a_scales[:16, :]
        wqkv_a_scales_1 = wqkv_a_scales[16:17, :]
        wqkv_a_scales_2 = wqkv_a_scales[17:, :]

        wqkv_a_scales_0 = wqkv_a_scales_0.reshape((16, 1, 64)).repeat(1, 8, 1).reshape(-1, 64)
        wqkv_a_scales = torch.cat([wqkv_a_scales_0, wqkv_a_scales_1, wqkv_a_scales_2], dim=0)
        assert wqkv_a_scales.shape == (130, 64)
        return wqkv_a.contiguous(), wqkv_a_scales.contiguous(), attn_norm_weight.clone()

    @staticmethod
    def common_to_tilert_fp8(
        wq_a: torch.Tensor,
        wq_a_scale: torch.Tensor,
        wkv_a: torch.Tensor,
        wkv_a_scale: torch.Tensor,
        wk: torch.Tensor,
        wk_scale: torch.Tensor,
        attn_norm_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert common weights to tilert weights.

        Args:
            wq_a: Common weight tensor.
            wq_a_scale: Common weight scale tensor.
            wkv_a: Common weight tensor.
            wkv_a_scale: Common weight scale tensor.
            wk: Common weight tensor.
            wk_scale: Common weight scale tensor.
            attn_norm_weight: Common attention norm weight tensor.
        Returns:
            tuple: Tilert fp8 weights.
        """
        wq_a_raw: torch.Tensor = wq_a.detach().clone()
        wkv_a_raw: torch.Tensor = wkv_a.detach().clone()
        wq_a_raw = torch.cat([wq_a_raw, wkv_a_raw[:512], wk, wkv_a_raw[512:]], dim=0)

        wq_a_raw = wq_a_raw.reshape(35, 64, 14, 512)
        wq_a_raw = wq_a_raw.permute(0, 2, 1, 3)

        wq_a_raw = wq_a_raw.reshape(35, 14, 16, 4, 4, 128)
        wq_a_copy = wq_a_raw.contiguous().clone()
        wq_a_raw[:, :, 1::2, :, :, :64] = wq_a_copy[:, :, 1::2, :, :, 64:]
        wq_a_raw[:, :, 1::2, :, :, 64:] = wq_a_copy[:, :, 1::2, :, :, :64]
        wq_a_raw = wq_a_raw.reshape(35, 14, 16, 4, 4, 2, 64)
        wq_a_copy = wq_a_raw.contiguous().clone()
        wq_a_raw[:, :, :, 2:, :, :, :32] = wq_a_copy[:, :, :, 2:, :, :, 32:]
        wq_a_raw[:, :, :, 2:, :, :, 32:] = wq_a_copy[:, :, :, 2:, :, :, :32]
        wq_a_raw = wq_a_raw.reshape(35, 14, 16, 4, 4, 2, 2, 32)
        wq_a_copy = wq_a_raw.contiguous().clone()
        wq_a_raw[:, :, :, 1::2, :, :, :, :16] = wq_a_copy[:, :, :, 1::2, :, :, :, 16:]
        wq_a_raw[:, :, :, 1::2, :, :, :, 16:] = wq_a_copy[:, :, :, 1::2, :, :, :, :16]

        wq_a_raw = wq_a_raw.reshape(35, 14, 16, 4, 4, 128)
        wq_a_raw = wq_a_raw.permute(0, 1, 4, 2, 3, 5).reshape(35, 14, -1).contiguous()
        wq_a_raw = wq_a_raw.reshape(35, 14, -1).contiguous()

        wq_s_raw: torch.Tensor = wq_a_scale.detach().clone()
        wkv_s_raw: torch.Tensor = wkv_a_scale.detach().clone()
        wq_s_raw = torch.cat([wq_s_raw, wkv_s_raw[:4], wk_scale, wkv_s_raw[4:]], dim=0)
        wq_s_raw = wq_s_raw.reshape(18, 1, 14, 4).repeat(1, 2, 1, 1).reshape(36, 1, 14, 4)
        wq_s_raw = wq_s_raw[:35].reshape(35, 14, -1).contiguous()
        wq_s_raw = wq_s_raw.view(torch.float8_e4m3fn)
        wq_as_raw = torch.cat([wq_a_raw, wq_s_raw], dim=-1)

        return wq_as_raw.contiguous(), attn_norm_weight.clone()

    @staticmethod
    def common_to_tilert_native_bf16(
        wq_a: torch.Tensor,
        wq_a_scale: torch.Tensor,
        wkv_a: torch.Tensor,
        wkv_a_scale: torch.Tensor,
        wk: torch.Tensor,
        wk_scale: torch.Tensor,
        attn_norm_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert common weights to weights for tilert native bf16 op.

        Args:
            wq_a: Common weight tensor.
            wq_a_scale: Common weight scale tensor.
            wkv_a: Common weight tensor.
            wkv_a_scale: Common weight scale tensor.
            wk: Common weight tensor.
            wk_scale: Common weight scale tensor.
            attn_norm_weight: Common attention norm weight tensor.
        Returns:
            tuple: Tilert weights for native bf16 op.
        """
        wq_a_scale = wq_a_scale.reshape((12, 56, 1)).repeat(1, 1, 128).reshape((12, 1, 7168))
        wq_a_scale = wq_a_scale.repeat(1, 128, 1).reshape((1536, 7168))
        wkv_a_scale = wkv_a_scale.reshape((5, 56, 1)).repeat(1, 1, 128).reshape((5, 1, 7168))
        wkv_a_scale = wkv_a_scale.repeat(1, 128, 1).reshape((-1, 7168))
        wkv_a_scale = wkv_a_scale[:576]
        wk_scale = wk_scale.reshape((1, 56, 1)).repeat(1, 1, 128).reshape((1, 1, 7168))
        wk_scale = wk_scale.repeat(1, 128, 1).reshape((128, 7168))
        wq_a = wq_a.reshape((1536, 7168)).float() * wq_a_scale.float()
        wkv_a = wkv_a.reshape((576, 7168)).float() * wkv_a_scale.float()
        wk = wk.reshape((128, 7168)).float() * wk_scale.float()
        weights = torch.cat([wq_a, wkv_a, wk], dim=0)
        assert weights.shape == (1536 + 576 + 128, 7168)
        return weights.to(torch.bfloat16).contiguous(), attn_norm_weight.clone()

    @staticmethod
    def common_to_tilert_native_bf16_warp_gemv(
        wq_a: torch.Tensor,
        wq_a_scale: torch.Tensor,
        wkv_a: torch.Tensor,
        wkv_a_scale: torch.Tensor,
        wk: torch.Tensor,
        wk_scale: torch.Tensor,
        attn_norm_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert common weights to weights for tilert native bf16 warp gemv op.

        Args:
            wq_a: Common weight tensor.
            wq_a_scale: Common weight scale tensor.
            wkv_a: Common weight tensor.
            wkv_a_scale: Common weight scale tensor.
            wk: Common weight tensor.
            wk_scale: Common weight scale tensor.
            attn_norm_weight: Common attention norm weight tensor.
        Returns:
            tuple: Tilert weights for native bf16 warp gemv op.
        """
        wq_a_scale = wq_a_scale.reshape((12, 56, 1)).repeat(1, 1, 128).reshape((12, 1, 7168))
        wq_a_scale = wq_a_scale.repeat(1, 128, 1).reshape((1536, 7168))
        wkv_a_scale = wkv_a_scale.reshape((5, 56, 1)).repeat(1, 1, 128).reshape((5, 1, 7168))
        wkv_a_scale = wkv_a_scale.repeat(1, 128, 1).reshape((-1, 7168))
        wkv_a_scale = wkv_a_scale[:576]
        wk_scale = wk_scale.reshape((1, 56, 1)).repeat(1, 1, 128).reshape((1, 1, 7168))
        wk_scale = wk_scale.repeat(1, 128, 1).reshape((128, 7168))
        wq_a = wq_a.reshape((1536, 7168)).float() * wq_a_scale.float()
        wkv_a = wkv_a.reshape((576, 7168)).float() * wkv_a_scale.float()
        wk = wk.reshape((128, 7168)).float() * wk_scale.float()
        # concatenate the weights
        weights = torch.cat([wq_a, wkv_a, wk], dim=0)
        assert weights.shape == (1536 + 576 + 128, 7168)

        weights = weights.reshape(140, 16, 7, 1024)
        weights = weights.transpose(1, 2)  # 140, 7, 16, 1024
        return weights.to(torch.bfloat16).contiguous(), attn_norm_weight.clone()

    @staticmethod
    def common_to_tilert_dequant_bf16(
        wq_a: torch.Tensor,
        wq_a_scale: torch.Tensor,
        wkv_a: torch.Tensor,
        wkv_a_scale: torch.Tensor,
        wk: torch.Tensor,
        wk_scale: torch.Tensor,
        attn_norm_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert common weights to weights for tilert dequant bf16 op.

        Args:
            wq_a: Common weight tensor.
            wq_a_scale: Common weight scale tensor.
            wkv_a: Common weight tensor.
            wkv_a_scale: Common weight scale tensor.
            wk: Common weight tensor.
            wk_scale: Common weight scale tensor.
            attn_norm_weight: Common attention norm weight tensor.
        Returns:
            tuple: Tilert weights for dequant bf16 op.
        """
        wq_a = wq_a.reshape((384, 4, 7168))
        wkv_a = wkv_a.reshape((144, 4, 7168))
        wk = wk.reshape((32, 4, 7168))
        wqkv = torch.cat([wq_a, wkv_a, wk], dim=0).reshape(140, 4, 4 * 7168)

        wq_a_scale = wq_a_scale.reshape((12, 1, 56)).repeat(1, 32, 1).reshape((384, 1, 56))
        wkv_a_scale = wkv_a_scale.reshape((5, 1, 56)).repeat(1, 32, 1).reshape((160, 1, 56))[:144]
        wk_scale = wk_scale.reshape((1, 1, 56)).repeat(1, 32, 1).reshape((32, 1, 56))
        wqkv_scales = torch.cat([wq_a_scale, wkv_a_scale, wk_scale], dim=0).reshape(140, 4, 56)
        wqkv_scales_swizzled = torch.zeros(140, 4, 64, dtype=torch.bfloat16, device=wq_a.device)
        # swizzle
        for i in range(64):
            wqkv_scales_swizzled[..., i] = wqkv_scales[..., ((i % 8) * 8 + i // 8) % 56]
        weights = torch.zeros(
            140, 4, 4 * 7168 + 64 * 2, dtype=torch.float8_e4m3fn, device=wq_a.device
        )
        weights_part = weights[:, :, : 4 * 7168]
        scales_part = weights[:, :, 4 * 7168 :]
        weights_part.copy_(wqkv)
        scales_part.copy_(wqkv_scales_swizzled.view(dtype=torch.float8_e4m3fn))
        return weights.contiguous(), attn_norm_weight.clone()

    @staticmethod
    def common_to_tilert_fp8_mma(
        wq_a: torch.Tensor,
        wq_a_scale: torch.Tensor,
        wkv_a: torch.Tensor,
        wkv_a_scale: torch.Tensor,
        wk: torch.Tensor,
        wk_scale: torch.Tensor,
        rmsnorm_gamma: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert common weights to weights for tilert fp8 mma op.

        Args:
            wq_a: Common weight tensor.
            wq_a_scale: Common weight scale tensor.
            wkv_a: Common weight tensor.
            wkv_a_scale: Common weight scale tensor.
            wk: Common weight tensor.
            wk_scale: Common weight scale tensor.
            rmsnorm_gamma: Common rmsnorm gamma tensor.
        Returns:
            tuple: Tilert weights for fp8 mma op.
        """
        assert wq_a.shape == (1536, 7168)
        assert wq_a_scale.shape == (12, 56)
        assert wkv_a.shape == (576, 7168)
        assert wkv_a_scale.shape == (5, 56)
        assert wk.shape == (128, 7168)
        assert wk_scale.shape == (1, 56)
        wq_a = wq_a.reshape(96, 16, 7168)
        wq_a_scale = wq_a_scale.reshape(12, 1, 56).repeat(1, 8, 1).reshape(96, 56)
        wkv_a = wkv_a.reshape(36, 16, 7168)
        wkv_a_scale = wkv_a_scale.reshape(5, 1, 56).repeat(1, 8, 1).reshape(40, 56)
        wkv_a_scale = wkv_a_scale[:36]

        wk = wk.reshape(8, 16, 7168)
        wk_scale = wk_scale.reshape(1, 1, 56).repeat(1, 8, 1).reshape(8, 56)
        wqkvia = torch.cat([wq_a, wkv_a, wk], dim=0)  # 140, 7168
        wqkvia_scale = torch.cat([wq_a_scale, wkv_a_scale, wk_scale], dim=0)  # 140, 56

        wqkvia_0 = wqkvia[..., :2048]
        wqkvia_0_scale = wqkvia_scale[..., :16].contiguous().view(torch.float8_e4m3fn)
        wqkvia_1 = wqkvia[..., 2048:4096]
        wqkvia_1_scale = wqkvia_scale[..., 16:32].contiguous().view(torch.float8_e4m3fn)
        wqkvia_2 = wqkvia[..., 4096:6144]
        wqkvia_2_scale = wqkvia_scale[..., 32:48].contiguous().view(torch.float8_e4m3fn)
        wqkvia_3 = wqkvia[..., 6144:7168]
        wqkvia_3_scale = wqkvia_scale[..., 48:56].contiguous().view(torch.float8_e4m3fn)

        wqkvia_0 = wqkvia_0.reshape(140, 16, 64, 32).transpose(1, 2)
        wqkvia_0 = RMSNormProjQAKVAKIWeightsConverter._swizzle_mma_16x32(wqkvia_0)
        wqkvia_0 = wqkvia_0.reshape(140, 16 * 2048)

        wqkvia_1 = wqkvia_1.reshape(140, 16, 64, 32).transpose(1, 2)
        wqkvia_1 = RMSNormProjQAKVAKIWeightsConverter._swizzle_mma_16x32(wqkvia_1)
        wqkvia_1 = wqkvia_1.reshape(140, 16 * 2048)

        wqkvia_2 = wqkvia_2.reshape(140, 16, 64, 32).transpose(1, 2)
        wqkvia_2 = RMSNormProjQAKVAKIWeightsConverter._swizzle_mma_16x32(wqkvia_2)
        wqkvia_2 = wqkvia_2.reshape(140, 16 * 2048)

        wqkvia_3 = wqkvia_3.reshape(140, 16, 32, 32).transpose(1, 2)
        wqkvia_3 = RMSNormProjQAKVAKIWeightsConverter._swizzle_mma_16x32(wqkvia_3)
        wqkvia_3 = wqkvia_3.reshape(140, 16 * 1024)
        padding_scale0 = torch.zeros((140, 48), dtype=torch.bfloat16, device=wq_a.device).view(
            torch.float8_e4m3fn
        )
        padding_scale1 = torch.zeros((140, 48), dtype=torch.bfloat16, device=wq_a.device).view(
            torch.float8_e4m3fn
        )
        padding_scale2 = torch.zeros((140, 48), dtype=torch.bfloat16, device=wq_a.device).view(
            torch.float8_e4m3fn
        )
        padding_scale3 = torch.zeros((140, 56), dtype=torch.bfloat16, device=wq_a.device).view(
            torch.float8_e4m3fn
        )
        wqkvia = torch.cat(
            [
                wqkvia_0,
                wqkvia_0_scale,
                padding_scale0,
                wqkvia_1,
                wqkvia_1_scale,
                padding_scale1,
                wqkvia_2,
                wqkvia_2_scale,
                padding_scale2,
                wqkvia_3,
                wqkvia_3_scale,
                padding_scale3,
            ],
            dim=1,
        )

        return wqkvia.contiguous(), rmsnorm_gamma.contiguous()


class RMSNormProjxWqkviaWeightsConverter(TilertWeightsConverter):
    """RMSNormProjxWqkvia weights converter"""

    @staticmethod
    def _swizzle_qmma_16x32(mat_in: torch.Tensor) -> torch.Tensor:
        assert mat_in.shape[-2] == 16 and mat_in.shape[-1] == 32
        assert mat_in.dtype == torch.float8_e4m3fn
        # PTX isa fig.88
        pre_shape = mat_in.shape[:-2]
        mat_in = mat_in.reshape(*pre_shape, 2, 8, 2, 4, 4).transpose(-4, -3).transpose(-5, -4)
        return mat_in.reshape(*pre_shape, 2 * 2, 8 * 4, 4).transpose(-3, -2)

    def convert_to_general(self, weights: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert the weights to general format.

        Args:
            weights: List of weights.

        Returns:
            Tuple of weights.
        """
        # Specialized for DS v3.2 model
        args = self.model_args
        assert (
            args.arch_name == "deepseek_v3_2"
        ), f"arch_name must be deepseek_v3_2, but got {args.arch_name}"
        with torch.inference_mode():
            x_rmsnorm_gamma, wq_a, wq_a_scale, wkv_a, wkv_a_scale, wk, wk_scale = weights
            q_lora_rank_scale_dim = args.q_lora_rank // args.block_size
            kv_lora_rank_scale_dim = args.kv_lora_rank // args.block_size + 1
            x_scale_dim = args.dim // args.block_size

            wq_a_scale = (
                wq_a_scale.reshape((q_lora_rank_scale_dim, x_scale_dim, 1))
                .repeat(1, 1, args.block_size)
                .reshape((q_lora_rank_scale_dim, 1, args.dim))
            )
            wq_a_scale = wq_a_scale.repeat(1, args.block_size, 1).reshape(
                (args.q_lora_rank, args.dim)
            )
            wkv_a_scale = (
                wkv_a_scale.reshape((kv_lora_rank_scale_dim, x_scale_dim, 1))
                .repeat(1, 1, args.block_size)
                .reshape((kv_lora_rank_scale_dim, 1, args.dim))
            )
            wkv_a_scale = wkv_a_scale.repeat(1, args.block_size, 1).reshape((-1, args.dim))
            wkv_a_scale = wkv_a_scale[: args.kv_lora_rank + args.qk_rope_head_dim]
            wk_scale = (
                wk_scale.reshape((1, x_scale_dim, 1))
                .repeat(1, 1, args.block_size)
                .reshape((1, 1, args.dim))
            )
            wk_scale = wk_scale.repeat(1, args.block_size, 1).reshape(
                (args.index_head_dim, args.dim)
            )
            wq_a = wq_a.reshape((args.q_lora_rank, args.dim)).float() * wq_a_scale.float()
            wkv_a = (
                wkv_a.reshape((args.kv_lora_rank + args.qk_rope_head_dim, args.dim)).float()
                * wkv_a_scale.float()
            )
            wk = wk.reshape((args.index_head_dim, args.dim)).float() * wk_scale.float()
            # concatenate the weights
            weights_tensor: torch.Tensor = torch.cat([wq_a, wkv_a, wk], dim=0)
            assert weights_tensor.shape == (
                args.q_lora_rank + args.kv_lora_rank + args.qk_rope_head_dim + args.index_head_dim,
                args.dim,
            )
            # hard-coded scheduling: reshape to 140, 16, 7, 1024
            weights_tensor = weights_tensor.reshape(140, 16, 7, 1024)
            weights_tensor = weights_tensor.transpose(1, 2)  # 140, 7, 16, 1024
        return x_rmsnorm_gamma, weights_tensor.to(torch.bfloat16).contiguous()

    def convert_to_decoupled(
        self, weights: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert the weights to decoupled format.

        Args:
            weights: List of weights.

        Returns:
            Tuple of weights.
        """
        arch_name = self.model_args.arch_name
        wqkvia_and_scales = None
        with torch.inference_mode():
            x_rmsnorm_gamma, wq_a, wq_a_scale, wkv_a, wkv_a_scale, wk, wk_scale = weights
            # Ensure the scales are in bfloat16
            if arch_name == "deepseek_v3_2":  # DS v3.2
                # Ensure the scales are in bfloat16 for DS v3.2
                wq_a_scale = wq_a_scale.to(torch.bfloat16)
                wkv_a_scale = wkv_a_scale.to(torch.bfloat16)
                wk_scale = wk_scale.to(torch.bfloat16)
                assert wq_a.shape == (1536, 7168)
                assert wq_a_scale.shape == (12, 56)
                assert wkv_a.shape == (576, 7168)
                assert wkv_a_scale.shape == (5, 56)
                assert wk.shape == (128, 7168)
                assert wk_scale.shape == (1, 56)
                wq_a = wq_a.reshape(96, 16, 7168)
                wq_a_scale = wq_a_scale.reshape(12, 1, 56).repeat(1, 8, 1).reshape(96, 56)
                wkv_a = wkv_a.reshape(36, 16, 7168)
                wkv_a_scale = wkv_a_scale.reshape(5, 1, 56).repeat(1, 8, 1).reshape(40, 56)
                wkv_a_scale = wkv_a_scale[:36]

                wk = wk.reshape(8, 16, 7168)
                wk_scale = wk_scale.reshape(1, 1, 56).repeat(1, 8, 1).reshape(8, 56)
                wqkvia = torch.cat([wq_a, wkv_a, wk], dim=0)  # 140, 7168
                wqkvia_scale = torch.cat([wq_a_scale, wkv_a_scale, wk_scale], dim=0)  # 140, 56

                wqkvia_0 = wqkvia[..., :2048]
                wqkvia_0_scale = wqkvia_scale[..., :16].contiguous().view(torch.float8_e4m3fn)
                wqkvia_1 = wqkvia[..., 2048:4096]
                wqkvia_1_scale = wqkvia_scale[..., 16:32].contiguous().view(torch.float8_e4m3fn)
                wqkvia_2 = wqkvia[..., 4096:6144]
                wqkvia_2_scale = wqkvia_scale[..., 32:48].contiguous().view(torch.float8_e4m3fn)
                wqkvia_3 = wqkvia[..., 6144:7168]
                wqkvia_3_scale = wqkvia_scale[..., 48:56].contiguous().view(torch.float8_e4m3fn)

                wqkvia_0 = wqkvia_0.reshape(140, 16, 64, 32).transpose(1, 2)
                wqkvia_0 = self._swizzle_qmma_16x32(wqkvia_0)
                wqkvia_0 = wqkvia_0.reshape(140, 16 * 2048)

                wqkvia_1 = wqkvia_1.reshape(140, 16, 64, 32).transpose(1, 2)
                wqkvia_1 = self._swizzle_qmma_16x32(wqkvia_1)
                wqkvia_1 = wqkvia_1.reshape(140, 16 * 2048)

                wqkvia_2 = wqkvia_2.reshape(140, 16, 64, 32).transpose(1, 2)
                wqkvia_2 = self._swizzle_qmma_16x32(wqkvia_2)
                wqkvia_2 = wqkvia_2.reshape(140, 16 * 2048)

                wqkvia_3 = wqkvia_3.reshape(140, 16, 32, 32).transpose(1, 2)
                wqkvia_3 = self._swizzle_qmma_16x32(wqkvia_3)
                wqkvia_3 = wqkvia_3.reshape(140, 16 * 1024)
                padding_scale0 = torch.zeros(
                    (140, 48), dtype=torch.bfloat16, device=wq_a.device
                ).view(torch.float8_e4m3fn)
                padding_scale1 = torch.zeros(
                    (140, 48), dtype=torch.bfloat16, device=wq_a.device
                ).view(torch.float8_e4m3fn)
                padding_scale2 = torch.zeros(
                    (140, 48), dtype=torch.bfloat16, device=wq_a.device
                ).view(torch.float8_e4m3fn)
                padding_scale3 = torch.zeros(
                    (140, 56), dtype=torch.bfloat16, device=wq_a.device
                ).view(torch.float8_e4m3fn)
                wqkvia_and_scales = torch.cat(
                    [
                        wqkvia_0,
                        wqkvia_0_scale,
                        padding_scale0,
                        wqkvia_1,
                        wqkvia_1_scale,
                        padding_scale1,
                        wqkvia_2,
                        wqkvia_2_scale,
                        padding_scale2,
                        wqkvia_3,
                        wqkvia_3_scale,
                        padding_scale3,
                    ],
                    dim=1,
                )
            elif arch_name == "glm_5":  # GLM5
                # Ensure the scales are in float32 for DS v3.2
                if wq_a_scale.dtype != torch.float32:
                    # TODO: remove this after the source weights are converted to float32
                    print(
                        "Warning: RMSNormProjxWqkviaWeightsConverter: "
                        + "wq_a_scale is not in float32, converting to float32."
                    )
                wq_a_scale = wq_a_scale.to(torch.float32)
                wkv_a_scale = wkv_a_scale.to(torch.float32)
                wk_scale = wk_scale.to(torch.float32)
                # (2048 + 576 + 128, 6144)
                wqkvia = torch.cat([wq_a, wkv_a, wk], dim=0).reshape(86, 32, 6144)
                # (16+5+1， 48)
                wq_a_scale = wq_a_scale.reshape((16, 1, 48)).repeat(1, 4, 1).reshape(64, 48)
                wkv_a_scale = wkv_a_scale.reshape((5, 1, 48)).repeat(1, 4, 1).reshape(20, 48)[:18]
                wk_scale = wk_scale.reshape((1, 1, 48)).repeat(1, 4, 1).reshape(4, 48)
                wqkvia_scales = torch.cat([wq_a_scale, wkv_a_scale, wk_scale], dim=0)  # (86, 48)
                wqkvia = wqkvia.reshape(86, 32, 6, 1024).transpose(1, 2).reshape(86, 6, 2, 16, 1024)
                wqkvia = wqkvia.reshape(86, 6, 2, 16, 32, 32).transpose(3, 4)
                wqkvia = self._swizzle_qmma_16x32(wqkvia).reshape(86, 6, 32 * 1024)
                wqkvia_scales = wqkvia_scales.reshape(86, 6, 8).view(torch.float8_e4m3fn)
                wqkvia_padding = torch.zeros(
                    (86, 6, 128 - wqkvia_scales.shape[-1]),
                    dtype=torch.float8_e4m3fn,
                    device=wq_a.device,
                )
                wqkvia_and_scales = torch.cat([wqkvia, wqkvia_scales, wqkvia_padding], dim=-1)
            else:
                raise ValueError(f"Unsupported architecture: {arch_name}")
        assert wqkvia_and_scales is not None
        return x_rmsnorm_gamma.float(), wqkvia_and_scales.contiguous()


@dataclass
class RMSNormProjxWqkviaRefWeightsAlias:
    """Reference weights alias for RMSNormProjxWqkvia."""

    x_rmsnorm_gamma = "input_layernorm.weight"
    q_a_weights = "self_attn.q_a_proj.weight"
    q_a_scales = "self_attn.q_a_proj.weight_scale_inv"
    kv_a_with_mqa_weights = "self_attn.kv_a_proj_with_mqa.weight"
    kv_a_with_mqa_scales = "self_attn.kv_a_proj_with_mqa.weight_scale_inv"
    wk_weights = "self_attn.indexer.wk.weight"
    wk_scales = "self_attn.indexer.wk.weight_scale_inv"

    @property
    def ref_tensor_alias(self) -> list[str]:
        return [
            self.x_rmsnorm_gamma,
            self.q_a_weights,
            self.q_a_scales,
            self.kv_a_with_mqa_weights,
            self.kv_a_with_mqa_scales,
            self.wk_weights,
            self.wk_scales,
        ]

    def __call__(self) -> list[str]:
        return self.ref_tensor_alias


@dataclass
class RMSNormProjxWqkviaTilertWeightsAlias:
    """TileRT weights alias for RMSNormProjxWqkvia."""

    x_rmsnorm_gamma = "x_rmsnorm_gamma"
    q_a_weights = "q_a_weights"
    q_a_scales = "q_a_scales"
    kv_a_with_mqa_weights = "kv_a_with_mqa_weights"
    kv_a_with_mqa_scales = "kv_a_with_mqa_scales"
    wk_weights = "wk_weights"
    wk_scales = "wk_scales"

    @property
    def tilert_tensor_alias(self) -> list[str]:
        return [
            self.x_rmsnorm_gamma,
            self.q_a_weights,
            self.q_a_scales,
            self.kv_a_with_mqa_weights,
            self.kv_a_with_mqa_scales,
            self.wk_weights,
            self.wk_scales,
        ]

    def __call__(self) -> list[str]:
        return self.tilert_tensor_alias


class RMSNormProjxWqkvia(TileRTModule):
    """RMSNormProjxWqkvia module"""

    def __init__(
        self,
        model_args: ModelArgs,
        num_devices: int,
        device_id: int,
        ref_weights_alias: RMSNormProjxWqkviaRefWeightsAlias | None = None,
        algorithm: RMSNormProjxWqkviaAlgorithm = RMSNormProjxWqkviaAlgorithm.GENERAL,
    ):
        super().__init__(
            self.__class__.__name__,
            model_args=model_args,
            num_devices=num_devices,
            device_id=device_id,
        )

        self.tilert_weights_alias = RMSNormProjxWqkviaTilertWeightsAlias()
        self.ref_weights_alias = (
            ref_weights_alias
            if ref_weights_alias is not None
            else RMSNormProjxWqkviaRefWeightsAlias()
        )

        self.arch_name = self.model_args.arch_name
        self.dim = self.model_args.dim
        self.q_lora_rank = self.model_args.q_lora_rank
        self.kv_lora_rank = self.model_args.kv_lora_rank
        self.qk_rope_head_dim = self.model_args.qk_rope_head_dim
        self.idx_head_dim = self.model_args.index_head_dim
        self.block_size = self.model_args.block_size
        self.eps = self.model_args.eps
        self.algorithm: RMSNormProjxWqkviaAlgorithm = algorithm

        # reference weights
        self.ref_norm_gamma: torch.Tensor | None = None
        self.ref_wq_a: torch.Tensor | None = None
        self.ref_wkv_a: torch.Tensor | None = None
        self.ref_wk: torch.Tensor | None = None

        # tilert weights
        self.tilert_norm_gamma: torch.Tensor | None = None
        self.tilert_wqkv_a: torch.Tensor | None = None
        # Legacy scale tensor for compatibility, to be removed in the future
        self.tilert_wqkv_a_scales = torch.zeros((130, 64), dtype=torch.bfloat16)

        # tilert vars
        self.x_rmsnorm_out: torch.Tensor | None = None
        self.q_out: torch.Tensor | None = None
        self.kv_out: torch.Tensor | None = None
        self.ki_out: torch.Tensor | None = None
        self.x_rmsnorm_quant_out: torch.Tensor | None = None
        self.x_rmsnorm_quant_scale_out: torch.Tensor | None = None

        self.profile_logs: torch.Tensor | None = None
        self.is_init = False

        # tilert_funcs
        self.rmsnorm_proj_func: Callable | None = None
        self.rmsnorm_func: Callable | None = None
        self.proj_func: Callable | None = None

        if self.arch_name == "deepseek_v3_2":
            self.rmsnorm_proj_func = rmsnorm_projx_wqkvia
            self.rmsnorm_func = rmsnorm_quant
            self.proj_func = projx_wqkvia
        elif self.arch_name == "glm_5":
            # Lazy import to avoid circular import
            self.rmsnorm_proj_func = None
            self.rmsnorm_func = rmsnorm_quant
            self.proj_func = projx_wqkvia
        else:
            raise ValueError(f"Unsupported architecture: {self.arch_name}")

        # tilert tensor aliases (3 output weight names for get_weights_list)
        self.tilert_tensor_alias: list[str] = [
            "x_rmsnorm_gamma",
            "qkv_wa_weights",
            "qkv_wa_scales",
        ]

    def get_weights_list(self) -> list[torch.Tensor]:
        """
        Get the weights list.

        Returns:
            List of weights.
        """
        assert self.algorithm is not None, "Algorithm is not set"
        if self.algorithm == RMSNormProjxWqkviaAlgorithm.GENERAL:
            return [self.tilert_norm_gamma, self.tilert_wqkv_a, self.tilert_wqkv_a_scales]
        return [self.tilert_norm_gamma, self.tilert_wqkv_a]

    def device_sharding(self, weights_map: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Device sharding.

        Args:
            input_layernorm_weight: Input layernorm weight.
            q_a_proj_weight: Q A proj weight.
            q_a_proj_weight_scale: Q A proj weight scale.
            kv_a_proj_weight: KV A proj weight.
            kv_a_proj_weight_scale: KV A proj weight scale.
            indexer_wk_weight: Indexer WK weight.
            indexer_wk_weight_scale: Indexer WK weight scale.

        Returns:
            Tuple of weights.
        """
        # repeat n times for device sharding
        # Using float to support both bfloat16 and float
        input_layernorm_weight = (
            weights_map[self.ref_weights_alias.x_rmsnorm_gamma][None, ...]
            .float()
            .repeat(self.num_devices, 1)
        )
        q_a_proj_weight = weights_map[self.ref_weights_alias.q_a_weights][None, ...].repeat(
            self.num_devices, 1, 1
        )
        q_a_proj_weight_scale = weights_map[self.ref_weights_alias.q_a_scales][None, ...].repeat(
            self.num_devices, 1, 1
        )
        kv_a_proj_weight = weights_map[self.ref_weights_alias.kv_a_with_mqa_weights][
            None, ...
        ].repeat(self.num_devices, 1, 1)
        kv_a_proj_weight_scale = weights_map[self.ref_weights_alias.kv_a_with_mqa_scales][
            None, ...
        ].repeat(self.num_devices, 1, 1)
        indexer_wk_weight = weights_map[self.ref_weights_alias.wk_weights][None, ...].repeat(
            self.num_devices, 1, 1
        )
        indexer_wk_weight_scale = weights_map[self.ref_weights_alias.wk_scales][None, ...].repeat(
            self.num_devices, 1, 1
        )
        return {
            self.tilert_weights_alias.x_rmsnorm_gamma: input_layernorm_weight,
            self.tilert_weights_alias.q_a_weights: q_a_proj_weight,
            self.tilert_weights_alias.q_a_scales: q_a_proj_weight_scale,
            self.tilert_weights_alias.kv_a_with_mqa_weights: kv_a_proj_weight,
            self.tilert_weights_alias.kv_a_with_mqa_scales: kv_a_proj_weight_scale,
            self.tilert_weights_alias.wk_weights: indexer_wk_weight,
            self.tilert_weights_alias.wk_scales: indexer_wk_weight_scale,
        }

    def init_reference_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        """
        Initialize the reference weights.

        Args:
            state_dict: State dictionary.
        """
        self.ref_norm_gamma = state_dict[self.ref_weights_alias()[0]]
        self.ref_wq_a = weight_dequant(
            state_dict[self.ref_weights_alias()[1]], state_dict[self.ref_weights_alias()[2]]
        )
        self.ref_wkv_a = weight_dequant(
            state_dict[self.ref_weights_alias()[3]], state_dict[self.ref_weights_alias()[4]]
        )
        self.ref_wk = weight_dequant(
            state_dict[self.ref_weights_alias()[5]], state_dict[self.ref_weights_alias()[6]]
        )

        assert self.ref_norm_gamma is not None
        assert self.ref_wq_a is not None
        assert self.ref_wkv_a is not None
        assert self.ref_wk is not None

        assert (
            self.ref_norm_gamma.shape[-1] == self.dim
        ), f"norm_gamma shape must be {self.dim}, but got {self.ref_norm_gamma.shape[-1]}"
        assert self.ref_wq_a.shape[-2] == self.q_lora_rank, (
            f"wq_a shape must be {self.q_lora_rank}, " + f"but got {self.ref_wq_a.shape[-2]}"
        )
        assert (
            self.ref_wq_a.shape[-1] == self.dim
        ), f"wq_a shape must be {self.dim}, but got {self.ref_wq_a.shape[-1]}"
        assert self.ref_wkv_a.shape[-2] == self.kv_lora_rank + self.qk_rope_head_dim, (
            f"wkv_a shape must be {self.kv_lora_rank + self.qk_rope_head_dim}, "
            + f"but got {self.ref_wkv_a.shape[-2]}"
        )
        assert (
            self.ref_wkv_a.shape[-1] == self.dim
        ), f"wkv_a shape must be {self.dim}, but got {self.ref_wkv_a.shape[-1]}"
        assert (
            self.ref_wk.shape[-2] == self.idx_head_dim
        ), f"wk shape must be {self.idx_head_dim}, but got {self.ref_wk.shape[-2]}"
        assert (
            self.ref_wk.shape[-1] == self.dim
        ), f"wk shape must be {self.dim}, but got {self.ref_wk.shape[-1]}"

    def init_tilert_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        """
        Initialize the tilert weights.

        Args:
            state_dict: State dictionary.
        """
        assert self.algorithm is not None, "Algorithm is not set"
        self.tilert_norm_gamma, self.tilert_wqkv_a = RMSNormProjxWqkviaWeightsConverter(
            self.model_args, self.num_devices
        ).dispatch(self.algorithm, [state_dict[alias] for alias in self.tilert_weights_alias()])

    def init_tilert_vars(self, batch_size: int, seq_len: int) -> None:
        """
        Initialize the tilert variables.

        Args:
            batch_size: Batch size.
            seq_len: Sequence length.
        """
        self.q_out = torch.zeros((batch_size, seq_len, self.q_lora_rank), dtype=torch.bfloat16)
        self.kv_out = torch.zeros((batch_size, seq_len, self.kv_lora_rank), dtype=torch.bfloat16)
        self.ki_out = torch.zeros((batch_size, seq_len, self.idx_head_dim), dtype=torch.bfloat16)
        self.x_rmsnorm_out = torch.zeros((batch_size, seq_len, self.dim), dtype=torch.bfloat16)
        if self.algorithm == RMSNormProjxWqkviaAlgorithm.DECOUPLED:
            self.x_rmsnorm_quant_out = torch.zeros(
                (batch_size, seq_len, self.dim), dtype=torch.float8_e4m3fn
            )
            self.x_rmsnorm_quant_scale_out = torch.zeros(
                (batch_size, seq_len, self.dim // self.block_size), dtype=torch.float32
            )
        self.profile_logs = get_profile_log_tensor()
        self.is_init = True

    def init_random_weights(self) -> None:
        """
        Initialize the random weights.

        Returns:
            None
        """
        q_scale_dim = self.q_lora_rank // self.block_size
        kv_scale_dim = (self.kv_lora_rank + self.qk_rope_head_dim) // self.block_size + 1
        wk_scale_dim = self.idx_head_dim // self.block_size
        dim_scale_dim = self.dim // self.block_size
        scale_dtype = torch.float32 if self.arch_name == "glm_5" else torch.bfloat16

        tensor_list = [
            torch.randn(self.dim, dtype=torch.float32),
            torch.randn(self.q_lora_rank, self.dim, dtype=torch.bfloat16).to(torch.float8_e4m3fn),
            torch.randn(q_scale_dim, dim_scale_dim, dtype=scale_dtype),
            torch.randn(
                self.kv_lora_rank + self.qk_rope_head_dim, self.dim, dtype=torch.bfloat16
            ).to(torch.float8_e4m3fn),
            torch.randn(kv_scale_dim, dim_scale_dim, dtype=scale_dtype),
            torch.randn(self.idx_head_dim, self.dim, dtype=torch.bfloat16).to(torch.float8_e4m3fn),
            torch.randn(wk_scale_dim, dim_scale_dim, dtype=scale_dtype),
        ]
        ref_state_dict = dict(zip(self.ref_weights_alias(), tensor_list))
        self.init_reference_weights(ref_state_dict)
        self.init_tilert_weights(
            {_k: _v[self.device_id] for _k, _v in self.device_sharding(ref_state_dict).items()}
        )

    def golden_forward(
        self,
        x: torch.Tensor,
        pe_cache: torch.Tensor,
        start_pos: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        assert self.ref_norm_gamma is not None
        assert self.ref_wq_a is not None
        assert self.ref_wkv_a is not None
        assert self.ref_wk is not None

        x_rmsnorm_out = torch.nn.functional.rms_norm(
            x.float(), [x.size(-1)], self.ref_norm_gamma, self.eps
        )

        q_out = torch.matmul(x_rmsnorm_out.float(), self.ref_wq_a.transpose(0, 1).float())
        kv_out = torch.matmul(x_rmsnorm_out.float(), self.ref_wkv_a.transpose(0, 1).float())
        kv_out, k_pe = torch.split(kv_out, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        bsz = k_pe.shape[0]
        seq_len = k_pe.shape[1]
        pe_cache[:bsz, start_pos : start_pos + seq_len].copy_(k_pe.to(torch.bfloat16))
        ki_out = torch.matmul(x_rmsnorm_out.float(), self.ref_wk.transpose(0, 1).float())
        return (
            x_rmsnorm_out.to(torch.bfloat16),
            q_out.to(torch.bfloat16),
            kv_out.to(torch.bfloat16),
            ki_out.to(torch.bfloat16),
        )

    def tilert_forward(
        self,
        x: torch.Tensor,
        pe_cache: torch.Tensor,
        start_pos: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.algorithm == RMSNormProjxWqkviaAlgorithm.GENERAL:
            assert self.rmsnorm_proj_func is not None
            self.rmsnorm_proj_func(
                x.to(torch.bfloat16),
                self.tilert_wqkv_a,
                self.tilert_wqkv_a_scales,
                self.tilert_norm_gamma,
                torch.tensor([start_pos], dtype=torch.int32, device=x.device),
                self.q_out,
                self.kv_out,
                pe_cache,
                self.ki_out,
                self.x_rmsnorm_out,
                self.profile_logs,
            )
        elif self.algorithm == RMSNormProjxWqkviaAlgorithm.DECOUPLED:
            assert self.rmsnorm_func is not None
            assert self.proj_func is not None
            self.rmsnorm_func(
                x.to(torch.bfloat16),
                self.tilert_norm_gamma,
                self.x_rmsnorm_out,
                self.x_rmsnorm_quant_out,
                self.x_rmsnorm_quant_scale_out,
                self.profile_logs,
            )
            self.proj_func(
                self.x_rmsnorm_quant_out,
                self.x_rmsnorm_quant_scale_out,
                self.tilert_wqkv_a,
                torch.tensor([start_pos], dtype=torch.int32, device=x.device),
                self.q_out,
                self.kv_out,
                pe_cache,
                self.ki_out,
                self.profile_logs,
            )
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")

        if self.flag_enable_profiling_log:
            parse_profile_log_tensor(
                self.profile_logs, self.get_profile_log_path(), [(self.op_name, 0.0)]
            )
        return self.x_rmsnorm_out, self.q_out, self.kv_out, self.ki_out

    def __call__(
        self,
        x: torch.Tensor,
        pe_cache: torch.Tensor,
        start_pos: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.golden_forward(x, pe_cache, start_pos)
