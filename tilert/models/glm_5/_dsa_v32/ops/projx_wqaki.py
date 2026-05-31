"""ProjxWqaki operation module."""

import torch

__all__ = [
    "projx_wqaki",
    "ProjxWqakiWeightsConverter",
]


def projx_wqaki(
    x_quant: torch.Tensor,
    x_scale: torch.Tensor,
    wqaki: torch.Tensor,
    out_q: torch.Tensor,
    out_ki: torch.Tensor,
    profile_logs: torch.Tensor,
    compute_kernel_type: str = "fp8mma",
    *,
    model_arch: str,
) -> None:
    """FP8 projection for q, ki.

    Args:
        x_quant: FP8 quantized hidden states [1, seq_len, hidden_dim].
        x_scale: Scale factors for x_quant.
        wqaki: Packed FP8 weights + scales for q, ki.
        out_q: Output q tensor.
        out_ki: Output ki tensor.
        profile_logs: Profile logs tensor.
        compute_kernel_type: Kernel type ("fp8mma", "fp8mma_68cta", "fp8mma_136cta").
        model_arch: Model architecture ("deepseek_v3_2" or "glm_5").
    """
    torch.ops.tilert.projx_wqaki_op(
        x_quant,
        x_scale,
        wqaki,
        out_q,
        out_ki,
        model_arch,
        compute_kernel_type,
        profile_logs,
        torch.empty(0, dtype=torch.int64, device=x_quant.device),
    )


class ProjxWqakiWeightsConverter:
    """Weight converter for ProjxWqaki kernel."""

    @staticmethod
    def _swizzle_qmma_16x32(mat_in: torch.Tensor) -> torch.Tensor:
        assert mat_in.shape[-2] == 16 and mat_in.shape[-1] == 32
        assert mat_in.dtype == torch.float8_e4m3fn
        pre_shape = mat_in.shape[:-2]
        mat_in = mat_in.reshape(*pre_shape, 2, 8, 2, 4, 4).transpose(-4, -3).transpose(-5, -4)
        return mat_in.reshape(*pre_shape, 2 * 2, 8 * 4, 4).transpose(-3, -2)

    @staticmethod
    def convert_dsv32(
        wq_a: torch.Tensor,
        wq_a_scale: torch.Tensor,
        wki: torch.Tensor,
        wki_scale: torch.Tensor,
    ) -> torch.Tensor:
        """Convert DSV3.2 weights to the packed format expected by the kernel."""
        with torch.inference_mode():
            wq_a_scale = wq_a_scale.to(torch.bfloat16)
            wki_scale = wki_scale.to(torch.bfloat16)

            dim = 7168
            q_rows = 1536
            ki_rows = 128
            total_rows = q_rows + ki_rows
            n_blocks = total_rows // 16
            scale_dim = dim // 128

            n_q_blocks = q_rows // 16
            n_ki_blocks = ki_rows // 16
            wq_a = wq_a.reshape(n_q_blocks, 16, dim)
            wq_a_scale = (
                wq_a_scale.reshape(wq_a_scale.shape[0], 1, scale_dim)
                .repeat(1, n_q_blocks // wq_a_scale.shape[0], 1)
                .reshape(n_q_blocks, scale_dim)
            )
            wki = wki.reshape(n_ki_blocks, 16, dim)
            wki_scale = (
                wki_scale.reshape(wki_scale.shape[0], 1, scale_dim)
                .repeat(1, n_ki_blocks // wki_scale.shape[0], 1)
                .reshape(n_ki_blocks, scale_dim)
            )

            wqaki = torch.cat([wq_a, wki], dim=0)
            wqaki_scale = torch.cat([wq_a_scale, wki_scale], dim=0)

            swizzle = ProjxWqakiWeightsConverter._swizzle_qmma_16x32

            wqaki_0 = wqaki[..., :2048]
            wqaki_0_scale = wqaki_scale[..., :16].contiguous().view(torch.float8_e4m3fn)
            wqaki_1 = wqaki[..., 2048:4096]
            wqaki_1_scale = wqaki_scale[..., 16:32].contiguous().view(torch.float8_e4m3fn)
            wqaki_2 = wqaki[..., 4096:6144]
            wqaki_2_scale = wqaki_scale[..., 32:48].contiguous().view(torch.float8_e4m3fn)
            wqaki_3 = wqaki[..., 6144:7168]
            wqaki_3_scale = wqaki_scale[..., 48:56].contiguous().view(torch.float8_e4m3fn)

            wqaki_0 = wqaki_0.reshape(n_blocks, 16, 64, 32).transpose(1, 2)
            wqaki_0 = swizzle(wqaki_0).reshape(n_blocks, 16 * 2048)

            wqaki_1 = wqaki_1.reshape(n_blocks, 16, 64, 32).transpose(1, 2)
            wqaki_1 = swizzle(wqaki_1).reshape(n_blocks, 16 * 2048)

            wqaki_2 = wqaki_2.reshape(n_blocks, 16, 64, 32).transpose(1, 2)
            wqaki_2 = swizzle(wqaki_2).reshape(n_blocks, 16 * 2048)

            wqaki_3 = wqaki_3.reshape(n_blocks, 16, 32, 32).transpose(1, 2)
            wqaki_3 = swizzle(wqaki_3).reshape(n_blocks, 16 * 1024)

            padding_scale0 = torch.zeros(
                (n_blocks, 48), dtype=torch.bfloat16, device=wq_a.device
            ).view(torch.float8_e4m3fn)
            padding_scale1 = torch.zeros(
                (n_blocks, 48), dtype=torch.bfloat16, device=wq_a.device
            ).view(torch.float8_e4m3fn)
            padding_scale2 = torch.zeros(
                (n_blocks, 48), dtype=torch.bfloat16, device=wq_a.device
            ).view(torch.float8_e4m3fn)
            padding_scale3 = torch.zeros(
                (n_blocks, 56), dtype=torch.bfloat16, device=wq_a.device
            ).view(torch.float8_e4m3fn)

            return torch.cat(
                [
                    wqaki_0,
                    wqaki_0_scale,
                    padding_scale0,
                    wqaki_1,
                    wqaki_1_scale,
                    padding_scale1,
                    wqaki_2,
                    wqaki_2_scale,
                    padding_scale2,
                    wqaki_3,
                    wqaki_3_scale,
                    padding_scale3,
                ],
                dim=1,
            ).contiguous()

    @staticmethod
    def convert_glm5_68cta(
        wq_a: torch.Tensor,
        wq_a_scale: torch.Tensor,
        wki: torch.Tensor,
        wki_scale: torch.Tensor,
    ) -> torch.Tensor:
        """Convert GLM5 weights to the packed format expected by the kernel."""
        with torch.inference_mode():
            wq_a_scale = wq_a_scale.to(torch.float32)
            wki_scale = wki_scale.to(torch.float32)

            dim = 6144
            q_rows = 2048
            ki_rows = 128
            total_rows = q_rows + ki_rows
            n_blocks = total_rows // 32
            scale_dim = dim // 128

            n_q_blocks = q_rows // 32
            n_ki_blocks = ki_rows // 32

            wqaki_raw = torch.cat([wq_a, wki], dim=0).reshape(n_blocks, 32, dim)

            wq_a_scale = (
                wq_a_scale.reshape(wq_a_scale.shape[0], 1, scale_dim)
                .repeat(1, n_q_blocks // wq_a_scale.shape[0], 1)
                .reshape(n_q_blocks, scale_dim)
            )
            wki_scale = (
                wki_scale.reshape(wki_scale.shape[0], 1, scale_dim)
                .repeat(1, n_ki_blocks // wki_scale.shape[0], 1)
                .reshape(n_ki_blocks, scale_dim)
            )
            wqaki_scales = torch.cat([wq_a_scale, wki_scale], dim=0)

            swizzle = ProjxWqakiWeightsConverter._swizzle_qmma_16x32

            wqaki_raw = wqaki_raw.reshape(n_blocks, 32, 6, 1024).transpose(1, 2)
            wqaki_raw = wqaki_raw.reshape(n_blocks, 6, 2, 16, 32, 32).transpose(3, 4)
            wqaki_raw = swizzle(wqaki_raw).reshape(n_blocks, 6, 32 * 1024)
            wqaki_scales = wqaki_scales.reshape(n_blocks, 6, 8).view(torch.float8_e4m3fn)
            wqaki_padding = torch.zeros(
                (n_blocks, 6, 128 - wqaki_scales.shape[-1]),
                dtype=torch.float8_e4m3fn,
                device=wq_a.device,
            )
            return torch.cat([wqaki_raw, wqaki_scales, wqaki_padding], dim=-1).contiguous()

    @staticmethod
    def convert_glm5_136cta(
        wq_a: torch.Tensor,
        wq_a_scale: torch.Tensor,
        wki: torch.Tensor,
        wki_scale: torch.Tensor,
    ) -> torch.Tensor:
        """Convert GLM5 weights to the packed format expected by the kernel."""
        with torch.inference_mode():
            wq_a_scale = wq_a_scale.to(torch.float32)
            wki_scale = wki_scale.to(torch.float32)

            dim = 6144
            q_rows = 2048
            ki_rows = 128
            total_rows = q_rows + ki_rows
            n_blocks = total_rows // 16
            scale_dim = dim // 128

            n_q_blocks = q_rows // 16
            n_ki_blocks = ki_rows // 16

            wq_a = wq_a.reshape(n_q_blocks, 16, dim)
            wq_a_scale = (
                wq_a_scale.reshape(wq_a_scale.shape[0], 1, scale_dim)
                .repeat(1, n_q_blocks // wq_a_scale.shape[0], 1)
                .reshape(n_q_blocks, scale_dim)
            )
            wki = wki.reshape(n_ki_blocks, 16, dim)
            wki_scale = (
                wki_scale.reshape(wki_scale.shape[0], 1, scale_dim)
                .repeat(1, n_ki_blocks // wki_scale.shape[0], 1)
                .reshape(n_ki_blocks, scale_dim)
            )

            wqaki_raw = torch.cat([wq_a, wki], dim=0)
            wqaki_scales = torch.cat([wq_a_scale, wki_scale], dim=0)

            swizzle = ProjxWqakiWeightsConverter._swizzle_qmma_16x32

            wqaki_raw = wqaki_raw.reshape(n_blocks, 16, 3, 2048).transpose(1, 2)
            wqaki_raw = wqaki_raw.reshape(n_blocks, 3, 1, 16, 64, 32).transpose(3, 4)
            wqaki_raw = swizzle(wqaki_raw).reshape(n_blocks, 3, 16 * 2048)
            wqaki_scales = wqaki_scales.reshape(n_blocks, 3, 16).view(torch.float8_e4m3fn)
            wqaki_padding = torch.zeros(
                (n_blocks, 3, 128 - wqaki_scales.shape[-1]),
                dtype=torch.float8_e4m3fn,
                device=wq_a.device,
            )
            return torch.cat([wqaki_raw, wqaki_scales, wqaki_padding], dim=-1).contiguous()
