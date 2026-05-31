"""RMSNormQuant operation module."""

from __future__ import annotations

import torch

__all__ = [
    "BLOCK_SIZE",
    "DIM_DEEPSEEK_V3_2",
    "DIM_GLM_5",
    "rmsnorm_quant",
]

BLOCK_SIZE = 128
DIM_DEEPSEEK_V3_2 = 7168
DIM_GLM_5 = 6144


def rmsnorm_quant(
    hidden_in: torch.Tensor,
    gamma_in: torch.Tensor,
    hidden_out: torch.Tensor,
    quant_hidden_out: torch.Tensor | None = None,
    quant_hidden_scale_out: torch.Tensor | None = None,
    profile_logs: torch.Tensor | None = None,
    compute_kernel_type: str = "general",
    *,
    model_arch: str,
) -> None:
    """
    Rmsnorm with optional activation quantization.

    Args:
        hidden_in: Input tensor (..., dim).
        gamma_in: RMSNorm gamma (dim,).
        hidden_out: RMSNorm output (..., dim).
        quant_hidden_out: Optional quantized output (..., dim). If None, no quant.
        quant_hidden_scale_out: Optional quant scale (..., dim // block_size). If None, no quant.
        profile_logs: Optional profile logs tensor.
    """
    if profile_logs is None:
        raise ValueError("profile_logs is required when calling rmsnorm_quant.")

    if quant_hidden_out is None or quant_hidden_scale_out is None:
        torch.ops.tilert.rmsnorm_op(
            hidden_in,
            gamma_in,
            hidden_out,
            model_arch,
            compute_kernel_type,
            profile_logs,
        )
    else:
        torch.ops.tilert.rmsnorm_quant_op(
            hidden_in,
            gamma_in,
            hidden_out,
            quant_hidden_out,
            quant_hidden_scale_out,
            model_arch,
            compute_kernel_type,
            profile_logs,
            torch.empty(0, dtype=torch.int64, device=hidden_in.device),
        )
