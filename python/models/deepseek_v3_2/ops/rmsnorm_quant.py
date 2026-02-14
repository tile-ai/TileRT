"""RMSNormQuant operation module.

Unified for deepseek_v3_2 (dim=7168) and glm_5 (dim=6144).
Dispatches by hidden_in.shape[-1]: 7168 -> rmsnorm_*_op, 6144 -> rmsnorm_*_glm5_op.
"""

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
) -> None:
    """
    Rmsnorm with optional activation quantization.

    Unified for deepseek_v3_2 (dim=7168) and glm_5 (dim=6144). Dispatches by
    hidden_in.shape[-1]: 7168 -> rmsnorm_op / rmsnorm_quant_op,
    6144 -> rmsnorm_glm5_op / rmsnorm_quant_glm5_op.

    Args:
        hidden_in: Input tensor (..., dim).
        gamma_in: RMSNorm gamma (dim,).
        hidden_out: RMSNorm output (..., dim).
        quant_hidden_out: Optional quantized output (..., dim). If None, no quant.
        quant_hidden_scale_out: Optional quant scale (..., dim // block_size). If None, no quant.
        profile_logs: Optional profile logs tensor.
    """
    dim = hidden_in.shape[-1]
    if dim == DIM_GLM_5:
        glm5_flag = "_glm5"
    elif dim == DIM_DEEPSEEK_V3_2:
        glm5_flag = ""
    else:
        raise ValueError(
            f"Unsupported hidden_in.shape[-1]: {dim}. "
            f"rmsnorm_quant supports {DIM_DEEPSEEK_V3_2} (deepseek_v3_2) or {DIM_GLM_5} (glm_5)."
        )
    if quant_hidden_out is None or quant_hidden_scale_out is None:
        quant_flag = ""
        quant_args = [hidden_in, gamma_in, hidden_out, profile_logs]
    else:
        quant_flag = "_quant"
        quant_args = [
            hidden_in,
            gamma_in,
            hidden_out,
            quant_hidden_out,
            quant_hidden_scale_out,
            profile_logs,
        ]
    if profile_logs is None:
        raise ValueError("profile_logs is required when calling rmsnorm_quant.")
    func_name = f"rmsnorm{quant_flag}{glm5_flag}_op"
    func_call = getattr(torch.ops.tilert, func_name)
    func_call(*quant_args)
