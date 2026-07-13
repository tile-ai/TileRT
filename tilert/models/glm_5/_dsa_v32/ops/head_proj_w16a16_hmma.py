"""HeadProj BF16-MMA operation for DeepSeek-V3.2 / GLM5."""

from __future__ import annotations

import torch

__all__ = [
    "head_proj_w16a16_hmma",
    "swizzle_head_proj_weight_bf16mma",
]


def head_proj_w16a16_hmma(
    hidden_in: torch.Tensor,
    weight_in: torch.Tensor,
    logits_out: torch.Tensor,
    profile_logs: torch.Tensor,
    model_arch: str,
    compute_kernel_type: str = "w16a16_hmma",
) -> None:
    torch.ops.tilert.head_proj_op(
        hidden_in,
        weight_in,
        logits_out,
        model_arch,
        compute_kernel_type,
        profile_logs,
    )


def _swizzle_mma_16x16(mat_in: torch.Tensor) -> torch.Tensor:
    assert mat_in.shape[-2] == 16 and mat_in.shape[-1] == 16
    pre = mat_in.shape[:-2]
    x = mat_in.reshape(*pre, 2, 8, 2, 4, 2).transpose(-4, -3).transpose(-5, -4)
    return x.reshape(*pre, 2 * 2, 8 * 4, 2).transpose(-3, -2)


def swizzle_head_proj_weight_bf16mma(weight: torch.Tensor) -> torch.Tensor:
    n, k = weight.shape
    assert n % 16 == 0 and k % 1024 == 0, "head_proj weight must be /16 in N and /1024 in K"
    n_tiles = n // 16
    k_pages = k // 1024
    k_inner = 1024 // 16
    w = weight.reshape(n_tiles, 16, k_pages, k_inner, 16)
    w = w.permute(0, 2, 3, 1, 4).contiguous()
    w = _swizzle_mma_16x16(w)
    return w.contiguous()
