"""GLM-5.2 MoE up/gate + SiLU, W8A8: the python twins of the device kernel."""

import torch

from tilert.models.glm_5_2_rocm.ops.upgate_silu import pair_interleave_rows

SEG_K = 128


def swizzle_pair_interleaved_k128(w_fp8: torch.Tensor, inter: int) -> torch.Tensor:
    rows, k = w_fp8.shape
    assert rows == 2 * inter and k % SEG_K == 0
    w8 = w_fp8.view(torch.uint8)
    perm = pair_interleave_rows(inter)
    t = torch.arange(rows // 16)
    c = torch.arange(k // SEG_K)
    h = torch.arange(2)
    lane = torch.arange(64)
    i = torch.arange(16)
    T, C, H, L, II = torch.meshgrid(t, c, h, lane, i, indexing="ij")
    rows_ix = perm[T * 16 + L % 16]
    ks = C * SEG_K + L // 16 * 32 + H * 16 + II
    return w8[rows_ix, ks].reshape(-1).contiguous()


FP8_MAX = 448.0
FP8_AMAX_EPS = 0.0001


def quant_std_blocks(x_bf16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    shape = x_bf16.shape
    xb = x_bf16.float().reshape(*shape[:-1], shape[-1] // SEG_K, SEG_K)
    amax = xb.abs().amax(dim=-1)
    scale = amax.clamp_min(FP8_AMAX_EPS) * torch.tensor(
        1.0 / FP8_MAX, dtype=torch.float32, device=xb.device
    )
    inv = torch.ones_like(scale) / scale
    q = (xb * inv.unsqueeze(-1)).to(torch.float8_e4m3fn).reshape(shape)
    return (q, scale)


def quant_act_row(norm_bf16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert norm_bf16.dtype == torch.bfloat16
    return quant_std_blocks(norm_bf16)


def quant_mid_rows(mid_bf16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert mid_bf16.dtype == torch.bfloat16
    return quant_std_blocks(mid_bf16)


__all__ = [
    "FP8_AMAX_EPS",
    "FP8_MAX",
    "SEG_K",
    "quant_act_row",
    "quant_mid_rows",
    "quant_std_blocks",
    "swizzle_pair_interleaved_k128",
]
