"""Utility functions for tilert models."""

__all__ = [
    "precompute_freqs_cis",
    "apply_rotary_emb",
]

import math
from enum import IntEnum

import torch

_FACTOR_OVERRIDE_UNSET = object()
_THETA_OVERRIDE_UNSET = object()


def precompute_freqs_cis(  # type: ignore[no-untyped-def]
    args,
    *,
    factor_override=_FACTOR_OVERRIDE_UNSET,
    theta_override=_THETA_OVERRIDE_UNSET,
) -> torch.Tensor:
    """
    Pre-computes frequency-based complex exponential values for rotary positional embeddings.

    Args:
        args (ModelArgs): Model arguments containing positional embedding parameters.
        factor_override: If unset, ``args.rope_factor`` is used. Pass a
            numeric value to override the factor inline.
        theta_override: If unset, ``args.rope_theta`` is used. Pass a numeric
            value to override the rope base. ``None`` is rejected.

    Returns:
        torch.Tensor: Precomputed complex exponential values for positional embeddings.
    """
    dim = args.qk_rope_head_dim
    seqlen = args.max_seq_len
    beta_fast = args.beta_fast
    beta_slow = args.beta_slow
    base = args.rope_theta if theta_override is _THETA_OVERRIDE_UNSET else theta_override
    factor = args.rope_factor if factor_override is _FACTOR_OVERRIDE_UNSET else factor_override

    def find_correction_dim(num_rotations: float, dim: int, base: float, max_seq_len: int) -> float:
        """
        Find correction dimension.

        Computes the correction dimension for a given number of rotations in the rotary positional
        embedding.

        Args:
            num_rotations (float): Number of rotations to compute the correction for.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            float: The correction dimension based on the input parameters.
        """
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(
        low_rot: float,
        high_rot: float,
        dim: int,
        base: float,
        max_seq_len: int,
    ) -> tuple[int, int]:
        """
        Find correction range.

        Computes the range of correction dimensions for rotary positional
            embeddings.

        Args:
            low_rot (float): Lower bound for the number of rotations.
            high_rot (float): Upper bound for the number of rotations.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            Tuple[int, int]: The range of correction dimensions (low, high),
                clamped to valid indices.
        """
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min_value: float, max_value: float, dim: int) -> torch.Tensor:
        """
        Linear ramp function.

        Computes a linear ramp function used to smooth values between a minimum
            and maximum range.

        Args:
            min (float): Minimum value for the ramp function.
            max (float): Maximum value for the ramp function.
            dim (int): Dimensionality of the ramp tensor.

        Returns:
            torch.Tensor: A tensor of shape (dim,) with values linearly
                interpolated between 0 and 1, clamped to the range [0, 1].
        """
        if min_value == max_value:
            max_value += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min_value) / (max_value - min_value)
        return torch.clamp(linear_func, 0, 1)

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if factor is not None and seqlen > args.original_seq_len:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, args.original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t_index = torch.arange(seqlen)
    freqs = torch.outer(t_index, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(
    x_in: torch.Tensor, freqs_cis: torch.Tensor, interleaved: bool = True
) -> torch.Tensor:
    """Applies rotary positional embeddings to the input tensor.

    Args:
        x_in: Input tensor with positional embeddings to be applied.
        freqs_cis: Precomputed complex exponential values for positional embeddings.
        interleaved: If True (default), adjacent pairs (x0,x1),(x2,x3)... form
            complex numbers. If False, half-half layout: (x0,x_{d/2}),(x1,x_{d/2+1})...
            The DeepSeek-V3.2-Exp indexer uses interleaved=False.

    Returns:
        torch.Tensor: Tensor with rotary embeddings applied.
    """
    dtype = x_in.dtype
    shape = x_in.shape
    if not interleaved:
        x_in = x_in.view(*shape[:-1], 2, -1).transpose(-1, -2).contiguous()
    x_in = torch.view_as_complex(x_in.float().view(*shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x_in.size(1), 1, x_in.size(-1))
    y_out = torch.view_as_real(x_in * freqs_cis).flatten(3)
    if not interleaved:
        y_out = torch.cat([y_out[..., 0::2], y_out[..., 1::2]], dim=-1)
    return y_out.to(dtype)


class SwizzleMode(IntEnum):
    """Swizzle mode."""

    SWIZZLE_NONE = 0
    SWIZZLE_32B = 32 // 16
    SWIZZLE_64B = 64 // 16
    SWIZZLE_128B = 128 // 16


def gen_tensor_swizzle_map_1d(
    rows: int, cols_in_16bytes: int, swizzle_mode: SwizzleMode = SwizzleMode.SWIZZLE_128B
) -> torch.Tensor:
    """
    Generate flattened 1D swizzle map for given tensor dimensions.

    Args:
        rows (int): Number of rows in the tensor.
        cols_in_16bytes (int): Number of columns in the tensor, in 16-byte units.
        swizzle_mode (SwizzleMode): Swizzle mode to use. Default is SWIZZLE_128B.

    Returns:
        torch.Tensor: Flattened 1D swizzle map, in 16-byte units.
    """
    idxs = torch.arange(rows * cols_in_16bytes, dtype=torch.int32)
    if swizzle_mode == SwizzleMode.SWIZZLE_NONE:
        return idxs
    row_ids = idxs // cols_in_16bytes
    col_ids = idxs % cols_in_16bytes
    col_ids = (row_ids % swizzle_mode) ^ col_ids
    return row_ids * cols_in_16bytes + col_ids
