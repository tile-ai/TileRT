from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from tilert.models.deepseek_v3_2.refs.kernel import act_quant, fp8_gemm, weight_dequant

__all__ = [
    "act_quant",
    "fp8_gemm",
    "weight_dequant",
    "init_func",
    "linear",
    "RMSNorm",
]

from tilert.models.deepseek_config import (
    block_size,
    gemm_impl,
)

_LAZY_IMPORTS = {"act_quant", "fp8_gemm", "weight_dequant"}


def __getattr__(name: str) -> object:
    if name in _LAZY_IMPORTS:
        from tilert.models.deepseek_v3_2.refs import kernel

        attr = getattr(kernel, name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _get_scale_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Return the dynamically attached ``scale`` tensor."""
    scale = getattr(tensor, "scale", None)
    if scale is None:
        raise AttributeError("Expected quantized tensor to carry a 'scale' attribute.")
    return cast(torch.Tensor, scale)


def init_func(x_in: torch.Tensor) -> torch.Tensor:
    x_dtype = x_in.dtype
    x_fp32 = x_in.to(torch.float32)
    if x_fp32.dim() >= 2:
        initial_tensor = nn.init.kaiming_uniform_(x_fp32)
    else:
        initial_tensor = nn.init.uniform_(x_fp32)
    return initial_tensor.to(x_dtype)


def linear(
    x_in: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    scale_fmt: str | None = None,
) -> torch.Tensor:
    """
    Applies a linear transformation to the incoming data: y = xA^T + b.

    Args:
        x_in (torch.Tensor): The input tensor.
        weight (torch.Tensor): The weight tensor. It may be quantized.
        bias (Optional[torch.Tensor]): The bias tensor to be added. Default is None.

    Returns:
        torch.Tensor: The result of the linear transformation.
    """
    if weight.element_size() > 1:
        return F.linear(x_in, weight, bias)

    from tilert.models.deepseek_v3_2.refs.kernel import act_quant, fp8_gemm, weight_dequant

    if gemm_impl == "bf16":
        weight = weight_dequant(weight, _get_scale_tensor(weight))
        return F.linear(x_in, weight, bias)

    x_quant: torch.Tensor
    scale: torch.Tensor
    x_quant, scale = act_quant(x_in, block_size, scale_fmt)
    y_out: torch.Tensor = fp8_gemm(x_quant, scale, weight, _get_scale_tensor(weight))
    if bias is not None:
        y_out += bias
    return y_out


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    Args:
        dim (int): Dimension of the input tensor.
        eps (float): Epsilon value for numerical stability. Defaults to 1e-6.
    """

    def __init__(self, dim: int, eps: float = 1e-6, weight: torch.Tensor | None = None):
        super().__init__()
        self.dim = dim
        self.eps = eps

        if weight is None:
            self.weight = nn.Parameter(init_func(torch.empty(dim, dtype=torch.float32)))
        else:
            self.weight = torch.nn.Parameter(weight)

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for RMSNorm.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor with the same shape as input.
        """
        dtype = torch.bfloat16
        if residual is None:
            x = x.float()
            var_s = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(var_s + self.eps)
            return (self.weight * x).to(dtype)

        x = residual = x.float() + residual.float()
        var_s = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var_s + self.eps)
        return (self.weight * x).to(dtype), residual.to(dtype)
