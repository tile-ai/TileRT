from typing import cast

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "init_func",
    "linear",
    "Linear",
    "RMSNorm",
    "LayerNorm",
    "ColumnParallelLinear",
    "RowParallelLinear",
    "ParallelEmbedding",
]

from tilert.models.deepseek_config import (
    block_size,
    gemm_impl,
    get_rank,
    get_world_size,
    is_distributed,
)
from tilert.models.deepseek_v3_2.refs.kernel import act_quant, fp8_gemm, weight_dequant


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

    This function supports specialized implementations based on quantization
    and tensor formats.

    Args:
        x_in (torch.Tensor): The input tensor.
        weight (torch.Tensor): The weight tensor. It may be quantized and
            requires dequantization for certain cases.
        bias (Optional[torch.Tensor]): The bias tensor to be added. Default is None.

    Returns:
        torch.Tensor: The result of the linear transformation, which may involve
        quantization-aware computations depending on the input parameters.

    Notes:
        - If `weight` is quantized (e.g., `element_size() == 1`), a dequantized version is used
          for computation.
        - If `gemm_impl == "bf16"`, dequantization and a `bf16` GEMM operation are applied.
        - For other cases, the function applies quantization to `x` and uses `fp8_gemm`
          for computation.
    """
    if weight.element_size() > 1:
        return F.linear(x_in, weight, bias)
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


class Linear(nn.Module):
    """
    Custom linear layer with support for quantized weights and optional bias.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """

    dtype = torch.bfloat16
    scale_fmt: str | None = None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        dtype: torch.dtype | None = None,
        weight: torch.Tensor | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        if weight is not None:
            self.weight = torch.nn.Parameter(weight)
        else:
            self.weight = nn.Parameter(
                init_func(torch.empty(out_features, in_features, dtype=dtype or Linear.dtype))
            )

        if self.weight.element_size() == 1:
            scale_out_features = (out_features + block_size - 1) // block_size
            scale_in_features = (in_features + block_size - 1) // block_size
            scale_param = nn.Parameter(
                init_func(
                    torch.empty(
                        scale_out_features,
                        scale_in_features,
                        dtype=torch.float32,
                    )
                )
            )
            self.scale = scale_param
            self.weight.scale = scale_param  # type: ignore[attr-defined]
        else:
            self.register_parameter("scale", None)

        if bias:
            self.bias = nn.Parameter(init_func(torch.empty(out_features)))
        else:
            self.register_parameter("bias", None)

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the custom linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor after linear computation.
        """
        return linear(x_in, self.weight, self.bias, self.scale_fmt)


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
        dtype = torch.bfloat16  # x.dtype
        if residual is None:
            x = x.float()
            var_s = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(var_s + self.eps)
            return (self.weight * x).to(dtype)

        x = residual = x.float() + residual.float()
        var_s = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var_s + self.eps)
        return (self.weight * x).to(dtype), residual.to(dtype)


class LayerNorm(nn.Module):
    """Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x.float(), (self.dim,), self.weight, self.bias, self.eps).type_as(x)


class ColumnParallelLinear(Linear):
    """
    Column parallel linear layer.

    Linear layer with column parallelism, splitting output features across
    distributed processes.

    Args:
        in_features (int): Number of input features.
        out_features (int): Total number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        dtype: torch.dtype | None = None,
    ):
        world_size = get_world_size()
        assert (
            out_features % world_size == 0
        ), f"Output features must be divisible by world size {world_size}"
        self.part_out_features = out_features // world_size
        super().__init__(in_features, self.part_out_features, bias, dtype)

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for column parallel linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor with column-parallel computation.
        """
        return linear(x_in, self.weight, self.bias)


class RowParallelLinear(Linear):
    """
    Linear layer with row parallelism, splitting input features across distributed processes.

    Args:
        in_features (int): Total number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        reduce_output: bool = True,
        dtype: torch.dtype | None = None,
    ):

        self.world_size = get_world_size()

        if in_features % self.world_size != 0:
            raise ValueError(
                f"Input features must be divisible by world size (world_size={self.world_size})"
            )

        self.part_in_features = in_features // self.world_size
        self.reduce_output = reduce_output

        super().__init__(self.part_in_features, out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for row parallel linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor with row-parallel computation.
        """
        y = linear(x, self.weight, None, self.scale_fmt)
        if self.reduce_output and is_distributed() > 1:
            y = y.float()
            dist.all_reduce(y)
        if self.bias is not None:
            y += self.bias
        return y.type_as(x)


class ParallelEmbedding(nn.Module):
    """
    Parallel embedding layer.

    Embedding layer with parallelism support across distributed processes.

    Args:
        vocab_size (int): Vocabulary size.
        dim (int): Embedding dimension.
    """

    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim

        self.world_size = get_world_size()
        self.rank = get_rank()

        assert (
            vocab_size % self.world_size == 0
        ), f"Vocabulary size must be divisible by world size {self.world_size}"

        self.part_vocab_size = vocab_size // self.world_size
        self.vocab_start_idx = self.rank * self.part_vocab_size
        self.vocab_end_idx = self.vocab_start_idx + self.part_vocab_size

        self.weight = nn.Parameter(init_func(torch.empty(self.part_vocab_size, self.dim)))

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for parallel embedding layer.

        Args:
            x (torch.Tensor): Input tensor containing token indices.

        Returns:
            torch.Tensor: Embedded representations.

        Raises:
            ValueError: If `world_size` is not defined.
        """
        if self.world_size > 1:
            mask = (x_in < self.vocab_start_idx) | (x_in >= self.vocab_end_idx)
            x_in = x_in - self.vocab_start_idx
            x_in[mask] = 0

        y_out = F.embedding(x_in, self.weight)

        if is_distributed():
            y_out[mask] = 0
            dist.all_reduce(y_out)
        return y_out
