"""DeepSeek v3.2 reference kernels (tilelang/triton implementations).

This package exposes helpers like `act_quant`, `fp8_gemm`, and `weight_dequant`
for tests and higher-level Python ops.

Note: `act_quant` and `fp8_gemm` require tilelang at *call* time, and
`weight_dequant` requires triton at *call* time, but importing this package
does not require tilelang or triton to be installed.
"""

from .kernel import act_quant, fp8_gemm, weight_dequant

__all__ = [
    "act_quant",
    "fp8_gemm",
    "weight_dequant",
]
