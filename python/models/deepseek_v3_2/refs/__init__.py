"""DeepSeek v3.2 reference kernels (tilelang/triton implementations).

This package exposes helpers like `act_quant`, `fp8_gemm`, and `weight_dequant`
for tests and higher-level Python ops.
"""

from .kernel import act_quant, fp8_gemm, weight_dequant

__all__ = [
    "act_quant",
    "fp8_gemm",
    "weight_dequant",
]
