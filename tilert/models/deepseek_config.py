"""Global configuration for DeepSeek models."""

from typing import Literal

import torch.distributed as dist

__all__ = [
    "get_world_size",
    "get_rank",
    "block_size",
    "gemm_impl",
]


def get_world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 8


def get_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


block_size = 128
gemm_impl: Literal["bf16", "fp8"] = "bf16"
