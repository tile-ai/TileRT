"""Global configuration for DeepSeek models."""

import os
from typing import Literal

import torch
import torch.distributed as dist

__all__ = [
    "get_world_size",
    "get_rank",
    "block_size",
    "gemm_impl",
    "attn_impl",
]


def is_distributed() -> bool:
    return bool(dist.is_initialized())


def get_world_size() -> int:
    # NOTE: default world size is 8, since tilert kernels implemented for 8 GPUs.
    # DO NOT modify this value unless you know how much it affects the tilert kernels.
    return dist.get_world_size() if dist.is_initialized() else 8


def get_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def init_distributed_training() -> tuple[int, int, bool]:
    """Initialize distributed training."""
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        is_distributed = True
    else:
        local_rank = 0
        world_rank = 0
        world_size = 1
        is_distributed = False

    torch.cuda.set_device(local_rank)
    torch.set_default_device(f"cuda:{local_rank}")
    torch.set_default_dtype(torch.bfloat16)

    if world_size > 1:
        dist.init_process_group(
            backend="nccl",
            world_size=world_size,
            rank=world_rank,
            init_method="env://",
            device_id=local_rank,
        )

    rank = get_rank()
    world_size = get_world_size()

    return rank, world_size, is_distributed


block_size = 128
gemm_impl: Literal["bf16", "fp8"] = "bf16"
attn_impl: Literal["naive", "absorb"] = "absorb"
