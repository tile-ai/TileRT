"""Utility functions for testing."""

from typing import Any

import torch

__all__ = [
    "cosine_similarity",
    "relative_l2_error",
    "get_profile_log_tensor",
    "SLICES_FOR_TILERT_OP",
]


# A hack: Add some extra bytes for storing instructions and glb bars.
# Note that it should be no smaller than ((vm::DefaultConfig::kOpInstructionBytes +
# vm::DefaultConfig::kMaxGlbBarsPerOp * sizeof(uint32_t)) *
# vm::DefaultConfig::kMaxNumOps) = 64 * (128 + 8);
# Here 1 slice is enough.
SLICES_FOR_TILERT_OP = 1


def get_profile_log_tensor(
    device_index: int = 0, device: torch.device | None = None, num_max_insts: int = 64
) -> torch.Tensor:
    """Get a profile log tensor for the given device index.

    Args:
        device_index: The index of the device.
        device: The device to use.

    Returns:
        A profile log tensor.
    """
    if device is None:
        device = torch.device("cuda", device_index)

    props = torch.cuda.get_device_properties(device_index)
    num_sm = props.multi_processor_count

    return torch.zeros(
        num_max_insts + 1 + SLICES_FOR_TILERT_OP,
        num_sm,
        16,
        dtype=torch.uint64,
        device=device,
    )


def cosine_similarity(gt: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Calculate the cosine similarity.

    Args:
        gt: The ground truth tensor.
        out: The output tensor.

    Returns:
        The cosine similarity.
    """
    return torch.nn.functional.cosine_similarity(gt.flatten(), out.flatten(), dim=-1)


def relative_l2_error(gt: torch.Tensor, out: torch.Tensor) -> Any:
    """Calculate the relative L2 error.

    Args:
        gt: The ground truth tensor.
        out: The output tensor.

    Returns:
        The relative L2 error.
    """
    return torch.norm(gt - out) / torch.norm(gt)
