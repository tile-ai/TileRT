"""Utility functions for testing."""

from typing import Any

import torch

__all__ = [
    "alloc_misc_ws",
    "cosine_similarity",
    "relative_l2_error",
    "get_profile_log_tensor",
    "SLICES_FOR_TILERT_OP",
]

SLICES_FOR_TILERT_OP = 1


def get_profile_log_tensor(
    device_index: int = 0,
    device: torch.device | None = None,
    num_max_insts: int = 64,
) -> torch.Tensor | None:
    """Get a profile log tensor for the given device index.

    Returns ``None`` when no CUDA GPUs are visible so the offline
    weight-conversion path can run with ``CUDA_VISIBLE_DEVICES=""``.

    Args:
        device_index: The index of the device.
        device: The device to use.

    Returns:
        A profile log tensor, or ``None`` if CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        return None
    if device is None:
        device = torch.device("cuda", device_index)

    props = torch.cuda.get_device_properties(device_index)
    num_sm = props.multi_processor_count

    return torch.zeros(
        num_max_insts + 1 + SLICES_FOR_TILERT_OP, num_sm, 16, dtype=torch.uint64, device=device
    )


def alloc_misc_ws(
    num_max_insts: int = 64,
    device_id: int = 0,
) -> torch.Tensor:
    """Allocate a misc workspace tensor.

    Args:
        num_max_insts: Maximum number of profiled instructions.
        device_id: CUDA device index to allocate on.

    Returns:
        A zeroed int64 tensor of shape (total_rows, num_sm, 16) on the
        requested CUDA device.
    """
    return torch.ops.tilert.alloc_misc_ws(num_max_insts, device_id)


def cosine_similarity(gt: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Calculate the cosine similarity.

    Args:
        gt: The ground truth tensor.
        out: The output tensor.

    Returns:
        The cosine similarity.
    """
    return torch.nn.functional.cosine_similarity(
        gt.flatten().float(), out.flatten().float(), dim=-1
    )


def relative_l2_error(gt: torch.Tensor, out: torch.Tensor) -> Any:
    """Calculate the relative L2 error.

    Args:
        gt: The ground truth tensor.
        out: The output tensor.

    Returns:
        The relative L2 error.
    """
    return torch.norm(gt - out) / torch.norm(gt)
