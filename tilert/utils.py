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


def copy_by_device_pair(
    copies: list[tuple[torch.Tensor, torch.Tensor]],
    streams: dict[tuple[int, int], torch.cuda.Stream],
) -> None:
    """Run ``dst.copy_(src)`` for every pair, one stream pair per device pair.

    torch fences a cross-device ``copy_`` against the current stream of both
    devices, so issuing many cache copies on the default streams runs them one
    at a time over a single link. Grouping them by (destination, source) device
    and giving each group its own streams lets the pairs overlap. ``streams``
    caches the streams between calls. Returns after every copy has finished.

    Args:
        copies: (destination, source) tensor pairs. Destinations must be CUDA
            tensors; sources may be CUDA or CPU tensors.
        streams: Cache of streams keyed by (device, peer device).
    """

    def stream(dev: int, peer: int) -> torch.cuda.Stream:
        key = (dev, peer)
        if key not in streams:
            streams[key] = torch.cuda.Stream(device=dev)
        return streams[key]

    by_pair: dict[tuple[int, int], list[tuple[torch.Tensor, torch.Tensor]]] = {}
    for dst, src in copies:
        src_dev = src.device.index if src.is_cuda else -1
        by_pair.setdefault((dst.device.index, src_dev), []).append((dst, src))
    devices = set()
    for (dst_dev, src_dev), group in by_pair.items():
        devices.add(dst_dev)
        dst_stream = stream(dst_dev, src_dev)
        if src_dev < 0:
            with torch.cuda.stream(dst_stream):
                for dst, src in group:
                    dst.copy_(src, non_blocking=True)
            continue
        devices.add(src_dev)
        with torch.cuda.stream(dst_stream), torch.cuda.stream(stream(src_dev, dst_dev)):
            for dst, src in group:
                dst.copy_(src, non_blocking=True)
    for st in streams.values():
        st.synchronize()
    for dev in devices:
        torch.cuda.synchronize(dev)
