"""Sparse index operation module."""

import torch

__all__ = [
    "sparse_index",
    "sparse_index_topk",
]


def sparse_index(
    q: torch.Tensor,  # noqa: VNE001
    kv: torch.Tensor,
    weights: torch.Tensor,
    logits: torch.Tensor,
    cur_pos: int,
    profile_logs: torch.Tensor,
) -> None:
    """
    Sparse index operation.

    Calculate sparse index using q * kv * weights.

    Args:
        q (torch.Tensor): The query tensor.
        kv (torch.Tensor): The key-value tensor.
        weights (torch.Tensor): The weights tensor.
        logits (torch.Tensor): The logits tensor.
        cur_pos (int): The position of the first token.
        profile_logs (torch.Tensor): Tensor for storing profiling logs.

    Returns:
        None
    """
    if q.dtype != torch.bfloat16:
        raise ValueError("input must be a bfloat16 tensor.")
    if kv.dtype != torch.bfloat16:
        raise ValueError("kv must be a bfloat16 tensor.")
    if weights.dtype != torch.bfloat16:
        raise ValueError("weights must be a bfloat16 tensor.")
    if logits.dtype != torch.float32:
        raise ValueError("logits must be a float32 tensor.")

    head = q.shape[-2]
    dim = q.shape[-1]

    if head != 64 and head != 32:
        raise ValueError(
            f"Unsupported head size: {head}. Sparse index op currently only \
                supports a head number of 64 or 32."
        )
    if dim != 128:
        raise ValueError("dim must be 128, as we precompute scale inner kernel")

    device = q.device
    if any(t.device != device for t in (kv, weights, logits, profile_logs)):
        raise ValueError(
            "sparse_index inputs must be on the same device: "
            f"q={device}, kv={kv.device}, weights={weights.device}, "
            f"logits={logits.device}, profile_logs={profile_logs.device}"
        )
    if head == 64:
        torch.ops.tilert.sparse_index_op(q, kv, weights, logits, cur_pos, profile_logs)
    elif head == 32:
        torch.ops.tilert.sparse_index_glm5_op(q, kv, weights, logits, cur_pos, profile_logs)


def sparse_index_topk(
    q: torch.Tensor,  # noqa: VNE001
    kv: torch.Tensor,
    weights: torch.Tensor,
    logits: torch.Tensor,
    indices: torch.Tensor,
    cur_pos: int,
    profile_logs: torch.Tensor,
) -> None:
    """
    Sparse index operation.

    Calculate sparse index using q * kv * weights.

    Args:
        q (torch.Tensor): The query tensor.
        kv (torch.Tensor): The key-value tensor.
        weights (torch.Tensor): The weights tensor.
        logits (torch.Tensor): The logits tensor.
        cur_pos (int): The position of the first token.
        profile_logs (torch.Tensor): Tensor for storing profiling logs.

    Returns:
        None
    """
    if q.dtype != torch.bfloat16:
        raise ValueError("input must be a bfloat16 tensor.")
    if kv.dtype != torch.bfloat16:
        raise ValueError("kv must be a bfloat16 tensor.")
    if weights.dtype != torch.bfloat16:
        raise ValueError("weights must be a bfloat16 tensor.")
    if logits.dtype != torch.float32:
        raise ValueError("logits must be a float32 tensor.")

    seqlen = q.shape[-3]
    head = q.shape[-2]
    dim = q.shape[-1]

    if head != 32:
        raise ValueError(
            f"Unsupported head size: {head}. Sparse index topk fused op currently only \
                supports a head number of 32."
        )
    if dim != 128:
        raise ValueError("dim must be 128, as we precompute scale inner kernel")

    device = q.device
    if any(t.device != device for t in (kv, weights, logits, indices, profile_logs)):
        raise ValueError(
            "sparse_index inputs must be on the same device: "
            f"q={device}, kv={kv.device}, weights={weights.device}, "
            f"logits={logits.device}, profile_logs={profile_logs.device}"
        )
    workspace = torch.zeros(seqlen, (200 * 1024 + 258), dtype=torch.int32, device=device)
    torch.ops.tilert.sparse_index_topk_glm5_op(
        q, kv, weights, logits, cur_pos, indices, workspace, profile_logs
    )
