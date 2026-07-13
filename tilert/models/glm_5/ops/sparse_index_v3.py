"""GLM5 sparse index op Python wrapper."""

import torch

__all__ = [
    "sparse_index_topk_v3",
]


def sparse_index_topk_v3(
    q: torch.Tensor,  # noqa: VNE001
    kv: torch.Tensor,
    weights: torch.Tensor,
    logits: torch.Tensor,
    indices: torch.Tensor,
    cur_pos: int,
    profile_logs: torch.Tensor,
) -> None:
    """GLM5 sparse index + top-k selection."""
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
            f"Unsupported head size: {head}. SparseIndexV3 fused op "
            "supports head number of 32 (GLM5)."
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
    workspace = torch.zeros(seqlen, (200 * 1024 + 260), dtype=torch.int32, device=device)
    torch.ops.tilert.sparse_index_topk_glm5_v3_op(
        q, kv, weights, logits, cur_pos, indices, workspace, profile_logs
    )
