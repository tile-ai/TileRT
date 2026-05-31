"""BroadcastSelectedTokenIds — P2P broadcast of idx_selects from GPU 0 to peers."""

import torch

__all__ = [
    "broadcast_selected_token_ids",
]


def broadcast_selected_token_ids(
    idx_selects: torch.Tensor,
    peer_bufs: torch.Tensor,
    flag_val: int,
    profile_logs: torch.Tensor,
    model_arch: str,
    compute_kernel_type: str = "bf16",
) -> None:
    """Broadcast idx_selects [1,S,2048] int32 from GPU 0 to peer GPUs.

    Args:
        idx_selects: Source tensor [1, S, 2048] int32 on GPU 0.
        peer_bufs: Device pointer array [N] int64 — each entry is a peer
            buffer address.
        flag_val: Synchronization flag value.
        profile_logs: Profile logs tensor.
        model_arch: Model architecture ("deepseek_v3_2" or "glm_5").
        compute_kernel_type: Compute kernel type ("bf16").
    """
    torch.ops.tilert.broadcast_selected_token_ids_op(
        idx_selects,
        peer_bufs,
        flag_val,
        model_arch,
        compute_kernel_type,
        profile_logs,
    )
