"""ReceiveSelectedTokenIds — receive idx_selects from GPU 0."""

import torch

__all__ = [
    "receive_selected_token_ids",
]


def receive_selected_token_ids(
    ll_buf: torch.Tensor,
    dst: torch.Tensor,
    expected_flag: int,
    profile_logs: torch.Tensor,
    model_arch: str,
    compute_kernel_type: str = "bf16",
) -> None:
    """Receive idx_selects from GPU 0.

    Args:
        ll_buf: Receive buffer on this GPU (written by GPU 0).
        dst: Destination idx_selects tensor [1, S, 2048] int32.
        expected_flag: Expected synchronization flag value.
        profile_logs: Profile logs tensor.
        model_arch: Model architecture ("deepseek_v3_2" or "glm_5").
        compute_kernel_type: Compute kernel type ("bf16").
    """
    torch.ops.tilert.receive_selected_token_ids_op(
        ll_buf,
        dst,
        expected_flag,
        model_arch,
        compute_kernel_type,
        profile_logs,
    )
