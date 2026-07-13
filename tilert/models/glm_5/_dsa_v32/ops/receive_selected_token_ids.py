"""ReceiveSelectedTokenIds — receive idx_selects from GPU 0."""

import torch

__all__ = [
    "receive_selected_token_ids",
]


def receive_selected_token_ids(
    recv_buf: torch.Tensor,
    dst: torch.Tensor,
    expected_flag: int,
    profile_logs: torch.Tensor,
    model_arch: str,
    compute_kernel_type: str = "bf16",
) -> None:
    torch.ops.tilert.receive_selected_token_ids_op(
        recv_buf,
        dst,
        expected_flag,
        model_arch,
        compute_kernel_type,
        profile_logs,
    )
