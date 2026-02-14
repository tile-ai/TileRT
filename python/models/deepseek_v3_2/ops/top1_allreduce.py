"""Top1 Allreduce operation"""

import torch

__all__ = [
    "top1_allreduce",
]


def top1_allreduce(
    logits: torch.Tensor,
    flag: int,
    index_out: torch.Tensor,
    profile_logs: torch.Tensor,
) -> None:
    """
    Define the Top1 Allreduce operation.

    Args:
        logits: Input tensor.
        flag: Flag.
        index_out: Output tensor.
        profile_logs: Profile logs tensor.
    """
    torch.ops.tilert.top1_allreduce_op(logits, flag, index_out, profile_logs)
