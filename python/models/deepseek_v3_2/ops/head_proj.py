"""HeadProj operation module."""

import torch

__all__ = [
    "head_proj",
]


def head_proj(
    hidden_in: torch.Tensor,
    weight_in: torch.Tensor,
    logits_out: torch.Tensor,
    profile_logs: torch.Tensor,
) -> None:
    """Head Projection operation."""
    torch.ops.tilert.head_proj_op(
        hidden_in,
        weight_in,
        logits_out,
        profile_logs,
    )
