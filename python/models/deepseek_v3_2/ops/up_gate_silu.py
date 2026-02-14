"""UpGateSiLU operation module."""

import torch

__all__ = [
    "up_gate_silu",
]


def up_gate_silu(
    hidden_in: torch.Tensor,
    expert_indices_in: torch.Tensor,
    experts_weights_in: torch.Tensor,
    hidden_out: torch.Tensor,
    profile_logs: torch.Tensor,
) -> None:
    """Up Gate SiLU operation."""
    torch.ops.tilert.up_gate_silu_op(
        hidden_in,
        expert_indices_in,
        experts_weights_in,
        hidden_out,
        profile_logs,
    )
