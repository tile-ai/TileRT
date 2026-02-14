"""ExpertSelect operation module."""

import torch

__all__ = [
    "expert_select",
    "expert_select_one_stage",
]


def expert_select(
    scores_in: torch.Tensor,
    bias_in: torch.Tensor,
    expert_probs_out: torch.Tensor,
    expert_indices_out: torch.Tensor,
    profile_logs: torch.Tensor,
) -> None:
    """
    Expert Select operation.

    Original two-stage expert select operation used in DeepSeek V3.2.
    """
    torch.ops.tilert.expert_select_op(
        scores_in,
        bias_in,
        expert_probs_out,
        expert_indices_out,
        profile_logs,
    )


def expert_select_one_stage(
    scores_in: torch.Tensor,
    bias_in: torch.Tensor,
    expert_probs_out: torch.Tensor,
    expert_indices_out: torch.Tensor,
    profile_logs: torch.Tensor,
) -> None:
    """Expert Select operation.

    Modified one-stage expert select operation used in Kimi and GLM.
    """
    torch.ops.tilert.expert_select_glm5_op(
        scores_in,
        bias_in,
        expert_probs_out,
        expert_indices_out,
        profile_logs,
    )
