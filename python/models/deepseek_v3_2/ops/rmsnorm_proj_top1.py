"""RMSNorm + head projection + top1 operation"""

import torch

__all__ = [
    "rmsnorm_proj_top1",
]


def rmsnorm_proj_top1(
    hidden_in: torch.Tensor,
    rmsnorm_gamma_in: torch.Tensor,
    head_projection_weights_in: torch.Tensor,
    token_id: torch.Tensor,
    profile_logs: torch.Tensor,
) -> None:
    """
    Define the RMSNormProjTop1 operation.

    Args:
        hidden_in: Input tensor.
        rmsnorm_gamma_in: Weight tensor.
        head_projection_weights_in: Weight tensor.
        token_id: Output tensor.
        profile_logs: Profile logs tensor.
    """
    torch.ops.tilert.rmsnorm_proj_top1_op(
        hidden_in, rmsnorm_gamma_in, head_projection_weights_in, token_id, profile_logs
    )
