"""topk operations module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from tilert.utils import get_profile_log_tensor

if TYPE_CHECKING:
    from tilert.models.deepseek_v3_2.model_args import ModelArgs


__all__ = [
    "TopK",
    "topk_approximate",
    "topk_accurate",
]


def topk_approximate(
    logits: torch.Tensor,
    seq_len: int,
    topk: int,
    profile_logs: torch.Tensor,
    model_arch: str,
    compute_kernel_type: str = "general",
) -> torch.Tensor:
    """
    Topk approximate operation.

    Topk approximate the input tensor `logits` and stores the result in `output_raw`.

    Args:
        logits (torch.Tensor): The input tensor.
        seq_len (int): valid data of logits.shape[-1]
        topk (int): The number of topk to approximate.
        profile_logs (torch.Tensor): The profile logs tensor.

    Returns:
        indices (torch.Tensor): The output tensor.
    """
    if logits.dtype != torch.float32:
        raise ValueError("logits must be a float32 tensor.")

    if topk != 2048:
        raise ValueError("topk must be 2048.")
    batch = logits.shape[0]
    if batch != 1:
        raise ValueError("batch must be 1 in this version")

    indices = torch.zeros(batch, topk, dtype=torch.int32, device=logits.device)
    torch.ops.tilert.topk_approximate_op(
        logits, indices, seq_len, model_arch, compute_kernel_type, profile_logs
    )

    return indices


def topk_accurate(
    logits: torch.Tensor,
    seq_len: int,
    topk: int,
    profile_logs: torch.Tensor,
    model_arch: str,
    compute_kernel_type: str = "general",
    ratio: int = 1,
) -> torch.Tensor:
    """
    Topk approximate operation.

    Topk approximate the input tensor `logits` and stores the result in `output_raw`.

    Args:
        logits (torch.Tensor): The input tensor.
        seq_len (int): length of last samples,
        for k=logits.shape[1] samples, the length is
        seq-k+1, seq-k+2, ..., seq-1, seq
        topk (int): The number of topk to approximate.
        profile_logs (torch.Tensor): The profile logs tensor.
        ratio (int): Token-domain to logits-trailing-dim compression factor.
    Returns:
        indices (torch.Tensor): The output tensor.
    """
    if logits.dtype != torch.float32:
        raise ValueError("logits must be a float32 tensor.")

    if topk not in (512, 1024, 2048):
        raise ValueError("topk must be 512, 1024, or 2048.")

    assert logits.shape[0] == 1, "batch must be 1 in this version"
    num_samples = logits.shape[1]

    indices = torch.zeros(num_samples, topk, dtype=torch.int32, device=logits.device)
    indices_ws = torch.zeros(1, num_samples, 4, topk * 2, dtype=torch.int32, device=logits.device)
    torch.ops.tilert.topk_accurate_op(
        logits,
        indices,
        seq_len - num_samples,
        indices_ws,
        model_arch,
        compute_kernel_type,
        profile_logs,
        ratio,
    )

    return indices


class TopK(nn.Module):
    """TopK operation with optional approximate kernel.

    Wraps topk_accurate / topk_approximate and provides golden_forward
    (reference implementation) and tilert_forward (TileRT kernel).
    """

    def __init__(self, use_approximate: bool = False, model_args: ModelArgs | None = None) -> None:
        super().__init__()
        self.use_approximate = use_approximate
        if model_args is None:
            from tilert.models.deepseek_v3_2.model_args import ModelArgs

            model_args = ModelArgs()
        self.model_args = model_args

    def golden_forward(
        self,
        logits: torch.Tensor,
        topk: int,
    ) -> torch.Tensor:
        """Reference forward: torch.topk on the last dimension.

        Args:
            logits: Scores tensor, shape (batch, ..., seq_len).
            topk: Number of top indices to return.

        Returns:
            Indices of top-k values along the last dimension.
        """
        seq_len = logits.shape[-1]
        return logits.topk(min(topk, seq_len), dim=-1)[1]

    def tilert_forward(
        self,
        logits: torch.Tensor,
        topk: int,
    ) -> torch.Tensor:
        """Tilert forward: batch of samples with varying valid length.

        Args:
            logits: Shape (batch, num_samples, cache_len).
            topk: Number of top indices to return.

        Returns:
            Indices tensor of shape (batch, num_samples, topk).
        """
        profile_logs = get_profile_log_tensor(device=logits.device)
        cache_len = logits.shape[-1]
        if self.use_approximate:
            indices = topk_approximate(
                logits, cache_len, topk, profile_logs, model_arch=self.model_args.arch_name
            )
        else:
            indices = topk_accurate(
                logits, cache_len, topk, profile_logs, model_arch=self.model_args.arch_name
            )
        if indices.dim() == 2:
            return indices.unsqueeze(0)
        return indices
