"""TopP operation module."""

import torch

__all__ = [
    "top_p",
]


def top_p(
    logits: torch.Tensor,
    in_indices: torch.Tensor,
    sampling_seed: torch.Tensor,
    positions: torch.Tensor,
    is_verify_mode: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    flag: int,
    indices: torch.Tensor,
    scores: torch.Tensor,
    debug_tensor: torch.Tensor,
    profile_logs: torch.Tensor,
) -> None:
    """top_p operation.

    Args:
        logits (Tensor): The logits tensor.
        in_indices (Tensor): The tensor containing input indices.
        sampling_seed (Tensor): Random seeds for each sequence position.
        positions (Tensor): Token positions for each sequence element.
        is_verify_mode (bool): A flag indicating if verify mode is enabled in MTP. When set to
                               `True`, the `in_indices` will be checked to check if it is in
                               the top-k values.
        temperature (float): The temperature parameter, used for scaling logits in softmax
                             calculations.
        top_p (float): The top-p value, used for nucleus sampling to restrict the selection to the
                       smallest set of tokens whose cumulative probability is greater than or equal
                       to `top_p`.
        top_k (int): The number of top-k values that occupy the top-p probability mass
                     during sampling.
        flag (int): Used in all reduction.
        indices (Tensor): The tensor containing output indices.
        scores (Tensor): The tensor containing corresponding scores for the indices.
        profile_logs (Tensor): A tensor for storing profiling log data during execution in MTP.
    """
    dim = logits.shape[-1]
    if dim == 19360:
        call_func = torch.ops.tilert.top_p_glm5_op
    elif dim == 16160:
        call_func = torch.ops.tilert.top_p_op
    else:
        raise ValueError(f"Unsupported dimension: {dim}")
    call_func(
        logits,
        in_indices,
        sampling_seed,
        positions,
        is_verify_mode,
        temperature,
        top_p,
        top_k,
        flag,
        indices,
        scores,
        debug_tensor,
        profile_logs,
    )
