from __future__ import annotations

__all__ = [
    "GREEDY_LOGPROBS_TOP_P",
    "TOP_K_DISABLED",
    "VLLM_DEFAULT_TOP_P",
    "resolve_top_k",
    "resolve_top_p",
]
GREEDY_LOGPROBS_TOP_P = 1e-09
VLLM_DEFAULT_TOP_P = 1.0


def resolve_top_p(sampling: dict, default: float = VLLM_DEFAULT_TOP_P) -> float:
    raw = sampling.get("top_p")
    if raw is None:
        return float(default)
    if isinstance(raw, bool):
        raise ValueError("top_p must be a number, got bool")
    return float(raw)


_KERNEL_TOP_K_POOL = 256
TOP_K_DISABLED = _KERNEL_TOP_K_POOL
_TOP_K_APPLIED_MAX = _KERNEL_TOP_K_POOL - 1


def resolve_top_k(sampling: dict) -> int:
    raw = sampling.get("top_k")
    if raw is None:
        return TOP_K_DISABLED
    k = int(raw)
    if k < 1 or k > _TOP_K_APPLIED_MAX:
        return TOP_K_DISABLED
    return k
