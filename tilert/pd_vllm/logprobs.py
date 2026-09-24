from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

__all__ = [
    "GREEDY_TEMPERATURE",
    "LOGPROB_UNAVAILABLE",
    "MIN_LOGPROBS_TEMPERATURE",
    "TOP_LOGPROBS_MAX",
    "LogprobsRequest",
    "LogprobsUnsupported",
    "build_logprobs",
    "resolve_logprobs_request",
]
TOP_LOGPROBS_MAX = 5
GREEDY_TEMPERATURE = 1e-05
MIN_LOGPROBS_TEMPERATURE = 0.2
LOGPROB_UNAVAILABLE = -9999.0


class LogprobsUnsupported(Exception):
    error_type = "invalid_logprobs"
    http_status = 400

    def to_payload(self) -> dict[str, str]:
        return {"error": str(self), "error_type": self.error_type}


@dataclass(frozen=True)
class LogprobsRequest:
    top_n: int


def resolve_logprobs_request(body: dict) -> LogprobsRequest | None:
    enabled = body.get("logprobs")
    if enabled is not None and (not isinstance(enabled, bool)):
        raise LogprobsUnsupported(f"logprobs must be a boolean, got {type(enabled).__name__}")
    raw = body.get("top_logprobs")
    top_n = 0
    if raw is not None:
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise LogprobsUnsupported(f"top_logprobs must be an integer, got {type(raw).__name__}")
        if not 0 <= raw <= TOP_LOGPROBS_MAX:
            raise LogprobsUnsupported(f"top_logprobs must be in [0, {TOP_LOGPROBS_MAX}], got {raw}")
        if raw > 0 and (not enabled):
            raise LogprobsUnsupported("when using top_logprobs, logprobs must be set to true")
        top_n = raw
    if not enabled:
        return None
    temp = body.get("temperature")
    if isinstance(temp, (int, float)) and (not isinstance(temp, bool)):
        t = float(temp)
        if GREEDY_TEMPERATURE <= t < MIN_LOGPROBS_TEMPERATURE:
            raise LogprobsUnsupported(
                f"logprobs require temperature >= {MIN_LOGPROBS_TEMPERATURE} or greedy (temperature < {GREEDY_TEMPERATURE}), got {t}"
            )
    return LogprobsRequest(top_n=top_n)


def _entry(token_id: int, logprob: float | None, decode_one: Callable[[int], str]) -> dict:
    text = decode_one(token_id)
    return {
        "token": text,
        "logprob": LOGPROB_UNAVAILABLE if logprob is None else _finite(logprob),
        "bytes": list(text.encode("utf-8")),
    }


def _finite(value: float) -> float:
    return LOGPROB_UNAVAILABLE if value == float("-inf") else float(value)


def build_logprobs(
    token_ids: list[int],
    token_logprobs: list[float | None],
    top_logprobs: list[list[tuple[int, float]]] | None,
    req: LogprobsRequest,
    decode_one: Callable[[int], str],
) -> dict:
    if len(token_logprobs) != len(token_ids):
        raise ValueError(
            f"token_logprobs has {len(token_logprobs)} entries for {len(token_ids)} tokens"
        )
    if top_logprobs is not None and len(top_logprobs) != len(token_ids):
        raise ValueError(
            f"top_logprobs has {len(top_logprobs)} entries for {len(token_ids)} tokens"
        )
    content = []
    for i, tid in enumerate(token_ids):
        item = _entry(tid, token_logprobs[i], decode_one)
        alts = [] if top_logprobs is None else top_logprobs[i][: req.top_n]
        item["top_logprobs"] = [_entry(alt_id, alt_lp, decode_one) for alt_id, alt_lp in alts]
        content.append(item)
    return {"content": content, "refusal": None}
