"""Chat-completions ``logprobs`` / ``top_logprobs``: request parsing and response assembly.

Contract (OpenAI chat completions, narrowed to the stricter vendor reading):

``logprobs``
    boolean, default false. "Whether to return log probabilities of the output
    tokens or not. If true, returns the log probabilities of each output token
    returned in the ``content`` of ``message``."

``top_logprobs``
    integer, default 0, range ``[0, 5]``. The number of most likely tokens to
    return at each position, each with a log probability. ``logprobs`` must be
    true if it is used.

Two consequences of that wording drive this module:

* logprobs cover ``message.content`` only. A reasoning segment lives in a
  different field (``reasoning_content``), so its tokens carry no logprobs.
* ``top_logprobs`` is capped at 5 here, not OpenAI's current 20. The API
  reference says 20 and the cookbook still says 5; 5 is the narrower of the
  two, so a client written against either gets a consistent answer.

Out of range is rejected rather than clamped, matching how vLLM handles a
logprobs count above its own cap.
"""

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

# At or below this the request is greedy: the engine takes its top-1 kernel, and
# temperature cannot change which token that picks.
GREEDY_TEMPERATURE = 1e-5

# Lowest temperature the decode sampler's log-probability export is valid at.
# Set by the engine, not by this contract: see the note in
# resolve_logprobs_request and RawPowSum in include/ops/deepseek_v3_2/top_p.cuh.
MIN_LOGPROBS_TEMPERATURE = 0.2

# OpenAI documents -9999.0 as the value standing in for "very unlikely", and
# uses it where a real log probability is not available. JSON has no -inf, so a
# missing or -inf entry must surface as this rather than null or Infinity.
LOGPROB_UNAVAILABLE = -9999.0


class LogprobsUnsupported(Exception):
    """The client's logprobs request cannot be served -> HTTP 400.

    Mirrors ``GrammarError``'s shape so the router can return it the same way.
    """

    error_type = "invalid_logprobs"
    http_status = 400

    def to_payload(self) -> dict[str, str]:
        return {"error": str(self), "error_type": self.error_type}


@dataclass(frozen=True)
class LogprobsRequest:
    """A validated request for logprobs. ``top_n`` is already in [0, 5]."""

    top_n: int


def resolve_logprobs_request(body: dict) -> LogprobsRequest | None:
    """Parse and validate ``logprobs`` / ``top_logprobs``.

    Returns ``None`` when the client did not ask for logprobs, so callers can
    skip the whole path. Raises :class:`LogprobsUnsupported` (400) on a request
    that is malformed or outside the supported range -- never degrades silently,
    because a client that asked for logprobs and got none has no way to tell.
    """
    enabled = body.get("logprobs")
    if enabled is not None and not isinstance(enabled, bool):
        raise LogprobsUnsupported(f"logprobs must be a boolean, got {type(enabled).__name__}")

    raw = body.get("top_logprobs")
    top_n = 0
    if raw is not None:
        # bool is an int subclass in Python; `top_logprobs: true` is a type
        # error, not a request for 1.
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise LogprobsUnsupported(f"top_logprobs must be an integer, got {type(raw).__name__}")
        if not 0 <= raw <= TOP_LOGPROBS_MAX:
            raise LogprobsUnsupported(f"top_logprobs must be in [0, {TOP_LOGPROBS_MAX}], got {raw}")
        if raw > 0 and not enabled:
            raise LogprobsUnsupported("when using top_logprobs, logprobs must be set to true")
        top_n = raw

    if not enabled:
        return None

    # Greedy and the top-p sampler both export a chosen-token value and a
    # candidate row, so both serve any top_logprobs up to TOP_LOGPROBS_MAX --
    # which is the greedy kernel's row width, the narrower of the two.
    #
    # The band between them is refused rather than degraded, for the reason an
    # out-of-range top_logprobs is refused: a wrong number is indistinguishable
    # from a right one. There the temperature genuinely selects the
    # distribution being sampled, so greedy's export cannot stand in for it,
    # and the top-p path's raw denominator loses accuracy (order 0.01 nat at
    # T = 0.1, 0.3 at T = 0.05). A non-numeric temperature is vLLM's to reject.
    temp = body.get("temperature")
    if isinstance(temp, (int, float)) and not isinstance(temp, bool):
        t = float(temp)
        if GREEDY_TEMPERATURE <= t < MIN_LOGPROBS_TEMPERATURE:
            raise LogprobsUnsupported(
                f"logprobs require temperature >= {MIN_LOGPROBS_TEMPERATURE} "
                f"or greedy (temperature < {GREEDY_TEMPERATURE}), got {t}"
            )
    return LogprobsRequest(top_n=top_n)


def _entry(token_id: int, logprob: float | None, decode_one: Callable[[int], str]) -> dict:
    """One ``{token, logprob, bytes}`` object.

    ``bytes`` carries the UTF-8 encoding of the token text, which is how a
    caller reassembles text when a single token is not valid UTF-8 on its own
    (byte-level BPE splits multi-byte characters across tokens).
    """
    text = decode_one(token_id)
    return {
        "token": text,
        "logprob": LOGPROB_UNAVAILABLE if logprob is None else _finite(logprob),
        "bytes": list(text.encode("utf-8")),
    }


def _finite(value: float) -> float:
    """JSON has no infinities; -inf becomes the documented sentinel."""
    return LOGPROB_UNAVAILABLE if value == float("-inf") else float(value)


def build_logprobs(
    token_ids: list[int],
    token_logprobs: list[float | None],
    top_logprobs: list[list[tuple[int, float]]] | None,
    req: LogprobsRequest,
    decode_one: Callable[[int], str],
) -> dict:
    """Assemble ``choices[].logprobs`` for the tokens of ``message.content``.

    ``token_ids`` are the content tokens in order -- callers must already have
    dropped any reasoning segment (see :func:`content_token_slice`).
    ``token_logprobs[i]`` is the log probability of ``token_ids[i]``.
    ``top_logprobs[i]`` is that position's candidate list, longest-first; it is
    truncated to ``req.top_n`` here so a decode node may return more than asked.

    ``refusal`` is always ``None``: this path produces no refusal channel, and
    the field is documented as nullable.
    """
    if len(token_logprobs) != len(token_ids):
        raise ValueError(
            f"token_logprobs has {len(token_logprobs)} entries for " f"{len(token_ids)} tokens"
        )
    if top_logprobs is not None and len(top_logprobs) != len(token_ids):
        raise ValueError(
            f"top_logprobs has {len(top_logprobs)} entries for " f"{len(token_ids)} tokens"
        )

    content = []
    for i, tid in enumerate(token_ids):
        item = _entry(tid, token_logprobs[i], decode_one)
        alts = [] if top_logprobs is None else top_logprobs[i][: req.top_n]
        item["top_logprobs"] = [_entry(alt_id, alt_lp, decode_one) for alt_id, alt_lp in alts]
        content.append(item)

    return {"content": content, "refusal": None}
