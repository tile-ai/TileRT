from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "CapabilityError",
    "CapabilityUnavailable",
    "InvalidParameter",
    "NodeCapabilities",
    "PROFILE_FIELD_NAMES",
    "STATIC_FIELD_NAMES",
    "TYPED_FIELD_NAMES",
    "engine_capabilities",
    "validate_generation_request",
]


class CapabilityError(Exception):
    error_type = "capability_error"
    http_status = 500

    def to_payload(self) -> dict[str, str]:
        return {"error": str(self), "error_type": self.error_type}


class InvalidParameter(CapabilityError):
    error_type = "invalid_parameter"
    http_status = 400


class CapabilityUnavailable(CapabilityError):
    error_type = "capability_unavailable"
    http_status = 501


@dataclass(frozen=True)
class NodeCapabilities:
    penalties: bool = False
    ignore_eos: bool = False

    def intersect(self, other: NodeCapabilities) -> NodeCapabilities:
        return NodeCapabilities(
            penalties=self.penalties and other.penalties,
            ignore_eos=self.ignore_eos and other.ignore_eos,
        )

    def to_payload(self) -> dict[str, bool]:
        return {"penalties": self.penalties, "ignore_eos": self.ignore_eos}

    @classmethod
    def from_payload(cls, payload: object) -> NodeCapabilities:
        if not isinstance(payload, dict):
            return cls()
        caps = payload.get("capabilities", payload)
        if not isinstance(caps, dict):
            return cls()
        return cls(
            penalties=caps.get("penalties") is True, ignore_eos=caps.get("ignore_eos") is True
        )


def engine_capabilities(engine: object) -> NodeCapabilities:

    def _ask(name: str) -> bool:
        probe = getattr(engine, name, None)
        if not callable(probe):
            return False
        try:
            return bool(probe())
        except Exception:
            return False

    return NodeCapabilities(
        penalties=_ask("supports_penalties"), ignore_eos=_ask("supports_ignore_eos")
    )


_EMPTY, _UNSET, _NUMBER, _FLAG, _COUNT = ("empty", "unset", "number", "flag", "count")
_STATIC_FIELDS: dict[str, tuple] = {
    "stop_token_ids": (
        _EMPTY,
        None,
        None,
        "the decode loop uses the model's own stop set and accepts no per-request ids",
    ),
    "min_tokens": (
        _COUNT,
        0,
        0,
        "the decode loop cannot suppress its stop set for a minimum length",
    ),
    "frequency_penalty": (
        _NUMBER,
        0.0,
        None,
        "the decode sampler implements repetition and presence penalties only",
    ),
    "min_p": (_NUMBER, 0.0, None, "the decode sampler implements top-p and top-k only"),
    "seed": (
        _UNSET,
        None,
        None,
        "the decode sampler's seed is per-process, so a per-request seed cannot make the reply reproducible",
    ),
    "logit_bias": (_EMPTY, None, None, "the decode sampler has no per-request logit bias"),
    "bad_words": (_EMPTY, None, None, "the decode loop has no bad-words matcher"),
    "allowed_token_ids": (_EMPTY, None, None, "the decode sampler has no per-request allow list"),
    "structured_outputs": (
        _EMPTY,
        None,
        None,
        "use response_format, which this stack translates into a decode-side grammar",
    ),
    "n": (
        _COUNT,
        1,
        1,
        "one prefill KV state is transferred per request, so the decode node produces exactly one sequence",
    ),
    "best_of": (
        _COUNT,
        1,
        1,
        "the decode node produces exactly one sequence, so there is nothing to select from",
    ),
    "use_beam_search": (_FLAG, False, None, "the decode node runs single-sequence AR/MTP decode"),
    "prompt_logprobs": (
        _UNSET,
        None,
        None,
        "the prefill instance is asked for one token, so no prompt distribution is collected",
    ),
    "logprob_token_ids": (
        _EMPTY,
        None,
        None,
        "the decode export returns the top candidates, not a caller-chosen vocab subset",
    ),
    "skip_special_tokens": (
        _FLAG,
        True,
        None,
        "the reply is detokenised for the output parser, which always consumes special tokens",
    ),
}
_PROFILE_FIELDS: dict[str, tuple] = {
    "repetition_penalty": (
        _NUMBER,
        1.0,
        "penalties",
        "this model's decode runtime has no penalty pre-pass",
    ),
    "presence_penalty": (
        _NUMBER,
        0.0,
        "penalties",
        "this model's decode runtime has no penalty pre-pass",
    ),
    "ignore_eos": (
        _FLAG,
        False,
        "ignore_eos",
        "this model's decode loop does not clear its stop set",
    ),
}
_TYPED_FIELDS: dict[str, tuple] = {
    "temperature": (_NUMBER, 0.0),
    "top_p": (_NUMBER, 0.0),
    "top_k": (_COUNT, None),
}
STATIC_FIELD_NAMES = tuple(_STATIC_FIELDS)
PROFILE_FIELD_NAMES = tuple(_PROFILE_FIELDS)
TYPED_FIELD_NAMES = tuple(_TYPED_FIELDS)
_GREEDY_TEMPERATURE = 1e-05


def _is_greedy(body: dict) -> bool:
    raw = body.get("temperature")
    if raw is None:
        return False
    try:
        return _lenient_float("temperature", raw) < _GREEDY_TEMPERATURE
    except InvalidParameter:
        return False


def _as_number(field: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidParameter(f"{field} must be a number, got {type(value).__name__}")
    return float(value)


def _as_flag(field: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise InvalidParameter(f"{field} must be a boolean, got {type(value).__name__}")
    return value


def _as_count(field: str, value: object, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidParameter(f"{field} must be an integer, got {type(value).__name__}")
    if value < minimum:
        raise InvalidParameter(f"{field} must be >= {minimum}, got {value}")
    return value


def _lenient_float(field: str, raw: object) -> float:
    if isinstance(raw, bool):
        raise InvalidParameter(f"{field} must be a number, got bool")
    try:
        return float(raw)
    except (TypeError, ValueError):
        raise InvalidParameter(f"{field} must be a number, got {raw!r}") from None


def _lenient_int(field: str, raw: object) -> int:
    value = _lenient_float(field, raw)
    if value != int(value):
        raise InvalidParameter(f"{field} must be an integer, got {raw!r}")
    return int(value)


def _is_neutral(field: str, value: object, kind: str, neutral, minimum: int | None) -> bool:
    if value is None:
        return True
    if kind == _EMPTY:
        return not value
    if kind == _UNSET:
        return False
    if kind == _NUMBER:
        return _as_number(field, value) == neutral
    if kind == _FLAG:
        return _as_flag(field, value) is neutral
    return _as_count(field, value, minimum if minimum is not None else 0) == neutral


def validate_generation_request(
    body: dict, capabilities: NodeCapabilities | None = None, adopted: dict | None = None
) -> None:
    for field, (kind, minimum) in _TYPED_FIELDS.items():
        if body.get(field) is None:
            continue
        if kind == _NUMBER:
            value = _lenient_float(field, body[field])
            if minimum is not None and value < minimum:
                raise InvalidParameter(f"{field} must be >= {minimum}, got {value}")
        else:
            _lenient_int(field, body[field])
    greedy = _is_greedy(body)
    for field, (kind, neutral, minimum, why) in _STATIC_FIELDS.items():
        if field not in body:
            continue
        if _is_neutral(field, body[field], kind, neutral, minimum):
            continue
        if field == "seed" and greedy:
            continue
        raise CapabilityUnavailable(
            f"{field} is not supported by the TileRT decode stage: {why}. The vLLM prefill instance would apply it to the first token and the decode node would ignore it for the rest of the reply, so the request is refused instead of served incorrectly."
        )
    caps = capabilities or NodeCapabilities()
    for field, (kind, neutral, attr, why) in _PROFILE_FIELDS.items():
        if field in body:
            value, origin = (body[field], "the request")
        elif adopted is not None and field in adopted:
            value, origin = (adopted[field], "this deployment's defaults")
        else:
            continue
        if _is_neutral(field, value, kind, neutral, None):
            continue
        if getattr(caps, attr):
            continue
        raise CapabilityUnavailable(
            f"{field} (from {origin}) is not supported by the decode node serving this pool: {why}. It would apply to the first token only, so the request is refused instead of served incorrectly."
        )
