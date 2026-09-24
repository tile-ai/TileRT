from __future__ import annotations

from pydantic import TypeAdapter, ValidationError

__all__ = ["InvalidOutputLength", "resolve_max_tokens"]
_DECLARED = ("max_tokens", "max_completion_tokens")
_PRECEDENCE = ("max_completion_tokens", "max_tokens")
_INT = TypeAdapter(int)


class InvalidOutputLength(ValueError):
    pass


def resolve_max_tokens(body: dict, default: int | None = None) -> int | None:
    seen: dict[str, int] = {}
    for name in _DECLARED:
        raw = body.get(name)
        if raw is None:
            continue
        try:
            seen[name] = _INT.validate_python(raw)
        except ValidationError:
            raise InvalidOutputLength(f"{name} must be an integer, got {raw!r}") from None
    for name in _PRECEDENCE:
        if name in seen:
            if seen[name] < 1:
                raise InvalidOutputLength(f"{name} must be at least 1, got {seen[name]}")
            return seen[name]
    return default
