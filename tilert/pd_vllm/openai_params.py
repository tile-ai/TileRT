"""Shared reading of the OpenAI request fields every backend accepts.

All backends expose the same OpenAI-compatible ``/v1/chat/completions``
surface (see ``docs/architecture.md``), so the rules for interpreting a client
body belong in exactly one place rather than once per subpackage. Nothing here
imports a backend; it is plain dict handling.
"""

from __future__ import annotations

from pydantic import TypeAdapter, ValidationError

__all__ = ["InvalidOutputLength", "resolve_max_tokens"]

# Two orders, because vLLM applies two.
#
# Declaration order on ``ChatCompletionRequest``
# (v0.24.0 protocol.py:202, 207).
# Pydantic validates every field and reports in this order, so a body wrong in
# both names is answered about ``max_tokens``.
_DECLARED = ("max_tokens", "max_completion_tokens")
# Precedence, applied after validation and on ``is not None`` -- NOT on
# truthiness, so a preferred ``0`` does not defer to the other name
# (v0.24.0 chat_completion/serving.py:302-306).
_PRECEDENCE = ("max_completion_tokens", "max_tokens")

_INT = TypeAdapter(int)


class InvalidOutputLength(ValueError):
    """The client's output length cannot be honoured; each edge maps it to 400.

    A ``ValueError`` subclass so a caller that already guards this call keeps
    working.
    """


def resolve_max_tokens(body: dict, default: int | None = None) -> int | None:
    """The output length for this request, or ``default`` if none was asked for.

    The single owner of this field. It used to own the precedence only, leaving
    coercion to whichever ``int()`` call happened to run first and the refusal
    to nobody -- so ``1.9`` was truncated to 1 on one path while ``"20.0"``
    raised on another, after the prefill had already run.

    Coercion is delegated to the request model's own validator rather than
    restated: ``max_tokens`` is a plain ``int | None`` on vLLM's
    ``ChatCompletionRequest``, so pydantic's lax rules ARE the specification.
    ``True`` is 1, ``"20"`` / ``" 20 "`` / ``"20.0"`` coerce, ``"1e3"`` and
    ``1.9`` and ``inf`` do not, and ``10**309`` is a valid integer that
    ``float()`` cannot hold. That table has no guessable edges, which is why
    every hand-written version of it has been wrong somewhere.

    Both names are coerced but only the effective one is range-checked, because
    vLLM's two layers have different reach: pydantic validates every declared
    field before the precedence is applied, while ``_verify_args`` then sees
    one resolved number. So ``{max_completion_tokens: 3, max_tokens: 1.9}`` is
    refused and ``{max_completion_tokens: 16, max_tokens: 0}`` is served.

    ``default`` is an argument because it is a property of the backend, not of
    the request: the PD decode node uses 256, ``serve_native`` 4096. Omitting
    it asks the question without answering it -- what the CLIENT requested,
    which is what an edge validating before it has a backend in hand needs.

    Pure and total, so calling it more than once per request is a repeated
    computation and not a second opinion. That is the property the two call
    sites lacked when each did its own coercion.

    Raises:
        InvalidOutputLength: unusable type or value; the edge answers 400.
    """
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
