"""What is decided about a request before anything observable happens.

Called before the vLLM prefill request, before a decode node is acquired, before
the connector claims anything, so a refused request costs nothing and holds
nothing. Both response paths ran an identical copy of these six checks; the
copies drifted twice under review.

Deliberately NOT here: ``validate_generation_request``, which needs the pool's
capabilities. That probe talks HTTP to every uncached node and blocks per wedged
one, while nothing below depends on it -- so the caller runs it after this
returns, on whichever thread discipline it has (a direct call, or a threadpool
off the event loop). The order is the point: a 400 for a malformed request must
not wait on an unreachable node.

Free of HTTP and asyncio: a dict in, a :class:`GatedRequest` or an exception out.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import TypeAdapter, ValidationError

from tilert.pd_vllm.capabilities import (
    CapabilityUnavailable,
    InvalidParameter,
)
from tilert.pd_vllm.grammar_spec import extract_request_grammar_spec
from tilert.pd_vllm.logprobs import (
    LogprobsRequest,
    LogprobsUnsupported,
    resolve_logprobs_request,
)
from tilert.pd_vllm.openai_params import (
    InvalidOutputLength,
    resolve_max_tokens,
)
from tilert.pd_vllm.stop_strings import resolve_stop

__all__ = [
    "GatedRequest",
    "gate_request",
    "refuse_unattributable_logprobs",
    "require_tokenizer_for_logprobs",
    "resolve_stop_request",
]


@dataclass(frozen=True)
class GatedRequest:
    """An accepted request, as the handlers need it.

    Every field is derived from the body, so a handler never re-reads it for the
    same decision -- `enable_thinking` was read three times per request, and the
    two paths disagreed about whether `/v1/completions` has a parser at all.
    """

    is_chat: bool
    thinking: bool
    stop: list[str] = field(default_factory=list)
    include_stop: bool = False
    logprobs_req: LogprobsRequest | None = None
    grammar_spec: dict | None = None


def gate_request(path: str, body: dict, *, tokenizer, parser_active) -> GatedRequest:
    """Accept the request or raise, without touching a backend.

    ``parser_active`` answers "would a parser run for this request", given the
    thinking flag: it decides one refusal and is a router-configuration
    question, not a request one.

    Raises:
        InvalidParameter: unusable type or value (400).
        CapabilityUnavailable: valid request, nothing to execute it (501).
        LogprobsUnsupported: logprobs asked for where they are not served (400).
        GrammarError: unusable constrained-decoding spec.
    """
    is_chat = path.endswith("chat/completions")
    grammar_spec = extract_request_grammar_spec(body)
    # Resolved here only to fail early: the router pins the prefill leg to
    # max_tokens=1 and strips both client names from it, so an unusable length
    # would otherwise surface from `_decode_body` with the prefill already
    # spent. The value itself is taken there, from the same function.
    try:
        resolve_max_tokens(body)
    except InvalidOutputLength as e:
        raise InvalidParameter(str(e)) from None
    logprobs_req = _logprobs_of(path, body)
    require_tokenizer_for_logprobs(logprobs_req, tokenizer)
    stop, include_stop = resolve_stop_request(body, tokenizer)
    # Chat only, and in this position, because both are vLLM's behaviour:
    # `ChatCompletionRequest` types the field and answers 422 for a string,
    # while `CompletionRequest` has no such field and `extra="allow"` accepts
    # and ignores it (measured on 0.25.1). Validating it for /v1/completions
    # would refuse a request vLLM serves; validating it before `stop` would
    # change which of two bad fields is reported.
    thinking = _thinking_enabled(body) if is_chat else True
    refuse_unattributable_logprobs(logprobs_req, stop, is_chat and parser_active(thinking))
    return GatedRequest(
        is_chat=is_chat,
        thinking=thinking,
        stop=stop,
        include_stop=include_stop,
        logprobs_req=logprobs_req,
        grammar_spec=grammar_spec,
    )


def _thinking_enabled(body: dict) -> bool:
    """Whether the request asked for the model's thinking segment.

    Raises:
        InvalidParameter: ``chat_template_kwargs`` present and not an object
            (400). vLLM declares it ``dict[str, Any] | None`` and answers 422 for
            anything else; reading `.get` off a string here raised AttributeError
            instead, which surfaced as a 500 for a client mistake.
    """
    ctk = body.get("chat_template_kwargs")
    if ctk is None:
        return True
    if not isinstance(ctk, dict):
        raise InvalidParameter(
            f"chat_template_kwargs must be an object, got " f"{type(ctk).__name__}"
        )
    return bool(ctk.get("enable_thinking", True))


def _logprobs_of(path: str, body: dict):
    """Validated logprobs request for this path, or None.

    Chat only. ``/v1/completions`` takes a differently-typed ``logprobs`` (a
    count, not a flag) and is not served here, so asking for it there is
    rejected with the same shape as the existing streaming rejection rather than
    being ignored.
    """
    if path.endswith("chat/completions"):
        return resolve_logprobs_request(body)
    if body.get("logprobs") is not None:
        raise LogprobsUnsupported("logprobs is supported on /v1/chat/completions")
    return None


def refuse_unattributable_logprobs(logprobs_req, stop, has_parser) -> None:
    """Refuse the one request shape whose logprob attribution has no answer.

    A ``stop`` makes the router hold text back, so what reaches the parser spans
    token boundaries and some tokens' channel becomes undecidable. Every other
    combination is served; ``reply``'s module docstring has the four cases and
    why this one has no answer rather than an expensive one.

    Raises:
        CapabilityUnavailable: all three asked for at once (501).
    """
    if logprobs_req is not None and stop and has_parser:
        raise CapabilityUnavailable(
            "logprobs cannot be attributed to message.content when stop "
            "strings and an output parser are both in play: the parser buffers "
            "across the boundary the stop hold-back creates, so some tokens' "
            "channel is undecidable. Drop one of stop / logprobs, or use a "
            "router started with --parser none."
        )


def require_tokenizer_for_logprobs(logprobs_req, tokenizer) -> None:
    """Refuse a logprobs request this router cannot name tokens for.

    Every entry carries the token's text, and ``--parser none`` without
    ``--model-path`` is a supported configuration with no tokenizer. 200 with
    ``logprobs: null`` would report success for a field the client asked for and
    did not get, after both backends computed the values.

    Raises:
        CapabilityUnavailable: 501.
    """
    if logprobs_req is not None and tokenizer is None:
        raise CapabilityUnavailable(
            "logprobs need a tokenizer to name each token, and this router has "
            "none: start it with --model-path to serve them."
        )


def resolve_stop_request(body: dict, tokenizer) -> tuple[list[str], bool]:
    """The request's stop strings and whether to keep them, or ``([], False)``.

    Not in the capability gate, because executing them needs a tokenizer and
    that is a router-side resource.

    Raises:
        InvalidParameter: unusable type or value (400).
        CapabilityUnavailable: asked for with no tokenizer (501).
    """
    try:
        stop = resolve_stop(body)
    except ValueError as e:
        raise InvalidParameter(str(e)) from e

    include = False
    if "include_stop_str_in_output" in body:
        # pydantic is the validator vLLM's request model uses, so the accepted
        # set is identical by construction. Measured on 2.13.4: `1`, `0` and
        # `"true"` accepted (refusing them would turn a working request into a
        # 400), explicit `null` refused (vLLM answers 422, and the router strips
        # the field before prefill so vLLM never gets to).
        try:
            include = TypeAdapter(bool).validate_python(body["include_stop_str_in_output"])
        except ValidationError:
            raise InvalidParameter(
                f"include_stop_str_in_output must be a boolean, got "
                f"{body['include_stop_str_in_output']!r}"
            ) from None
    # No check for `include` without `stop`: with no stop strings the flag is a
    # no-op, and vLLM serves such a request. Refusing it would turn a request
    # that works against a native endpoint into a 400 here.

    if stop and tokenizer is None:
        raise CapabilityUnavailable(
            "stop strings need a tokenizer to match against the reply text, "
            "and this router has none: matching is text-level, so the decode "
            "node's token-id stop set cannot serve it."
        )
    return stop, include
