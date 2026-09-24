from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import TypeAdapter, ValidationError

from tilert.pd_vllm.capabilities import CapabilityUnavailable, InvalidParameter
from tilert.pd_vllm.grammar_spec import extract_request_grammar_spec
from tilert.pd_vllm.logprobs import LogprobsRequest, LogprobsUnsupported, resolve_logprobs_request
from tilert.pd_vllm.openai_params import InvalidOutputLength, resolve_max_tokens
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
    is_chat: bool
    thinking: bool
    stop: list[str] = field(default_factory=list)
    include_stop: bool = False
    logprobs_req: LogprobsRequest | None = None
    grammar_spec: dict | None = None


def gate_request(path: str, body: dict, *, tokenizer, parser_active) -> GatedRequest:
    is_chat = path.endswith("chat/completions")
    grammar_spec = extract_request_grammar_spec(body)
    try:
        resolve_max_tokens(body)
    except InvalidOutputLength as e:
        raise InvalidParameter(str(e)) from None
    logprobs_req = _logprobs_of(path, body)
    require_tokenizer_for_logprobs(logprobs_req, tokenizer)
    stop, include_stop = resolve_stop_request(body, tokenizer)
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
    ctk = body.get("chat_template_kwargs")
    if ctk is None:
        return True
    if not isinstance(ctk, dict):
        raise InvalidParameter(f"chat_template_kwargs must be an object, got {type(ctk).__name__}")
    return bool(ctk.get("enable_thinking", True))


def _logprobs_of(path: str, body: dict):
    if path.endswith("chat/completions"):
        return resolve_logprobs_request(body)
    if body.get("logprobs") is not None:
        raise LogprobsUnsupported("logprobs is supported on /v1/chat/completions")
    return None


def refuse_unattributable_logprobs(logprobs_req, stop, has_parser) -> None:
    if logprobs_req is not None and stop and has_parser:
        raise CapabilityUnavailable(
            "logprobs cannot be attributed to message.content when stop strings and an output parser are both in play: the parser buffers across the boundary the stop hold-back creates, so some tokens' channel is undecidable. Drop one of stop / logprobs, or use a router started with --parser none."
        )


def require_tokenizer_for_logprobs(logprobs_req, tokenizer) -> None:
    if logprobs_req is not None and tokenizer is None:
        raise CapabilityUnavailable(
            "logprobs need a tokenizer to name each token, and this router has none: start it with --model-path to serve them."
        )


def resolve_stop_request(body: dict, tokenizer) -> tuple[list[str], bool]:
    try:
        stop = resolve_stop(body)
    except ValueError as e:
        raise InvalidParameter(str(e)) from e
    include = False
    if "include_stop_str_in_output" in body:
        try:
            include = TypeAdapter(bool).validate_python(body["include_stop_str_in_output"])
        except ValidationError:
            raise InvalidParameter(
                f"include_stop_str_in_output must be a boolean, got {body['include_stop_str_in_output']!r}"
            ) from None
    if stop and tokenizer is None:
        raise CapabilityUnavailable(
            "stop strings need a tokenizer to match against the reply text, and this router has none: matching is text-level, so the decode node's token-id stop set cannot serve it."
        )
    return (stop, include)
