"""What the gate accepts, refuses, and refuses to look at.

`gate_request` is a pure function, so these run without HTTP, a tokenizer or a
backend. What they pin is not that the checks exist -- other suites cover each
one -- but the two things a single gate can silently get wrong: WHICH fields it
validates, and IN WHICH ORDER, since only one error reaches the client.

Both were found by a differential against the pre-refactor router rather than by
reasoning: the same request matrix through both, byte-compared.

  CUDA_VISIBLE_DEVICES= python -m pytest tests/pd_vllm/test_request_gate.py -v
"""

from __future__ import annotations

import pytest

from tilert.pd_vllm.capabilities import (
    CapabilityUnavailable,
    InvalidParameter,
)
from tilert.pd_vllm.request_gate import gate_request

CHAT = "/v1/chat/completions"
COMP = "/v1/completions"


class _Tok:
    def decode(self, ids, skip_special_tokens=False):
        return ""


_TOK = _Tok()


def gate(path, body, tokenizer=_TOK, parser=False):
    return gate_request(path, body, tokenizer=tokenizer, parser_active=lambda thinking: parser)


# --------------------------------------------------------------------------- #
# chat_template_kwargs: a chat field, validated only there
# --------------------------------------------------------------------------- #
def test_a_non_dict_chat_template_kwargs_is_refused_on_chat():
    """vLLM types the field `dict[str, Any] | None` on `ChatCompletionRequest`

    and answers 422 for a string. Reading `.get` off one raised AttributeError
    here once, which surfaced as a 500 for a client mistake.
    """
    with pytest.raises(InvalidParameter):
        gate(CHAT, {"chat_template_kwargs": "oops"})


@pytest.mark.parametrize(
    "extra,why",
    [
        ({}, "no grammar: the field reaches nothing at all"),
        (
            {"response_format": {"type": "json_object"}},
            "with a grammar the node gets `enable_thinking`, and True is what the "
            "absent field would have meant",
        ),
    ],
)
def test_the_same_value_is_served_on_completions(extra, why):
    """Measured on vLLM 0.25.1: `CompletionRequest` has no
    `chat_template_kwargs` field and `extra="allow"`, so it accepts and ignores
    one. Refusing it here would 400 a request vLLM serves.

    The pre-refactor router served it without a grammar and answered 502 WITH
    one -- it raised inside the handler, after a node was acquired and prefill
    had run, and a generic handler flattened it. Both now serve.
    """
    req = gate(COMP, {"chat_template_kwargs": "oops", **extra})
    assert req.thinking is True, why
    assert req.is_chat is False


def test_thinking_is_read_once_and_carried():
    """It decided the parser session and the decode body separately before, from
    two reads of the same body.
    """
    assert gate(CHAT, {"chat_template_kwargs": {"enable_thinking": False}}).thinking is False
    assert gate(CHAT, {}).thinking is True


# --------------------------------------------------------------------------- #
# order: only one error reaches the client
# --------------------------------------------------------------------------- #
def test_an_unusable_stop_is_reported_before_an_unusable_template_kwarg():
    """Two bad fields, one answer.

    The order is the pre-refactor order, kept because a client reading the message would
    otherwise see a different one after a refactor that was supposed to change nothing.
    """
    with pytest.raises(InvalidParameter) as e:
        gate(CHAT, {"stop": [""], "chat_template_kwargs": "oops"})
    assert "stop" in str(e.value), str(e.value)


def test_a_grammar_is_reported_before_anything_else():
    from tilert.pd_vllm.grammar_spec import GrammarError

    with pytest.raises(GrammarError):
        gate(CHAT, {"response_format": "not an object", "stop": [""]})


# --------------------------------------------------------------------------- #
# the refusals themselves, at the gate rather than after backend work
# --------------------------------------------------------------------------- #
def test_stop_without_a_tokenizer_is_refused():
    with pytest.raises(CapabilityUnavailable):
        gate(CHAT, {"stop": ["END"]}, tokenizer=None)


def test_logprobs_without_a_tokenizer_is_refused():
    with pytest.raises(CapabilityUnavailable):
        gate(CHAT, {"logprobs": True, "temperature": 0.6}, tokenizer=None)


def test_stop_with_logprobs_and_a_parser_is_refused():
    with pytest.raises(CapabilityUnavailable):
        gate(CHAT, {"stop": ["END"], "logprobs": True, "temperature": 0.6}, parser=True)


def test_the_same_three_without_the_parser_are_served():
    req = gate(CHAT, {"stop": ["END"], "logprobs": True, "temperature": 0.6}, parser=False)
    assert req.stop == ["END"]
    assert req.logprobs_req is not None


def test_the_gate_does_not_probe_capabilities():
    """`validate_generation_request` stays at the call sites on purpose: its
    probe blocks per unreachable node, and nothing here depends on it, so a 400
    for a malformed request must not wait on one.

    Asserted as an interface fact -- the gate takes no capabilities argument --
    because the cost of getting it wrong is latency nothing would measure.
    """
    import inspect

    params = inspect.signature(gate_request).parameters
    assert set(params) == {"path", "body", "tokenizer", "parser_active"}
