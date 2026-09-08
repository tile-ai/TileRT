"""The prefill request must not inherit client fields that fight its overrides.

``build_prefill_body`` forwards the client body verbatim apart from the handful
of fields the split needs (``max_tokens=1``, ``stream=False``, logprobs,
``kv_transfer_params``). Two client fields survive that rewrite and break it:

``stream_options``
    vLLM refuses ``stream_options`` unless ``stream`` is true --
    ``ChatCompletionRequest.validate_stream_options`` is a ``mode="before"``
    model_validator, so the request is rejected with 400 before the model is
    even looked at. We force ``stream=False``, so any client that sent
    ``stream_options`` got its prefill 400'd and the router turned that into a
    502 for the caller.

``max_completion_tokens``
    vLLM prefers it over ``max_tokens`` (``ChatCompletionRequest`` resolves
    ``max_completion_tokens`` first when both are present), so it overrides our
    ``max_tokens=1`` and the prefill instance decodes the client's whole output
    length -- the split still "works", it just stops being a split.

Both are unconditional in ``vllm bench serve --backend openai-chat``, which is
why the official benchmark failed every request against a PD deployment.

No GPU, no tilert, no vllm: ``pd_router`` imports fastapi/requests/uvicorn but
nothing that needs a device, and ``build_prefill_body`` is pure.

Run:
  CUDA_VISIBLE_DEVICES= python -m pytest \
      tests/pd_vllm/test_prefill_body_rewrite.py -v
"""

from __future__ import annotations

import pytest

from tilert.pd_vllm.decode_pool import DecodeNode
from tilert.pd_vllm.pd_router import build_prefill_body

CHAT = "/v1/chat/completions"
COMPLETIONS = "/v1/completions"


@pytest.fixture
def node() -> DecodeNode:
    return DecodeNode("10.0.0.2", 5556, 5557)


def _bench_body() -> dict:
    """What ``vllm bench serve --backend openai-chat`` puts on the wire.

    Mirrors ``vllm/benchmarks/lib/endpoint_request_func.py``: streaming with
    usage accounting, output length via ``max_completion_tokens``.
    """
    return {
        "model": "glm5.1",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.0,
        "max_completion_tokens": 3000,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


# --------------------------------------------------------------------------- #
# the two fields that must not survive
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("path", [CHAT, COMPLETIONS])
@pytest.mark.parametrize("field", ["stream_options", "max_completion_tokens"])
def test_contradicting_field_is_dropped(path, field, node) -> None:
    out = build_prefill_body(path, _bench_body(), node)
    assert field not in out


def test_stream_options_cannot_coexist_with_our_stream_false(node) -> None:
    """The exact pair vLLM rejects with 400."""
    out = build_prefill_body(CHAT, _bench_body(), node)
    assert out["stream"] is False
    assert "stream_options" not in out


def test_prefill_decodes_exactly_one_token(node) -> None:
    """max_tokens=1 must be the only output-length field left standing."""
    out = build_prefill_body(CHAT, _bench_body(), node)
    assert out["max_tokens"] == 1
    assert "max_completion_tokens" not in out


def test_max_completion_tokens_does_not_leak_into_max_tokens(node) -> None:
    """Dropping it must not be implemented by copying it over max_tokens."""
    body = _bench_body()
    body.pop("stream_options")
    out = build_prefill_body(CHAT, body, node)
    assert out["max_tokens"] == 1


@pytest.mark.parametrize("path", [CHAT, COMPLETIONS])
def test_absent_fields_need_no_special_case(path, node) -> None:
    """A plain non-streaming client must be rewritten without raising."""
    out = build_prefill_body(path, {"model": "m", "prompt": "hi", "max_tokens": 128}, node)
    assert out["max_tokens"] == 1
    assert out["stream"] is False


# --------------------------------------------------------------------------- #
# everything else about the rewrite is unchanged
# --------------------------------------------------------------------------- #


def test_chat_asks_for_top_logprobs(node) -> None:
    """The router reads the first token id out of chat logprobs."""
    out = build_prefill_body(CHAT, _bench_body(), node)
    assert out["logprobs"] is True
    assert out["top_logprobs"] == 1


def test_completions_logprobs_is_a_count(node) -> None:
    out = build_prefill_body(COMPLETIONS, {"model": "m", "prompt": "hi"}, node)
    assert out["logprobs"] == 1
    assert "top_logprobs" not in out


def test_kv_transfer_params_point_at_the_chosen_node(node) -> None:
    out = build_prefill_body(CHAT, _bench_body(), node)
    assert out["kv_transfer_params"] == {
        "tilert_host": "10.0.0.2",
        "tilert_ctrl_port": 5556,
    }


def test_unrelated_client_fields_are_forwarded(node) -> None:
    body = _bench_body()
    body["top_p"] = 0.95
    body["repetition_penalty"] = 1.1
    out = build_prefill_body(CHAT, body, node)
    assert out["top_p"] == 0.95
    assert out["repetition_penalty"] == 1.1
    assert out["messages"] == body["messages"]


def test_client_body_is_not_mutated(node) -> None:
    """The caller reuses `body` for the decode request; it must survive intact."""
    body = _bench_body()
    build_prefill_body(CHAT, body, node)
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert body["max_completion_tokens"] == 3000
