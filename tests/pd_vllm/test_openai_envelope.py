"""The response envelope must match what an OpenAI client expects to read.

Three gaps, all of them things a client reads unconditionally:

* ``usage.total_tokens`` was sent on the streaming path and omitted on the
  non-streaming one, so the same deployment answered a client's
  ``usage.total_tokens`` with a number or a KeyError depending on ``stream``.
* ``usage.prompt_tokens`` is copied from the prefill response, where it can be
  absent — and ``null`` violates the contract as surely as a missing key.
* ``created`` was read from the clock per chunk, so one streamed response carried
  several timestamps. vLLM threads a single ``created_time`` through every chunk;
  a client that groups or de-duplicates by ``(id, created)`` needs that.

Shapes are taken from vLLM rather than from the spec alone, because that is the
thing this endpoint stands in for: ``UsageInfo`` declares all three fields as
integers, and non-streaming responses go out as
``JSONResponse(content=result.model_dump())`` with no ``exclude_none``, so a
declared-but-unset field appears as null. Streaming is the opposite — chunks use
``model_dump_json(exclude_unset=True)``, so an unset field is omitted there.

CPU only -- no GPU, no tilert, no real vLLM.

Run:
  CUDA_VISIBLE_DEVICES= python -m pytest \
      tests/pd_vllm/test_openai_envelope.py -v
"""

import pytest
from fastapi.testclient import TestClient

from tilert.pd_vllm import pd_router, presentation
from tilert.pd_vllm.pd_router import build_usage

# --------------------------------------------------------------------------- #
# build_usage: vLLM's UsageInfo shape
# --------------------------------------------------------------------------- #


class _Stream:
    """The two facts `blocking_choice` reads off a reply stream."""

    stop_reason = None
    completion_tokens = 3

    def finish_reason(self, from_node):
        return from_node


def test_usage_carries_all_three_fields():
    """``UsageInfo`` declares prompt_tokens, completion_tokens and total_tokens.

    Omitting the total is what made the two paths disagree.
    """
    assert set(build_usage(7, 3)) == {"prompt_tokens", "completion_tokens", "total_tokens"}


def test_the_total_is_the_sum():
    u = build_usage(7, 3)
    assert (u["prompt_tokens"], u["completion_tokens"], u["total_tokens"]) == (7, 3, 10)


@pytest.mark.parametrize("prompt", [None, 0, "5"])
def test_prompt_tokens_is_always_an_integer(prompt):
    """It is copied from the prefill response, which may not carry it. ``null``

    breaks a client arithmetic-ing over usage just as a missing key does.
    """
    u = build_usage(prompt, 3)
    assert isinstance(u["prompt_tokens"], int)
    assert isinstance(u["total_tokens"], int)


def test_an_absent_prompt_count_is_zero_not_null():
    u = build_usage(None, 4)
    assert u["prompt_tokens"] == 0
    assert u["total_tokens"] == 4


# --------------------------------------------------------------------------- #
# The two paths agree, over real HTTP
# --------------------------------------------------------------------------- #
class _Resp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


PREFILL = {
    "id": "cmpl-env",
    "choices": [{"logprobs": {"content": [{"token": "token_id:7"}]}}],
    "usage": {"prompt_tokens": 5},
    "model": "m",
}
DECODED = {"rid": "x", "token_ids": [7, 8], "seq_len": 8, "timing_ms": {"finish_reason": "stop"}}
BODY = {"messages": [{"role": "user", "content": "hi"}]}


class _StubTokenizer:
    _VOCAB = {7: "hi", 8: " there"}

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self._VOCAB.get(i, "") for i in ids)


def _client(monkeypatch):
    monkeypatch.setattr(
        pd_router.requests,
        "get",
        lambda url, timeout=None, **kw: _Resp(
            {"capabilities": {"penalties": True, "ignore_eos": True}}
        ),
    )

    def fake_post(url, json=None, timeout=None, **kw):
        return _Resp(DECODED if url.endswith("/pd/decode") else PREFILL)

    monkeypatch.setattr(pd_router.requests, "post", fake_post)
    pool = pd_router.Pool([pd_router.DecodeNode("127.0.0.1", 5556, 5557)])
    ctx = pd_router.RouterCtx(
        "http://vllm.invalid", pool, tokenizer=_StubTokenizer(), parser_name="none"
    )
    return TestClient(pd_router.build_app(ctx))


def test_non_streaming_usage_has_a_total(monkeypatch):
    r = _client(monkeypatch).post("/v1/chat/completions", json=BODY)
    assert r.status_code == 200, r.text
    usage = r.json()["usage"]
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


def test_non_streaming_usage_fields_are_integers(monkeypatch):
    r = _client(monkeypatch).post("/v1/chat/completions", json=BODY)
    usage = r.json()["usage"]
    for field, value in usage.items():
        assert isinstance(value, int), f"{field} is {type(value).__name__}"


def test_usage_survives_a_prefill_that_reports_none(monkeypatch):
    """The prefill response is another service's output; it may omit usage."""
    monkeypatch.setattr(
        pd_router.requests, "get", lambda url, timeout=None, **kw: _Resp({"capabilities": {}})
    )

    def fake_post(url, json=None, timeout=None, **kw):
        if url.endswith("/pd/decode"):
            return _Resp(DECODED)
        return _Resp({**PREFILL, "usage": None})

    monkeypatch.setattr(pd_router.requests, "post", fake_post)
    pool = pd_router.Pool([pd_router.DecodeNode("127.0.0.1", 5556, 5557)])
    ctx = pd_router.RouterCtx(
        "http://vllm.invalid", pool, tokenizer=_StubTokenizer(), parser_name="none"
    )
    client = TestClient(pd_router.build_app(ctx))
    usage = client.post("/v1/chat/completions", json=BODY).json()["usage"]
    assert usage["prompt_tokens"] == 0
    assert usage["total_tokens"] == usage["completion_tokens"]


# --------------------------------------------------------------------------- #
# Declared-but-null fields on the non-streaming choice
# --------------------------------------------------------------------------- #
def test_logprobs_is_declared_null_when_not_requested(monkeypatch):
    """vLLM's non-streaming response is dumped without ``exclude_none``, so a
    client can read ``choices[0].logprobs`` unconditionally.
    """
    r = _client(monkeypatch).post("/v1/chat/completions", json=BODY)
    choice = r.json()["choices"][0]
    assert "logprobs" in choice and choice["logprobs"] is None


def test_stop_reason_is_declared(monkeypatch):
    """vLLM carries it for legacy reasons, and it is how a client tells "stopped
    on a stop string" from "stopped on EOS".

    Always null here: the stop strings and stop_token_ids that would populate it
    are refused by the capability gate, so there is nothing it could name. The
    field is present rather than absent so the distinction is readable at all.
    """
    r = _client(monkeypatch).post("/v1/chat/completions", json=BODY)
    choice = r.json()["choices"][0]
    assert "stop_reason" in choice and choice["stop_reason"] is None


def test_a_requested_logprobs_still_wins_over_the_null_default(monkeypatch):
    """The null is a default, not an override."""

    def fake_post(url, json=None, timeout=None, **kw):
        if url.endswith("/pd/decode"):
            return _Resp({**DECODED, "logprobs": {"lp": [-0.5, -0.6], "tp": [[], []]}})
        return _Resp(PREFILL)

    monkeypatch.setattr(
        pd_router.requests,
        "get",
        lambda url, timeout=None, **kw: _Resp(
            {"capabilities": {"penalties": True, "ignore_eos": True}}
        ),
    )
    monkeypatch.setattr(pd_router.requests, "post", fake_post)
    pool = pd_router.Pool([pd_router.DecodeNode("127.0.0.1", 5556, 5557)])
    ctx = pd_router.RouterCtx(
        "http://vllm.invalid", pool, tokenizer=_StubTokenizer(), parser_name="none"
    )
    r = TestClient(pd_router.build_app(ctx)).post(
        "/v1/chat/completions",
        json={**BODY, "logprobs": True, "top_logprobs": 0, "temperature": 0.6},
    )
    assert r.status_code == 200, r.text
    assert r.json()["choices"][0]["logprobs"] is not None


# --------------------------------------------------------------------------- #
# created is stamped once per response
# --------------------------------------------------------------------------- #
def test_created_is_an_integer_timestamp(monkeypatch):
    body = _client(monkeypatch).post("/v1/chat/completions", json=BODY).json()
    assert isinstance(body["created"], int)


def test_no_path_reads_the_clock_per_chunk():
    """Source-level, because reproducing a multi-second stream to catch a
    one-second drift would be a slow and flaky test for a property that is
    plainly visible in the code: vLLM threads a single ``created_time`` through
    every chunk, and each ``int(time.time())`` inside a chunk builder is a
    response that can carry more than one timestamp.
    """
    import pathlib

    src = pathlib.Path(pd_router.__file__).read_text()
    # One assignment is expected (the per-response stamp); a call inside a chunk
    # payload is not.
    for marker in ('"created": int(time.time())', '"created": int(time.time()), "model"'):
        assert marker not in src, f"a chunk builder still reads the clock: {marker}"
    assert src.count("int(time.time())") <= 1, (
        "more than one clock read: each is a chance for one response to carry " "two timestamps"
    )


@pytest.mark.parametrize(
    "choice_of,kw,why",
    [
        (
            presentation.textless_choice,
            {"from_node": "length"},
            "no tokenizer: ids are all the reply can carry",
        ),
        (
            presentation.blocking_choice,
            {"from_node": "length", "logprobs_asked": False},
            "with a tokenizer: alongside the text",
        ),
    ],
)
def test_completions_keeps_token_ids(choice_of, kw, why):
    """``token_ids`` is NOT a non-standard wart to remove.

    vLLM declares it on its own choice model — "not part of the OpenAI spec but
    is useful for tracing the tokens in agent scenarios" — so carrying it on
    /v1/completions matches the thing this endpoint stands in for.

    Asserted on the field, not on the source line that sets it: the grep this
    replaces failed when the line moved to `presentation.py` unchanged.
    """
    if choice_of is presentation.blocking_choice:
        kw["stream"] = _Stream()
        kw["got"] = presentation.collect([])
    choice, _ = choice_of(is_chat=False, token_ids=[5, 6, 7], **kw)
    assert choice["token_ids"] == [5, 6, 7], why
    chat, _ = choice_of(is_chat=True, token_ids=[5, 6, 7], **dict(kw))
    assert "token_ids" not in chat, "chat carries the message instead"
