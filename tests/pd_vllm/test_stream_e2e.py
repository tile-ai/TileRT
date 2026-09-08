r"""A client `stream: true` must still stream without `stream_options` on prefill.

`build_prefill_body` strips two fields a streaming client always sends, which
invites the obvious worry: did we just turn streaming off? No -- the fields were
going to the wrong backend. Three HTTP conversations are in play and only the
middle one is forced non-streaming:

    client --stream:true--> router --stream:false--> vLLM prefill  (1 token + KV)
                                  \--stream:true--> decode node   (tokens 2..N)
       <---- SSE text/event-stream

`stream_options` is meaningful only between client and router (the client wants
usage accounting) but was being copied onto the prefill request, the one hop
that *must* be non-streaming. The client's intent is not lost: the router
satisfies `include_usage` itself in a trailing usage-only chunk, and the
client's output length reaches the decode node where it belongs.

The other tests in this directory pin `build_prefill_body` in isolation. This
one runs the real `build_app` against stub backends over real HTTP, so it also
covers the seams those unit tests cannot see: SSE framing, incremental
detokenisation, and the NDJSON->SSE conversion from the decode node back to the
client.

The fake vLLM reproduces `ChatCompletionRequest.validate_stream_options`
verbatim, so deleting the drop-list makes this test fail the same way a real
deployment does -- with a 4xx and zero chunks, not a subtle assertion.

No GPU, no tilert, no vllm.

Run:
  CUDA_VISIBLE_DEVICES= python -m pytest \
      tests/pd_vllm/test_stream_e2e.py -v
"""

from __future__ import annotations

import json
import threading
import time

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from tilert.pd_vllm.decode_pool import DecodeNode, Pool
from tilert.pd_vllm.pd_router import RouterCtx, build_app, build_prefill_body, should_include_usage

# What `vllm bench serve --backend openai-chat` puts on the wire
# (vllm/benchmarks/lib/endpoint_request_func.py): streaming, usage accounting,
# and the output length under the *new* OpenAI field name.
BENCH_BODY = {
    "model": "stub-model",
    "messages": [{"role": "user", "content": "hi"}],
    "temperature": 0.0,
    "max_completion_tokens": 3000,
    "stream": True,
    "stream_options": {"include_usage": True},
}

# BENCH_BODY carries temperature 0.0, which is what the benchmark sends and
# what the non-logprobs cases must keep. Logprobs cannot be served there: the
# engine's export is only valid down to MIN_LOGPROBS_TEMPERATURE, and at
# temperature ~= 0 it takes a greedy graph that never runs the top-p sampler at
# all. Requests asking for logprobs therefore override it.
LOGPROBS_TEMPERATURE = 0.6

FIRST_TOKEN_ID = 100  # sampled by prefill, handed to decode
DECODE_TOKEN_IDS = [101, 102, 103]
EXPECTED_TEXT = "ello world"


def STUB_LOGPROB(token_id: int) -> float:
    return -0.5 - 0.1 * (token_id % 5)  # what StubTokenizer makes of the decode tokens


class Captured:
    """Bodies the stub backends received, for asserting on both hops."""

    def __init__(self):
        self.prefill: dict = {}
        self.decode: dict = {}


class StubTokenizer:
    _VOCAB = {101: "ello", 102: " wor", 103: "ld"}

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self._VOCAB.get(i, "") for i in ids)


def _make_vllm(cap: Captured) -> FastAPI:
    """Prefill instance: validates like vLLM, replies like vLLM."""
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        cap.prefill = await request.json()

        # vllm/entrypoints/openai/protocol.py, ChatCompletionRequest:
        # a mode="before" model_validator, so this fires during body parsing --
        # before the model is looked at. That is why the bug is model-agnostic.
        if cap.prefill.get("stream_options") and not cap.prefill.get("stream"):
            return JSONResponse(
                {
                    "error": {
                        "message": "Stream options can only be defined " "when `stream=True`.",
                        "param": "stream_options",
                        "code": 400,
                    }
                },
                status_code=400,
            )

        # vLLM runs with --return-tokens-as-token-ids, so the router reads the
        # first token id out of the logprobs as "token_id:N".
        return JSONResponse(
            {
                "id": "cmpl-stub",
                "model": "stub-model",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "length",
                        "logprobs": {"content": [{"token": f"token_id:{FIRST_TOKEN_ID}"}]},
                        "message": {"role": "assistant", "content": "H"},
                    }
                ],
                "usage": {"prompt_tokens": 7, "completion_tokens": 1},
            }
        )

    return app


def _make_decode(cap: Captured) -> FastAPI:
    """Decode node: NDJSON token batches then a done line, like /pd/decode."""
    app = FastAPI()

    @app.get("/capabilities")
    def capabilities():
        """A real decode node declares what it can execute, and the router
        refuses the optional fields no node claims. A stub without this endpoint
        would make every such request 501 here for a reason that has nothing to
        do with what these tests are about, so it declares full support -- the
        gate itself is tested in test_request_capabilities.py.
        """
        return {
            "profile": "stub",
            "engine": "StubEngine",
            "capabilities": {"penalties": True, "ignore_eos": True, "logprobs": True},
        }

    @app.post("/pd/decode")
    async def decode(request: Request):
        cap.decode = await request.json()

        # The real decode server has two branches; mirror both so the
        # non-streaming handler can be exercised too.
        if not cap.decode.get("stream"):
            out = {
                "rid": cap.decode["rid"],
                "token_ids": DECODE_TOKEN_IDS,
                "seq_len": 8,
                "timing_ms": {"finish_reason": "stop"},
            }
            n = cap.decode.get("top_logprobs")
            if n is not None:
                out["logprobs"] = {
                    "lp": [STUB_LOGPROB(t) for t in DECODE_TOKEN_IDS],
                    "tp": [
                        [[t + k, STUB_LOGPROB(t) - k] for k in range(n)] for t in DECODE_TOKEN_IDS
                    ],
                }
            return out

        n = cap.decode.get("top_logprobs")

        def gen():
            for tid in DECODE_TOKEN_IDS:
                line = {"t": [tid]}
                if n is not None:
                    line["lp"] = [STUB_LOGPROB(tid)]
                    line["tp"] = [[[tid + k, STUB_LOGPROB(tid) - k] for k in range(n)]]
                yield json.dumps(line) + "\n"
            yield json.dumps({"done": True, "finish_reason": "stop"}) + "\n"

        return StreamingResponse(gen(), media_type="application/x-ndjson")

    @app.post("/pd/cancel")
    async def cancel():
        # The router fires this on any incomplete stream; accept and ignore.
        return {"ok": True}

    return app


def _serve(app) -> tuple[uvicorn.Server, int]:
    """Run `app` on an ephemeral port; return the server and the port."""
    cfg = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error")
    server = uvicorn.Server(cfg)
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(200):
        if server.started:
            return server, server.servers[0].sockets[0].getsockname()[1]
        time.sleep(0.05)
    raise RuntimeError("stub server did not start")


@pytest.fixture(scope="module")
def _no_proxy_for_loopback():
    """Keep a local proxy from hijacking the stub traffic.

    The router reaches both stubs with `requests` / `httpx`, which honour
    `http_proxy` from the environment. On a box that has one set (common — this
    is also why a real router deployment must set `no_proxy` for its internal
    addresses), the outbound call is proxied, never arrives, and the test hangs
    to its read timeout instead of failing usefully. Pin it here so the test
    does not depend on how the caller's shell is configured.
    """
    mp = pytest.MonkeyPatch()
    for var in ("no_proxy", "NO_PROXY"):
        mp.setenv(var, "127.0.0.1,localhost")
    for var in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        mp.delenv(var, raising=False)
    yield
    mp.undo()


@pytest.fixture(scope="module")
def stack(_no_proxy_for_loopback):
    """vLLM stub + decode stub + the real router, wired together."""
    cap = Captured()
    _, vllm_port = _serve(_make_vllm(cap))
    _, decode_port = _serve(_make_decode(cap))

    node = DecodeNode("127.0.0.1", 5556, decode_port)
    ctx = RouterCtx(f"http://127.0.0.1:{vllm_port}", Pool([node]), StubTokenizer(), "none")
    _, router_port = _serve(build_app(ctx))

    yield f"http://127.0.0.1:{router_port}", cap


@pytest.fixture(scope="module")
def streamed(stack):
    """Drive one streaming chat request end to end; return chunks + capture.

    `trust_env=False` keeps a proxy in the environment from hijacking the
    loopback call -- the router's own outbound `requests` calls need `no_proxy`
    for the same reason in a real deployment.
    """
    url, cap = stack
    chunks: list[str] = []
    with httpx.Client(timeout=30, trust_env=False) as c:
        with c.stream("POST", f"{url}/v1/chat/completions", json=BENCH_BODY) as r:
            status, ctype = r.status_code, r.headers.get("content-type", "")
            for line in r.iter_lines():
                if line.startswith("data: "):
                    chunks.append(line[6:])
    return status, ctype, chunks, cap


# --------------------------------------------------------------------------- #
# client <-> router -- the client asked for SSE and gets SSE
# --------------------------------------------------------------------------- #


def test_streaming_request_is_not_rejected(streamed) -> None:
    """The regression itself: this was 400 (or 502 before #17), with 0 chunks."""
    status, _, _, _ = streamed
    assert status == 200


def test_response_is_server_sent_events(streamed) -> None:
    _, ctype, _, _ = streamed
    assert ctype.startswith("text/event-stream")


def test_tokens_arrive_as_separate_chunks(streamed) -> None:
    """More chunks than tokens would be wrong; one chunk would mean buffering."""
    _, _, chunks, _ = streamed
    assert len(chunks) > len(DECODE_TOKEN_IDS)


def test_stream_terminates_with_done_sentinel(streamed) -> None:
    _, _, chunks, _ = streamed
    assert chunks[-1] == "[DONE]"


def test_deltas_reassemble_into_the_decoded_text(streamed) -> None:
    """Incremental detokenisation must concatenate back to the whole string."""
    _, _, chunks, _ = streamed
    # The trailing usage chunk carries `choices: []`, so index defensively.
    text = "".join(
        ch["delta"].get("content", "")
        for c in chunks[:-1]
        for ch in json.loads(c).get("choices", [])
    )
    assert text == EXPECTED_TEXT


def test_usage_is_reported_despite_dropping_stream_options(streamed) -> None:
    """The client asked for include_usage; the router answers it itself."""
    _, _, chunks, _ = streamed
    usage = next((u for c in chunks[:-1] if (u := json.loads(c).get("usage"))), None)
    assert usage == {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}


def test_usage_rides_a_chunk_of_its_own(streamed) -> None:
    """Usage sits after the finish_reason chunk, on `choices: []`.

    A client that stops reading at the first chunk bearing a finish_reason --
    the shape vLLM and the OpenAI API emit, and what `vllm bench serve` does --
    would never see usage carried on that same chunk.
    """
    _, _, chunks, _ = streamed
    payloads = [json.loads(c) for c in chunks[:-1]]
    usage_idx = next(i for i, p in enumerate(payloads) if "usage" in p)
    finish_idx = next(
        i for i, p in enumerate(payloads) if any(ch.get("finish_reason") for ch in p["choices"])
    )
    assert finish_idx < usage_idx
    assert payloads[usage_idx]["choices"] == []
    assert "usage" not in payloads[finish_idx]


@pytest.fixture(scope="module")
def streamed_without_usage_opt_in(stack):
    """One streaming request that does NOT ask for usage accounting."""
    url, _ = stack
    body = {k: v for k, v in BENCH_BODY.items() if k != "stream_options"}
    chunks = []
    with httpx.Client(timeout=30, trust_env=False) as c:
        with c.stream("POST", f"{url}/v1/chat/completions", json=body) as r:
            assert r.status_code == 200
            for line in r.iter_lines():
                if line.startswith("data: ") and line[6:] != "[DONE]":
                    chunks.append(json.loads(line[6:]))
    return chunks


def test_no_empty_choices_chunk_without_the_opt_in(streamed_without_usage_opt_in) -> None:
    """`choices: []` must stay opt-in.

    A client that never sent `stream_options` has not declared it can handle
    that shape, and the canonical streaming loop -- `chunk.choices[0].delta`,
    straight out of the OpenAI docs -- raises IndexError on it. OpenAI gates
    the usage chunk behind include_usage for exactly this reason.
    """
    assert all(p["choices"] for p in streamed_without_usage_opt_in)


def test_the_prefill_drop_does_not_consume_the_clients_opt_in() -> None:
    """Dropping stream_options for vLLM must not disarm the usage gate.

    Two features read the same field with opposite intent: vLLM must NOT see it
    (it 400s the pair against the forced stream=False, #14) while the gate MUST,
    and the gate runs after the prefill call. That holds only because
    ``build_prefill_body`` drops from a copy -- stripping in place would switch
    every streaming client to "no usage" with nothing failing.
    """
    body = {"model": "m", "messages": [], "stream": True, "stream_options": {"include_usage": True}}
    prefill = build_prefill_body("/v1/chat/completions", body, DecodeNode("h", 5556, 8000))
    assert "stream_options" not in prefill  # vLLM must not see it
    assert should_include_usage(body) is True  # the gate still must


def test_no_usage_at_all_without_the_opt_in(streamed_without_usage_opt_in) -> None:
    """Absent include_usage means no usage anywhere -- not usage relocated.

    The router used to answer usage unconditionally, riding the chunk that
    closes the choice. That is worse than non-conformant: a client written as
    ``if chunk.choices: ... elif chunk.usage:`` takes the choices branch and
    never reads it, which is how the InferenceX bench reported `Total
    generated tokens: 0`. vLLM and SGLang both gate outright; so do we.
    """
    assert not any("usage" in p for p in streamed_without_usage_opt_in)


@pytest.mark.parametrize(
    "body,wanted",
    [
        ({"stream_options": {"include_usage": True}}, True),
        ({"stream_options": {"include_usage": False}}, False),
        ({"stream_options": {}}, False),
        ({}, False),
        # A client serialising an unset option as null, and one sending the wrong
        # kind: neither may crash the stream on an attribute the router assumed.
        ({"stream_options": None}, False),
        ({"stream_options": "true"}, False),
    ],
)
def test_usage_opt_in_predicate(body, wanted) -> None:
    assert should_include_usage(body) is wanted


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"stream_options": None},
        {"stream_options": {"include_usage": False}},
    ],
)
def test_force_flag_overrides_an_absent_or_false_opt_in(body) -> None:
    """The deployment-level escape hatch, mirroring vLLM's
    enable_force_include_usage and SGLang's
    stream_response_default_include_usage.
    """
    assert should_include_usage(body, force=True) is True


def test_force_flag_defaults_off() -> None:
    """Off unless an operator says otherwise -- the whole point of gating."""
    ctx = RouterCtx("http://unused", Pool([]), None, "none")
    assert ctx.force_include_usage is False
    assert should_include_usage({}, ctx.force_include_usage) is False


# --------------------------------------------------------------------------- #
# router <-> vLLM prefill -- the one hop that must not stream
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("field", ["stream_options", "max_completion_tokens"])
def test_contradicting_field_never_reaches_vllm(streamed, field) -> None:
    _, _, _, cap = streamed
    assert field not in cap.prefill


def test_prefill_leg_is_non_streaming_and_one_token(streamed) -> None:
    _, _, _, cap = streamed
    assert cap.prefill["stream"] is False
    assert cap.prefill["max_tokens"] == 1


# --------------------------------------------------------------------------- #
# router <-> decode node -- where the client's intent actually lands
# --------------------------------------------------------------------------- #


def test_decode_leg_streams(streamed) -> None:
    _, _, _, cap = streamed
    assert cap.decode["stream"] is True


def test_client_output_length_reaches_the_decode_leg(streamed) -> None:
    """Dropped from prefill, honoured here -- the split is still a split."""
    _, _, _, cap = streamed
    assert cap.decode["max_tokens"] == BENCH_BODY["max_completion_tokens"]


def test_prefill_token_is_handed_to_decode(streamed) -> None:
    """Token 1 comes from prefill's logprobs; decode continues from it."""
    _, _, _, cap = streamed
    assert cap.decode["first_token_id"] == FIRST_TOKEN_ID


def test_both_length_fields_resolve_the_way_vllm_resolves_them(stack) -> None:
    """A client sending both must get vLLM's answer, not ours.

    ``resolve_max_tokens`` is unit-tested on its own; this pins the wiring --
    that the router reads the client body through it on the way to decode.
    """
    url, cap = stack
    body = dict(BENCH_BODY, max_tokens=128, max_completion_tokens=3000)
    with httpx.Client(timeout=30, trust_env=False) as c:
        with c.stream("POST", f"{url}/v1/chat/completions", json=body) as r:
            for _ in r.iter_lines():
                pass
    assert cap.decode["max_tokens"] == 3000


def test_ignore_eos_reaches_the_decode_node(stack) -> None:
    """Only the decode node can act on the flag.

    The prefill request is pinned to max_tokens=1 and never reaches a stop
    token, so the whole effect lives on the engine the decode node drives.
    Dropping it is silent: the request succeeds and just stops at the first
    EOS, which is what a fixed-length benchmark asked it not to do.
    """
    url, cap = stack
    body = dict(BENCH_BODY, ignore_eos=True)
    with httpx.Client(timeout=30, trust_env=False) as c:
        with c.stream("POST", f"{url}/v1/chat/completions", json=body) as r:
            for _ in r.iter_lines():
                pass
    assert cap.decode["sampling"]["ignore_eos"] is True


# --------------------------------------------------------------------------- #
# logprobs validation happens before either backend is contacted
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "body,frag",
    [
        ({"top_logprobs": 3}, "logprobs must be set to true"),
        ({"logprobs": True, "top_logprobs": 6}, "[0, 5]"),
        ({"logprobs": True, "top_logprobs": -1}, "[0, 5]"),
    ],
)
def test_bad_logprobs_request_is_rejected_without_touching_a_backend(stack, body, frag) -> None:
    """A 400 must come from the router itself: no prefill, no decode, and the
    stub captures must stay empty for this request.
    """
    url, cap = stack
    cap.prefill.clear()
    cap.decode.clear()
    with httpx.Client(timeout=30, trust_env=False) as c:
        r = c.post(f"{url}/v1/chat/completions", json=dict(BENCH_BODY, stream=False, **body))
    assert r.status_code == 400
    payload = r.json()
    assert payload["error_type"] == "invalid_logprobs"
    assert frag in payload["error"]
    assert cap.prefill == {} and cap.decode == {}


def test_logprobs_on_completions_is_rejected(stack) -> None:
    """The count-typed `logprobs` of /v1/completions is not served here."""
    url, _ = stack
    with httpx.Client(timeout=30, trust_env=False) as c:
        r = c.post(
            f"{url}/v1/completions", json={"model": "stub-model", "prompt": "hi", "logprobs": 2}
        )
    assert r.status_code == 400
    assert "chat/completions" in r.json()["error"]


@pytest.fixture(scope="module")
def logprobs_reply(stack):
    """One non-streaming chat request asking for logprobs, end to end."""
    url, cap = stack
    with httpx.Client(timeout=30, trust_env=False) as c:
        r = c.post(
            f"{url}/v1/chat/completions",
            json=dict(
                BENCH_BODY,
                stream=False,
                temperature=LOGPROBS_TEMPERATURE,
                logprobs=True,
                top_logprobs=3,
            ),
        )
    return r, cap


def test_logprobs_request_is_forwarded_to_the_decode_node(logprobs_reply):
    _, cap = logprobs_reply
    assert cap.decode["top_logprobs"] == 3


def test_logprobs_reach_the_client_in_openai_shape(logprobs_reply):
    r, _ = logprobs_reply
    assert r.status_code == 200
    lp = r.json()["choices"][0]["logprobs"]
    assert set(lp) == {"content", "refusal"}
    assert lp["refusal"] is None
    assert len(lp["content"]) == len(DECODE_TOKEN_IDS)


def test_each_entry_describes_its_token(logprobs_reply):
    r, _ = logprobs_reply
    content = r.json()["choices"][0]["logprobs"]["content"]
    for tid, item in zip(DECODE_TOKEN_IDS, content):
        assert item["logprob"] == STUB_LOGPROB(tid)
        assert item["bytes"] == list(item["token"].encode("utf-8"))
        assert len(item["top_logprobs"]) == 3


def test_reassembled_tokens_match_the_message_content(logprobs_reply):
    """The array must line up with what the client actually received."""
    r, _ = logprobs_reply
    body = r.json()
    content = body["choices"][0]["logprobs"]["content"]
    assert "".join(c["token"] for c in content) == body["choices"][0]["message"]["content"]


def test_logprobs_is_null_not_absent_when_not_requested(stack):
    """The field is declared, its value is null -- which is what vLLM sends.

    Non-streaming responses go out as `JSONResponse(content=result.model_dump())`
    there, with no `exclude_none`, so `ChatCompletionResponseChoice.logprobs`
    appears as null. (Streaming is the other way round: chunks use
    `exclude_unset=True`, so an unset logprobs is omitted -- which is what the
    streaming assertions below check.)

    The intent this replaces is unchanged: nothing is fabricated when logprobs
    were not asked for. Only the spelling of "nothing" moved, from an absent key
    to an explicit null a client can read unconditionally.
    """
    url, _ = stack
    with httpx.Client(timeout=30, trust_env=False) as c:
        r = c.post(f"{url}/v1/chat/completions", json=dict(BENCH_BODY, stream=False))
    choice = r.json()["choices"][0]
    assert "logprobs" in choice
    assert choice["logprobs"] is None


# --------------------------------------------------------------------------- #
# streaming logprobs: each chunk owns the tokens its text came from
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def streamed_logprobs(stack):
    url, cap = stack
    chunks = []
    with httpx.Client(timeout=30, trust_env=False) as c:
        with c.stream(
            "POST",
            f"{url}/v1/chat/completions",
            json=dict(BENCH_BODY, temperature=LOGPROBS_TEMPERATURE, logprobs=True, top_logprobs=2),
        ) as r:
            assert r.status_code == 200
            for line in r.iter_lines():
                if line.startswith("data: ") and line[6:] != "[DONE]":
                    chunks.append(json.loads(line[6:]))
    return chunks


def _choices(chunks):
    """Every choice across `chunks`.

    Flattened rather than indexed at [0] because the trailing usage chunk
    carries `choices: []`.
    """
    return [ch for c in chunks for ch in c["choices"]]


def test_content_chunks_carry_logprobs(streamed_logprobs) -> None:
    lp_choices = [ch for ch in _choices(streamed_logprobs) if ch.get("logprobs")]
    assert lp_choices, "no chunk carried logprobs"
    for ch in lp_choices:
        lp = ch["logprobs"]
        assert set(lp) == {"content", "refusal"}
        assert lp["content"]


def test_streamed_entries_cover_every_decode_token_once(streamed_logprobs) -> None:
    """No token may be dropped or double-counted by the pending buffer."""
    got = [
        e["logprob"]
        for ch in _choices(streamed_logprobs)
        if ch.get("logprobs")
        for e in ch["logprobs"]["content"]
    ]
    assert got == [STUB_LOGPROB(t) for t in DECODE_TOKEN_IDS]


def test_streamed_logprob_text_matches_the_delta(streamed_logprobs) -> None:
    """A chunk's entries must reassemble that chunk's own content delta."""
    for ch in _choices(streamed_logprobs):
        if not ch.get("logprobs"):
            continue
        joined = "".join(e["token"] for e in ch["logprobs"]["content"])
        assert joined == ch["delta"].get("content", "")


def test_streaming_without_logprobs_has_no_logprobs_key(stack) -> None:
    url, _ = stack
    chunks = []
    with httpx.Client(timeout=30, trust_env=False) as c:
        with c.stream("POST", f"{url}/v1/chat/completions", json=BENCH_BODY) as r:
            for line in r.iter_lines():
                if line.startswith("data: ") and line[6:] != "[DONE]":
                    chunks.append(json.loads(line[6:]))
    assert not any("logprobs" in ch for ch in _choices(chunks))


# --------------------------------------------------------------------------- #
# envelope: one response, one timestamp
# --------------------------------------------------------------------------- #
def test_every_chunk_of_one_response_shares_a_created(streamed):
    """vLLM threads a single ``created_time`` through every chunk it builds.

    Reading the clock per chunk gave one response several timestamps, which
    breaks a client that groups or de-duplicates by ``(id, created)``. Asserted
    behaviourally here — the source-level guard in test_openai_envelope.py
    catches a reintroduction that this stream is too short to expose.
    """
    _, _, chunks, _ = streamed
    stamps = {json.loads(c)["created"] for c in chunks if c != "[DONE]"}
    assert len(stamps) == 1, f"one response carried {len(stamps)} timestamps"


def test_the_trailing_usage_chunk_shares_it_too(streamed):
    """The usage chunk is built by a different function; it must agree."""
    _, _, chunks, _ = streamed
    parsed = [json.loads(c) for c in chunks if c != "[DONE]"]
    usage_chunks = [p for p in parsed if p.get("usage") is not None]
    assert usage_chunks, "no usage chunk to compare"
    assert {p["created"] for p in parsed} == {usage_chunks[0]["created"]}


def test_streamed_usage_carries_a_consistent_total(streamed):
    _, _, chunks, _ = streamed
    for c in chunks:
        if c == "[DONE]":
            continue
        u = json.loads(c).get("usage")
        if u is None:
            continue
        assert u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"]
        for k, v in u.items():
            assert isinstance(v, int), f"{k} is {type(v).__name__}"
