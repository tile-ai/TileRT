"""``stop`` served on the router's side of the detokeniser.

`stop` is a property of the decoded TEXT, so it cannot be served by handing
extra ids to the decode loop: a stop string routinely spans two tokens, and
byte-level BPE can split one character across tokens. The decode node emits ids;
the router is where text exists.

Three layers, tested at the level each one actually decides something:

* :func:`check_stop_strings` / :class:`StopWindow` -- the matcher, ported from
  vLLM, and the hold-back that keeps text off the wire while it could still turn
  out to be the start of a stop.
* :func:`resolve_stop_request` -- what the router accepts.
* the router -- that both response paths answer the same request the same way.

Per-token attribution (logprob entries, `completion_tokens`) is a property of
`ReplyStream` and is pinned in test_reply.py, which needs no
HTTP. This file does not repeat it.

No GPU, no tilert, no vllm.

Run:
  CUDA_VISIBLE_DEVICES= python -m pytest tests/pd_vllm/test_stop_strings.py -v
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

from tilert.pd_vllm import pd_router
from tilert.pd_vllm.capabilities import (
    CapabilityUnavailable,
    InvalidParameter,
)
from tilert.pd_vllm.decode_pool import DecodeNode, Pool
from tilert.pd_vllm.pd_router import (
    RouterCtx,
    build_app,
    build_prefill_body,
)
from tilert.pd_vllm.request_gate import resolve_stop_request
from tilert.pd_vllm.stop_strings import (
    StopWindow,
    check_stop_strings,
    resolve_stop,
)


def drive(w: StopWindow, delta: str) -> str:
    """Absorb a delta and take whatever the client may now see.

    `push` and `take` are separate on purpose -- absorbing text and deciding how
    much of it is safe to send are different questions, and that split is why the
    window needs no released/unreleased queue. Tests that only care about the
    combination go through here.
    """
    w.push(delta)
    return w.take()


# --------------------------------------------------------------------------- #
# The matcher: same request, same character, as vLLM
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text,new,stop,want",
    [
        ("hello STOP", 5, ["STOP"], ("STOP", 6)),
        ("hello STOP", 5, ["ZZ"], None),
        ("hello", 5, [], None),
        ("hello STOP", 0, ["STOP"], None),
    ],
)
def test_the_matcher_reports_the_cut(text, new, stop, want):
    assert check_stop_strings(text, new, stop, False) == want


def test_a_stop_straddling_the_delta_boundary_is_found():
    """The search starts before the new text, not at it.

    "STO" was already emitted and "P" just arrived, so a matcher looking only at
    the new characters would miss the stop entirely -- and the reply would run
    past it. This is the rule the vLLM port exists for.
    """
    assert check_stop_strings("aSTOP", 1, ["STOP"], False) == ("STOP", 1)


def test_the_stop_that_completes_earliest_wins():
    """MTP appends a batch of tokens at once, so several stops can match in one step.

    The earliest-completing one is what a one-token-at-a-time stream would have hit, so the
    reply does not depend on the batch size.
    """
    got = check_stop_strings("aXbYc", 5, ["Y", "X"], False)
    assert got == ("X", 1)


def test_ties_go_to_stop_list_order():
    assert check_stop_strings("abAB", 4, ["AB", "B"], False) == ("AB", 2)


def test_include_in_output_cuts_after_the_stop_instead_of_before():
    assert check_stop_strings("hi STOP!", 8, ["STOP"], True) == ("STOP", 7)


def test_include_in_output_needs_no_cut_when_the_stop_ends_the_text():
    """-1 means "the text is already right", which is the common case: the stop
    completes on the token that just arrived.
    """
    assert check_stop_strings("hi STOP", 7, ["STOP"], True) == ("STOP", -1)


# --------------------------------------------------------------------------- #
# The hold-back: text on the wire cannot be recalled
# --------------------------------------------------------------------------- #
def test_the_tail_that_could_start_a_stop_is_held():
    t = StopWindow(["STOP"])
    assert drive(t, "hello STO") == "hello ", "STO could still become STOP"
    assert drive(t, "P") == ""
    assert t.stopped == "STOP"


def test_the_held_tail_is_released_once_it_cannot_be_a_stop():
    """The hold-back is a fixed len(stop)-1 characters, not the matched prefix.

    "STORY" proves "STO" was not a stop, but "RY" could itself begin one, so
    three characters stay held. Releasing eagerly would need the matcher to
    report how much of the tail is still live, which buys nothing: the delay is
    bounded by the longest stop string.
    """
    t = StopWindow(["STOP"])
    assert drive(t, "hello STO") == "hello "
    assert drive(t, "RY") == "ST"
    assert t.take(final=True) == "ORY"


def test_the_held_tail_is_released_at_end_of_stream():
    t = StopWindow(["STOP"])
    assert drive(t, "hi ST") == "hi"
    assert t.take(final=True) == " ST"


def test_nothing_is_held_when_no_stop_was_asked_for():
    """The pass-through matters: holding back would delay every token of every
    request that does not use stop, which is nearly all of them. It also does
    not accumulate the text, which cost a copy of the whole reply per token.
    """
    t = StopWindow([])
    assert drive(t, "hello") == "hello"
    assert t.take(final=True) == ""
    assert len(t._text) == 0, "nothing retained"


def test_pushes_after_a_stop_return_nothing():
    """The decode node cannot see text, so it keeps sending. The reply ended."""
    t = StopWindow(["STOP"])
    drive(t, "aSTOP")
    assert drive(t, " more") == "", "the reply ended; later text is not it"
    assert t.take(final=True) == ""


def test_every_character_comes_out_exactly_once():
    """Held text must be released later, not dropped, and not sent twice."""
    t = StopWindow(["STOP"])
    assert (drive(t, "hello STO") + drive(t, "RY") + t.take(final=True)) == "hello STORY"


@pytest.mark.parametrize(
    "raw,want",
    [
        (None, []),
        ("END", ["END"]),
        (["A", "B"], ["A", "B"]),
        ([], []),
    ],
)
def test_stop_is_normalised_to_a_list(raw, want):
    assert resolve_stop({"stop": raw}) == want


@pytest.mark.parametrize("raw", [[""], ["", "END"], ""])
def test_an_empty_stop_string_is_rejected_not_dropped(raw):
    """It matches at position 0 of everything, so it is not a neutral value the
    way `[]` is -- dropping it serves an unrestricted completion to a client who
    asked for a restricted one.

    vLLM raises `ValueError("stop cannot contain an empty string.")`, verified
    against 0.25.1. The router strips `stop` from the prefill request, so vLLM
    no longer gets the chance to say so and the router has to.
    """
    with pytest.raises(ValueError):
        resolve_stop({"stop": raw})


# --------------------------------------------------------------------------- #
# What the router accepts
# --------------------------------------------------------------------------- #
class _Tok:
    def decode(self, ids, skip_special_tokens=False):
        return "".join(_VOCAB.get(i, "") for i in ids)


def test_stop_needs_no_tokenizer_when_it_is_not_asked_for():
    assert resolve_stop_request({}, None) == ([], False)


def test_stop_strings_without_a_tokenizer_are_refused_not_ignored():
    """`--parser none` with no `--model-path`: there is no text to match
    against, and silently ignoring the field would serve a reply that runs past
    the client's stop.
    """
    with pytest.raises(CapabilityUnavailable):
        resolve_stop_request({"stop": ["END"]}, None)


@pytest.mark.parametrize(
    "body",
    [
        {"stop": 5},
        {"stop": [1, 2]},
        {"stop": [""]},
    ],
)
def test_an_unusable_stop_request_is_a_400(body):
    with pytest.raises(InvalidParameter):
        resolve_stop_request(body, _Tok())


def test_the_flag_without_a_stop_is_a_no_op_not_an_error():
    """vLLM serves it, so refusing it here would turn a request that works
    against a native endpoint into a 400. With no stop strings the flag has
    nothing to include.
    """
    assert resolve_stop_request({"include_stop_str_in_output": True}, _Tok()) == ([], True)


# `include_stop_str_in_output` is coerced by the same validator vLLM's request
# model uses, so the two stacks accept and refuse the same values. Verified
# against `ChatCompletionRequest` on vLLM 0.25.1 / pydantic 2.13.4 -- being
# STRICTER here would turn a request vLLM serves into a 400, which is the
# mistake the capability gate's top_k handling already warns about.
@pytest.mark.parametrize(
    "value,want",
    [
        (True, True),
        (False, False),
        (1, True),
        (0, False),
        ("true", True),
        ("False", False),
    ],
)
def test_the_flag_is_coerced_exactly_as_vllm_coerces_it(value, want):
    assert resolve_stop_request({"stop": ["A"], "include_stop_str_in_output": value}, _Tok()) == (
        ["A"],
        want,
    )


@pytest.mark.parametrize("value", [None, 2, "maybe", [], {}])
def test_a_value_vllm_would_refuse_is_a_400(value):
    """An explicit `null` included: vLLM's field is `bool = False`, not
    `bool | None`, so pydantic answers 422 for it. The router strips the field
    before prefill, so vLLM never gets the chance to say so.
    """
    with pytest.raises(InvalidParameter):
        resolve_stop_request({"stop": ["A"], "include_stop_str_in_output": value}, _Tok())


def test_the_prefill_request_does_not_carry_the_stop_fields():
    """The prefill instance generates one token.

    Matching there could truncate it or report finish_reason="stop" for a prefill that in fact
    succeeded -- and the router is going to match over the whole reply anyway.
    """
    out = build_prefill_body(
        "/v1/chat/completions",
        {"messages": [], "stop": ["END"], "include_stop_str_in_output": True},
        DecodeNode(host="h", ctrl_port=1, http_port=2),
    )
    assert "stop" not in out
    assert "include_stop_str_in_output" not in out


# --------------------------------------------------------------------------- #
# The router: one request, two paths, one answer
# --------------------------------------------------------------------------- #
# "Hello, world. STOP tail" over eight tokens, with the stop split across two of
# them so the hold-back and the cross-token match are both exercised.
_VOCAB = {
    201: "Hello",
    202: ", world",
    203: ". ",
    204: "ST",
    205: "OP",
    206: " tail",
    207: " more",
    208: "!",
}
# The decode node's reply INCLUDES the prefill's token: `PDEngine.decode`
# returns "completion ids (incl. first_token_id)" and fires on_token for it too,
# so both response paths see it. Its logprob is null -- the node echoed the
# token rather than sampling it.
_FIRST = 201
_IDS = [201, 202, 203, 204, 205, 206, 207, 208]
_FULL = "".join(_VOCAB[i] for i in _IDS)


def _serve(app):
    cfg = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error")
    server = uvicorn.Server(cfg)
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(200):
        if server.started:
            return server.servers[0].sockets[0].getsockname()[1]
        time.sleep(0.05)
    raise RuntimeError("stub server did not start")


def _make_vllm():
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        assert "stop" not in body, "the prefill leg must not match stop strings"
        return {
            "id": "cmpl-stop",
            "model": "stub",
            "choices": [
                {
                    "logprobs": {
                        "content": [
                            {
                                "token": f"token_id:{_FIRST}",
                                "logprob": -0.125,
                                "top_logprobs": [
                                    {"token": f"token_id:{_FIRST}", "logprob": -0.125}
                                ],
                            }
                        ]
                    }
                }
            ],
            "usage": {"prompt_tokens": 4},
            "kv_transfer_params": {},
        }

    @app.post("/v1/completions")
    async def completions(request: Request):
        body = await request.json()
        assert "stop" not in body, "the prefill leg must not match stop strings"
        # /v1/completions reports the first token under `tokens`, not `content`.
        return {
            "id": "cmpl-stop",
            "model": "stub",
            "choices": [{"logprobs": {"tokens": [f"token_id:{_FIRST}"]}}],
            "usage": {"prompt_tokens": 4},
            "kv_transfer_params": {},
        }

    return app


# What the decode node saw, so a test can tell early cancellation from mere
# truncation of a reply that was generated in full.
_SEEN: dict = {"emitted": 0, "cancelled": [], "stream": None, "decode_body": {}}


def _make_decode(
    filler: int = 0,
    batched: bool = False,
    error_after: int | None = None,
    error_type: str | None = None,
    truncate_after: int | None = None,
    drop_tp: bool = False,
    garbage_after: int | None = None,
    lose_lines_after: int | None = None,
    null_lp: bool = False,
    status: int | None = None,
):
    app = FastAPI()

    @app.get("/capabilities")
    def capabilities():
        return {
            "profile": "stub",
            "engine": "StubEngine",
            "capabilities": {"penalties": True, "ignore_eos": True, "logprobs": True},
        }

    def _lp(n, ids):
        out = {"lp": [None] * len(ids) if null_lp else [-0.5] * len(ids)}
        if not drop_tp:
            out["tp"] = [[[t, -0.5 - k] for k in range(n)] for t in ids]
        return out

    @app.post("/pd/decode")
    async def decode(request: Request):
        body = await request.json()
        if status is not None:
            # A node answering with a status rather than a stream: the shape
            # both handlers have to classify before any body exists.
            return JSONResponse({"error": "engine down"}, status_code=status)
        n = body.get("top_logprobs")
        _SEEN["stream"] = bool(body.get("stream"))
        _SEEN["emitted"] = 0
        _SEEN["decode_body"] = body
        if not body.get("stream"):
            out = {
                "rid": body["rid"],
                "token_ids": _IDS,
                "seq_len": 8,
                "timing_ms": {"finish_reason": "length"},
            }
            if n is not None:
                out["logprobs"] = _lp(n, _IDS)
                out["logprobs"]["lp"][0] = None  # echoed, not sampled
            return out

        if batched:
            # What a real node does: it drains its queue with `get_nowait()`
            # before writing, so one line carries everything the engine has
            # produced -- measured at 30 tokens on a live pair. A stop then
            # lands part-way into a batch.
            def gen_batched():
                ids = _IDS + [207] * filler
                line = {"t": ids}
                if n is not None:
                    line["lp"] = [None] * len(ids) if null_lp else [None] + [-0.5] * (len(ids) - 1)
                    line["tp"] = [[[t, -0.5]] for t in ids]
                yield json.dumps(line) + "\n"
                yield json.dumps({"done": True, "finish_reason": "length"}) + "\n"

            return StreamingResponse(gen_batched(), media_type="application/x-ndjson")

        def gen():
            # `filler` stands in for a client's large max_tokens: the node keeps
            # going because it cannot see text.
            for k, tid in enumerate(_IDS + [207] * filler):
                if lose_lines_after is not None and k >= lose_lines_after:
                    # Silently drop the rest, then still send a normal `done`.
                    continue
                if garbage_after is not None and k == garbage_after:
                    # Not JSON: `DecodeReader.feed` raises on it.
                    yield "{this is not json\n"
                    return
                if truncate_after is not None and k == truncate_after:
                    # A clean EOF with no done/error: a proxy cutting the body,
                    # or a node dying mid-reply.
                    return
                if error_after is not None and k == error_after:
                    err = {"error": "engine exploded"}
                    if error_type is not None:
                        err["error_type"] = error_type
                    yield json.dumps(err) + "\n"
                    return
                line = {"t": [tid]}
                if n is not None:
                    line["lp"] = [None if (k == 0 or null_lp) else -0.5]
                    if not drop_tp:
                        line["tp"] = [[[tid, -0.5 - j] for j in range(n)]]
                _SEEN["emitted"] = k + 1
                yield json.dumps(line) + "\n"
            # `n` is already the requested top_logprobs in this closure, so the
            # token count needs its own name.
            declared = _SEEN["emitted"]
            if lose_lines_after is not None:
                # The node's own count, as a real one reports it -- unchanged by
                # the lines that went missing.
                declared = len(_IDS) + filler
            yield json.dumps({"done": True, "n": declared, "finish_reason": "length"}) + "\n"

        return StreamingResponse(gen(), media_type="application/x-ndjson")

    @app.post("/pd/cancel")
    async def cancel(request: Request):
        _SEEN["cancelled"].append((await request.json()).get("rid"))
        return {"ok": True}

    return app


@pytest.fixture(scope="module")
def _proxy_off():
    """Keep a local proxy from hijacking the stub traffic.

    The router reaches both stubs with `requests`, which honours `http_proxy`
    from the environment. On a box that has one set -- which is also why a real
    deployment must set `no_proxy` for its internal addresses -- the call is
    proxied, never arrives, and the test hangs to its read timeout.
    """
    mp = pytest.MonkeyPatch()
    for var in ("no_proxy", "NO_PROXY"):
        mp.setenv(var, "127.0.0.1,localhost")
    for var in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        mp.delenv(var, raising=False)
    yield
    mp.undo()


@pytest.fixture(scope="module")
def router(_proxy_off):
    vllm_port = _serve(_make_vllm())
    decode_port = _serve(_make_decode())
    node = DecodeNode("127.0.0.1", 5556, decode_port)
    ctx = RouterCtx(f"http://127.0.0.1:{vllm_port}", Pool([node]), _Tok(), "none")
    return f"http://127.0.0.1:{_serve(build_app(ctx))}"


def _body(**over):
    b = {
        "model": "stub",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 64,
        "temperature": 0.0,
    }
    b.update(over)
    return b


def _post(router, **over):
    with httpx.Client(timeout=30, trust_env=False) as c:
        r = c.post(f"{router}/v1/chat/completions", json=_body(**over))
    return r


def _stream(router, **over):
    """Drive the streaming path; return (choice-shaped dict, usage)."""
    text, finish, stop_reason, usage, entries = "", None, "sentinel", None, []
    with httpx.Client(timeout=30, trust_env=False) as c:
        with c.stream(
            "POST", f"{router}/v1/chat/completions", json=_body(stream=True, **over)
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line.startswith("data: ") or line[6:] == "[DONE]":
                    continue
                ch = json.loads(line[6:])
                if ch.get("usage") is not None:
                    usage = ch["usage"]
                for choice in ch["choices"]:
                    text += choice["delta"].get("content", "") or ""
                    entries += (choice.get("logprobs") or {}).get("content") or []
                    if choice.get("finish_reason"):
                        finish = choice["finish_reason"]
                        stop_reason = choice.get("stop_reason", "absent")
    return {
        "text": text,
        "finish_reason": finish,
        "stop_reason": stop_reason,
        "entries": entries,
    }, usage


def test_a_stop_string_truncates_the_reply(router):
    got = _post(router, stop=["STOP"]).json()
    assert got["choices"][0]["message"]["content"] == "Hello, world. "


def test_the_stop_spans_two_tokens(router):
    """ "ST" and "OP" arrive as separate tokens, so a matcher working per token
    would never see the string.
    """
    assert "STOP" in _FULL
    assert _VOCAB[204] + _VOCAB[205] == "STOP"
    assert (
        _post(router, stop=["STOP"]).json()["choices"][0]["message"]["content"] == "Hello, world. "
    )


def test_include_stop_str_in_output_keeps_it(router):
    got = _post(router, stop=["STOP"], include_stop_str_in_output=True).json()
    assert got["choices"][0]["message"]["content"] == "Hello, world. STOP"


def test_stop_reason_names_the_string_that_ended_the_reply(router):
    """A client cannot recover it from the text: the string was cut out."""
    choice = _post(router, stop=["STOP"]).json()["choices"][0]
    assert choice["stop_reason"] == "STOP"
    assert choice["finish_reason"] == "stop"


def test_stop_reason_is_null_when_no_stop_string_matched(router):
    choice = _post(router, stop=["ZZZZ"]).json()["choices"][0]
    assert choice["stop_reason"] is None
    assert choice["finish_reason"] == "length", "what the decode node said"
    assert choice["message"]["content"] == _FULL


def test_usage_counts_what_ran_up_to_the_stop(router):
    """The stop truncates the text, not the count.

    204 ("ST") and 205 ("OP") contributed no visible text and are still counted:
    they ran. vLLM does the same -- its `completion_tokens` is `len()` of the
    detokeniser's untruncated id list. What is NOT counted is the tokens after
    the stop, which the node only generated because it cannot see text.
    """
    got = _post(router, stop=["STOP"]).json()
    assert got["usage"]["completion_tokens"] == 5, "201..205"
    assert _post(router).json()["usage"]["completion_tokens"] == 8


def test_completions_reports_every_id_the_node_sent(router):
    """`token_ids` says what ran, `text` says what came back, and a stop makes
    them differ.

    vLLM's `CompletionOutput.token_ids` is the detokeniser's untruncated list
    for the same reason. Slicing it to the visible text would report fewer
    tokens than `usage` bills for, and could not represent a stop that cut
    inside a token anyway.
    """
    with httpx.Client(timeout=30, trust_env=False) as c:
        r = c.post(
            f"{router}/v1/completions",
            json={"model": "stub", "prompt": "hi", "max_tokens": 64, "stop": ["STOP"]},
        )
    assert r.status_code == 200, r.text[:300]
    choice = r.json()["choices"][0]
    assert choice["text"] == "Hello, world. "
    assert choice["token_ids"] == [201, 202, 203, 204, 205]
    assert len(choice["token_ids"]) == r.json()["usage"]["completion_tokens"]


@pytest.mark.parametrize(
    "over",
    [
        {},
        {"stop": ["STOP"]},
        {"stop": ["STOP"], "include_stop_str_in_output": True},
        {"stop": ["ZZZZ"]},
        {"stop": [". ", "STOP"]},
    ],
)
def test_streaming_and_non_streaming_answer_the_same(router, over):
    """The property the assembler exists for, checked over real HTTP.

    Text, finish_reason, stop_reason and completion_tokens all have to match:
    a client switching `stream` must not get a different reply.
    """
    blocking = _post(router, **over).json()
    streamed, usage = _stream(router, **over)
    choice = blocking["choices"][0]
    assert streamed["text"] == choice["message"]["content"]
    assert streamed["finish_reason"] == choice["finish_reason"]
    assert streamed["stop_reason"] == choice["stop_reason"]
    assert usage is None or usage == blocking["usage"]


@pytest.mark.parametrize("over", [{}, {"stop": ["STOP"]}])
def test_the_two_paths_agree_on_logprobs_too(router, over):
    """One entry per token the reply contains, the same entries either way."""
    extra = {"logprobs": True, "top_logprobs": 1, "temperature": 0.6}
    blocking = _post(router, **over, **extra).json()
    streamed, _ = _stream(router, **over, **extra)
    want = blocking["choices"][0]["logprobs"]["content"]
    assert len(want) == blocking["usage"]["completion_tokens"]
    assert [e["token"] for e in streamed["entries"]] == [e["token"] for e in want]
    assert [e["logprob"] for e in streamed["entries"]] == [e["logprob"] for e in want]


def test_the_streamed_usage_opt_in_agrees_with_the_blocking_one(router):
    _, usage = _stream(router, stop=["STOP"], stream_options={"include_usage": True})
    assert usage["completion_tokens"] == 5


def test_a_stop_string_is_no_longer_refused(router):
    """It was a 501 until the router grew a matcher."""
    assert _post(router, stop=["STOP"]).status_code == 200


@pytest.mark.parametrize(
    "over",
    [
        {"stop": 5},
        {"stop": [""]},
        {"stop": ["A"], "include_stop_str_in_output": None},
    ],
)
def test_an_unusable_stop_request_is_rejected_before_any_backend(router, over):
    assert _post(router, **over).status_code == 400


# --------------------------------------------------------------------------- #
# The point of serving stop at all: it has to stop something
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def long_router(_proxy_off):
    """A node that keeps emitting long past the stop, as a real one does."""
    vllm_port = _serve(_make_vllm())
    decode_port = _serve(_make_decode(filler=200))
    node = DecodeNode("127.0.0.1", 5556, decode_port)
    ctx = RouterCtx(f"http://127.0.0.1:{vllm_port}", Pool([node]), _Tok(), "none")
    return f"http://127.0.0.1:{_serve(build_app(ctx))}"


def _wait_for_cancel(timeout=5.0):
    """The cancel goes out on a daemon thread so a dead node cannot delay the
    reply, so it lands after the response the test just read.
    """
    deadline = time.time() + timeout
    while time.time() < deadline and not _SEEN["cancelled"]:
        time.sleep(0.02)
    return list(_SEEN["cancelled"])


def test_a_non_streaming_stop_stops_the_decode_node(long_router):
    """Not just the reply: the node too.

    The node cannot see text, so a router that asks for the whole sequence up
    front can only trim the answer after the fact -- the node still runs to
    max_tokens and holds its slot for all of it, and the client waits for it.
    So a request with a stop reads the node's streaming protocol even though its
    own reply is not streamed, stops reading at the match, and cancels.

    The stub emits 208 tokens; the stop lands on the fifth.
    """
    _SEEN["cancelled"].clear()
    got = _post(long_router, stop=["STOP"]).json()
    assert got["choices"][0]["message"]["content"] == "Hello, world. "
    assert _SEEN["stream"] is True, "asked the node to stream"
    assert _SEEN["emitted"] < 20, f"read {_SEEN['emitted']} of 208 tokens -- did not stop early"
    assert _wait_for_cancel(), "and told the node to stop"


def test_a_request_without_a_stop_does_not_pay_for_the_stream(long_router):
    """Nothing to match, so nothing to react to mid-generation.

    The blocking protocol stays the default, and a reply that ran to completion is not
    cancelled.
    """
    _SEEN["cancelled"].clear()
    assert _post(long_router).status_code == 200
    assert _SEEN["stream"] is False
    time.sleep(0.3)
    assert not _SEEN["cancelled"], "it finished; there is nothing to cancel"


@pytest.fixture(scope="module")
def batched_router(_proxy_off):
    """A node that delivers the whole sequence in one line, as a fast one does."""
    vllm_port = _serve(_make_vllm())
    decode_port = _serve(_make_decode(filler=25, batched=True))
    node = DecodeNode("127.0.0.1", 5556, decode_port)
    ctx = RouterCtx(f"http://127.0.0.1:{vllm_port}", Pool([node]), _Tok(), "none")
    return f"http://127.0.0.1:{_serve(build_app(ctx))}"


def test_a_batch_that_straddles_the_stop_is_not_counted_whole(batched_router):
    """The tokens behind the stop, inside the same line, are not the reply.

    Found on a live pair: the node's first line carried 30 tokens, the stop
    landed on the sixth, and `token_ids` reported all 30 against a
    `completion_tokens` of 6. One source for the id list, the count and the
    entries is what keeps them from disagreeing.
    """
    with httpx.Client(timeout=30, trust_env=False) as c:
        r = c.post(
            f"{batched_router}/v1/completions",
            json={
                "model": "stub",
                "prompt": "hi",
                "max_tokens": 512,
                "stop": ["STOP"],
                "temperature": 0.0,
            },
        )
    assert r.status_code == 200, r.text[:300]
    j = r.json()
    choice = j["choices"][0]
    assert choice["text"] == "Hello, world. "
    assert choice["token_ids"] == [201, 202, 203, 204, 205], "up to the stop"
    assert len(choice["token_ids"]) == j["usage"]["completion_tokens"]


def test_the_entries_of_a_straddled_batch_match_the_count(batched_router):
    """Same for logprobs: one entry per token counted, no more."""
    with httpx.Client(timeout=30, trust_env=False) as c:
        r = c.post(
            f"{batched_router}/v1/chat/completions",
            json={
                "model": "stub",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 512,
                "stop": ["STOP"],
                "temperature": 0.6,
                "logprobs": True,
                "top_logprobs": 1,
            },
        )
    assert r.status_code == 200, r.text[:300]
    j = r.json()
    ents = j["choices"][0]["logprobs"]["content"]
    assert len(ents) == j["usage"]["completion_tokens"] == 5


# --------------------------------------------------------------------------- #
# Ordering and fail-fast
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def erroring_router(_proxy_off):
    """A node that fails part-way through, while a tail is still held back."""
    vllm_port = _serve(_make_vllm())
    decode_port = _serve(_make_decode(error_after=4))
    node = DecodeNode("127.0.0.1", 5556, decode_port)
    ctx = RouterCtx(f"http://127.0.0.1:{vllm_port}", Pool([node]), _Tok(), "none")
    return f"http://127.0.0.1:{_serve(build_app(ctx))}"


def test_a_decode_error_does_not_jump_ahead_of_held_text(erroring_router):
    """The error marker must come after the reply's own text, not through it.

    With a stop that never matches, the matcher is holding up to len(stop)-1
    characters when the node fails. Emitting the marker before releasing them
    gives the client `prefix[decode error]suffix` -- text out of order, which is
    worse than the error itself.
    """
    text = ""
    with httpx.Client(timeout=30, trust_env=False) as c:
        with c.stream(
            "POST", f"{erroring_router}/v1/chat/completions", json=_body(stream=True, stop=["ZZZZ"])
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line.startswith("data: ") or line[6:] == "[DONE]":
                    continue
                for ch in json.loads(line[6:])["choices"]:
                    text += ch["delta"].get("content", "") or ""
    assert "[decode error:" in text, "the failure is reported"
    body, _, tail = text.partition("[decode error:")
    assert "]" in tail
    after = tail.split("]", 1)[1]
    assert after == "", f"reply text arrived after the error marker: {after!r}"
    # 201..204 -> "Hello" ", world" ". " "ST"; the trailing newline is the
    # marker's own prefix. The held "ST" is the point: it is inside `body`.
    assert body == "Hello, world. ST\n", "all of the reply came first"


def test_a_malformed_stop_does_not_wait_on_a_hung_node():
    """`stop: 5` is answerable from the request alone.

    The capability probe talks HTTP to every uncached node and blocks for
    `Pool.CAPS_TIMEOUT_S` (2 s) per node that accepts the connection and then
    says nothing -- which is what a wedged node looks like, and is different
    from a refused connection, which fails instantly. Validating after the
    probe makes a deterministic 400 pay for a backend its answer never
    depended on.
    """
    import socket

    from fastapi.testclient import TestClient

    # Accept the connection, then never answer: `requests` blocks on read until
    # its timeout. A closed port would be refused immediately and prove nothing.
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(8)
    hung_port = listener.getsockname()[1]
    held = []

    def _swallow():
        while True:
            try:
                held.append(listener.accept()[0])
            except OSError:
                return

    threading.Thread(target=_swallow, daemon=True).start()
    try:
        node = DecodeNode("127.0.0.1", 5556, hung_port)
        ctx = RouterCtx(f"http://127.0.0.1:{hung_port}", Pool([node]), _Tok(), "none")
        client = TestClient(build_app(ctx))
        t0 = time.time()
        r = client.post(
            "/v1/chat/completions",
            json={"model": "stub", "stop": 5, "messages": [{"role": "user", "content": "hi"}]},
        )
        dt = time.time() - t0
        assert r.status_code == 400, r.text[:200]
        assert dt < Pool.CAPS_TIMEOUT_S, (
            f"took {dt:.1f}s against a {Pool.CAPS_TIMEOUT_S}s probe timeout "
            f"-- it waited on the capability probe"
        )
    finally:
        listener.close()
        for sock in held:
            sock.close()


def test_logprobs_without_a_tokenizer_are_refused_not_nulled(_proxy_off):
    """A 200 with `logprobs: null` reports success for a field the client asked
    for and did not get -- and both backends compute the values first.

    `--parser none` with no `--model-path` is a supported configuration; it just
    cannot name tokens. Refused at the door, before any backend work.
    """
    from fastapi.testclient import TestClient

    node = DecodeNode("127.0.0.1", 5556, _serve(_make_decode()))
    ctx = RouterCtx(
        f"http://127.0.0.1:{_serve(_make_vllm())}", Pool([node]), None, "none"
    )  # no tokenizer
    client = TestClient(build_app(ctx))
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "stub",
            "logprobs": True,
            "top_logprobs": 1,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert r.status_code == 501, r.text[:200]
    assert r.json()["error_type"] == "capability_unavailable"
    # Without logprobs the same router still serves the request.
    ok = client.post(
        "/v1/chat/completions",
        json={"model": "stub", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert ok.status_code == 200, ok.text[:200]
    assert ok.json()["choices"][0]["logprobs"] is None


@pytest.fixture(scope="module")
def grammar_violation_router(_proxy_off):
    """A node that reports a grammar violation while a tail is still held."""
    vllm_port = _serve(_make_vllm())
    decode_port = _serve(_make_decode(error_after=4, error_type="grammar_violation"))
    node = DecodeNode("127.0.0.1", 5556, decode_port)
    ctx = RouterCtx(f"http://127.0.0.1:{vllm_port}", Pool([node]), _Tok(), "none")
    return f"http://127.0.0.1:{_serve(build_app(ctx))}"


def test_a_grammar_violation_does_not_drop_held_text(grammar_violation_router):
    """The fail-closed branch returns early, so it has to flush too.

    It emits an SSE error event and `[DONE]` and returns without reaching the
    flush after the loop -- so with a non-matching stop the characters the
    matcher was still holding vanish from a reply the client otherwise keeps.
    The generic decode-error branch already flushed; this one did not.
    """
    text, saw_error = "", False
    with httpx.Client(timeout=30, trust_env=False) as c:
        with c.stream(
            "POST",
            f"{grammar_violation_router}/v1/chat/completions",
            json=_body(stream=True, stop=["ZZZZ"]),
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line.startswith("data: ") or line[6:] == "[DONE]":
                    continue
                payload = json.loads(line[6:])
                if payload.get("error"):
                    saw_error = True
                    continue
                for ch in payload.get("choices", []):
                    text += ch["delta"].get("content", "") or ""
    assert saw_error, "the violation is still reported"
    # 201..204 -> "Hello" ", world" ". " "ST"; "ST" is the held tail.
    assert text == "Hello, world. ST", f"held text was dropped: {text!r}"


@pytest.fixture(scope="module")
def truncating_router(_proxy_off):
    """A node whose body ends with no `done` and no `error`."""
    vllm_port = _serve(_make_vllm())
    # Cut AFTER the point where "STOP" completes (201..205), so the two cases
    # below differ by the stop matching rather than by how much was emitted.
    decode_port = _serve(_make_decode(truncate_after=7))
    node = DecodeNode("127.0.0.1", 5556, decode_port)
    ctx = RouterCtx(f"http://127.0.0.1:{vllm_port}", Pool([node]), _Tok(), "none")
    return f"http://127.0.0.1:{_serve(build_app(ctx))}"


def test_a_stream_ending_without_a_terminal_message_is_not_a_success(truncating_router):
    """A partial completion must not be reported as a finished one.

    A stop-bearing non-streaming request reads the node's NDJSON protocol. If
    that body reaches a clean EOF before `done`, `error` or a stop match -- a
    proxy cutting it short, a node dying mid-reply -- assembling what arrived
    would answer 200 with `finish_reason: "stop"` and a null `stop_reason` for a
    reply that was truncated.
    """
    r = _post(truncating_router, stop=["ZZZZ"])
    assert r.status_code == 502, r.text[:200]
    assert r.json()["error_type"] == "decode_truncated"


def test_a_stop_match_is_still_a_legitimate_early_exit(truncating_router):
    """The stop case leaves the loop early on purpose and must stay a 200."""
    r = _post(truncating_router, stop=["STOP"])
    assert r.status_code == 200, r.text[:200]
    assert r.json()["choices"][0]["stop_reason"] == "STOP"


def test_the_role_chunk_still_precedes_every_delta(router):
    """Guards a merge resolution rather than a feature of this PR.

    #34 made the opening `delta.role` chunk lazy -- sent when there is
    something to send, not unconditionally -- and this PR rewrote the loop it
    sits in. Resolving that conflict wrongly would let content deltas reach the
    client before the role, which nothing else here would notice.
    """
    order = []
    with httpx.Client(timeout=30, trust_env=False) as c:
        with c.stream("POST", f"{router}/v1/chat/completions", json=_body(stream=True)) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line.startswith("data: ") or line[6:] == "[DONE]":
                    continue
                for ch in json.loads(line[6:]).get("choices", []):
                    d = ch["delta"]
                    if d.get("role"):
                        order.append("role")
                    elif d.get("content"):
                        order.append("content")
    assert order.count("role") == 1, f"role sent {order.count('role')} times"
    assert order[0] == "role", f"a delta preceded the role: {order[:3]}"


# --------------------------------------------------------------------------- #
# The retained window
# --------------------------------------------------------------------------- #
class _WholeTextTracker:
    """The obvious implementation: keep everything, never trim.

    The reference the windowed tracker has to agree with. Ported from what
    `StopWindow` was before the window, so a divergence shows up as a
    difference in released text rather than as a performance number.
    """

    def __init__(self, stop, include=False):
        self.stop, self.include = list(stop), include
        self._hold = 0 if include else (max((len(x) for x in self.stop), default=1) - 1)
        self.text, self._released, self.stopped = "", 0, None

    def push(self, delta):
        """The old shape on purpose: absorb and release in one call.

        The reference keeps the WHOLE text and never trims, which is what the
        window has to stay equivalent to.
        """
        if self.stopped is not None or not delta:
            return ""
        if not self.stop:
            return delta
        self.text += delta
        hit = check_stop_strings(self.text, len(delta), self.stop, self.include)
        if hit is not None:
            self.stopped, cut = hit
            if cut != -1:
                self.text = self.text[:cut]
            out = self.text[self._released :]
            self._released = len(self.text)
            return out
        end = max(self._released, len(self.text) - self._hold)
        out = self.text[self._released : end]
        self._released = end
        return out

    def finish(self):
        if self.stopped is not None:
            return ""
        out = self.text[self._released :]
        self._released = len(self.text)
        return out


@pytest.mark.parametrize("slack", [0, 4096])
@pytest.mark.parametrize(
    "stop,include",
    [
        (["STOP"], False),
        (["STOP"], True),
        (["Observation:", "\n\n"], False),
        (["NEVERMATCHES"], False),
    ],
)
def test_the_window_releases_exactly_what_keeping_everything_would(stop, include, slack):
    """The window is an allocation change and must not be a behaviour change.

    Driven with deltas that straddle the boundary in every way that matters:
    one character at a time, in chunks, and with the stop split across two.

    `slack=0` makes the trim fire on every push, which is the only way a short
    run reaches it at all -- at the shipped 4096 the window never fills here, so
    the trim would go untested and a look-back cut short by it would not show.
    That case matters most with `include_stop_str_in_output`, where nothing is
    held back and a straddling stop therefore sits in ALREADY RELEASED text.
    """
    import random

    rng = random.Random(20260828)
    alphabet = ["a", "ST", "OP", "\n", "Observation", ":", "STOP", "x", "\n\n"]
    for _ in range(40):
        deltas = [rng.choice(alphabet) for _ in range(60)]
        a, b = StopWindow(stop, include), _WholeTextTracker(stop, include)
        a._slack = slack  # per instance now, not a class attribute
        out_a = "".join(drive(a, d) for d in deltas) + a.take(final=True)
        out_b = "".join(b.push(d) for d in deltas) + b.finish()
        assert out_a == out_b, f"released text differs on {deltas[:6]}"
        assert a.stopped == b.stopped


def test_the_retained_text_does_not_grow_with_the_reply():
    """A stop that never matches must not make the router hold the completion.

    Keeping all of it cost a copy of the response per token -- quadratic in the
    length, 0.75 s and 200 KB over a 200k-token generation -- for text nothing
    reads once it has gone out.
    """
    t = StopWindow(["NEVERMATCHES"])
    for _ in range(200_000):
        drive(t, "x")
    assert len(t._text) < 8_192, f"retained {len(t._text):,} characters"


def test_a_stop_still_matches_after_the_window_has_trimmed():
    """The look-back a straddling match needs survives trimming."""
    t = StopWindow(["STOP"])
    released = "".join(drive(t, "x") for _ in range(20_000))
    released += drive(t, "ST")  # could still become the stop, so held back
    released += drive(t, "OP")  # completes it, long after the first trim
    assert t.stopped == "STOP"
    assert released == "x" * 20_000, "the stop and its prefix are cut"


@pytest.fixture(scope="module")
def failing_status_router(_proxy_off):
    """A node whose /pd/decode answers 500 with an unclassified body."""
    vllm_port = _serve(_make_vllm())
    decode_port = _serve(_make_decode(status=500))
    node = DecodeNode("127.0.0.1", 5556, decode_port)
    ctx = RouterCtx(f"http://127.0.0.1:{vllm_port}", Pool([node]), _Tok(), "none")
    return f"http://127.0.0.1:{_serve(build_app(ctx))}"


def test_both_paths_refuse_an_unclassified_status_the_same_way(failing_status_router):
    """Same node behaviour, same answer, whether or not a stream was asked for.

    They differed: the blocking path let `raise_for_status` reach a generic
    handler and reported its message, the streaming one reported `decode call
    failed` with the status. A client could not write one error handler for a
    node that was down.
    """
    blocking = _post(failing_status_router, stop=["STOP"])
    with httpx.Client(timeout=30, trust_env=False) as c:
        streaming = c.post(
            f"{failing_status_router}/v1/chat/completions", json=_body(stream=True, stop=["STOP"])
        )
    assert blocking.status_code == 502, blocking.text[:200]
    assert streaming.status_code == 502, streaming.text[:200]
    assert blocking.json() == streaming.json(), "one shape for one fault"
    body = blocking.json()
    assert body["error"] == "decode call failed", body
    assert body["status"] == 500, body
    assert body["rid"], body


@pytest.fixture(scope="module")
def null_logprobs_router(_proxy_off):
    """A node whose `lp` rows are correctly sized and entirely null."""
    vllm_port = _serve(_make_vllm())
    decode_port = _serve(_make_decode(null_lp=True))
    node = DecodeNode("127.0.0.1", 5556, decode_port)
    ctx = RouterCtx(f"http://127.0.0.1:{vllm_port}", Pool([node]), _Tok(), "none")
    return f"http://127.0.0.1:{_serve(build_app(ctx))}"


@pytest.mark.parametrize(
    "stop,why",
    [
        (["STOP"], "streaming protocol -- a stop puts a blocking request on it"),
        (None, "blocking protocol -- one object, same validator"),
    ],
)
def test_a_null_logprob_past_the_echoed_token_is_refused(null_logprobs_router, stop, why):
    """Only the first token may report null: it was sampled by prefill, so no
    decode-side value exists and the router fills it from the prefill reply.

    Anywhere else a null is a node that cannot produce what was asked for, and
    the length check alone passed it through: `build_logprobs` turned it into
    -9999.0, the value OpenAI documents for "very unlikely", so the client read
    an engine fault as a measurement. This is the invariant the decode server
    already states by raising `LogprobsUnavailable` past position 0; the router
    is where an older or faulty node has to be caught.
    """
    kw = {"stop": stop} if stop else {}
    r = _post(null_logprobs_router, logprobs=True, top_logprobs=1, temperature=0.6, **kw)
    assert r.status_code == 501, (why, r.text[:200])
    assert r.json()["error_type"] == "logprobs_unavailable", why


@pytest.fixture(scope="module")
def no_candidates_router(_proxy_off):
    """A node that answers with `lp` but no `tp`."""
    vllm_port = _serve(_make_vllm())
    decode_port = _serve(_make_decode(drop_tp=True))
    node = DecodeNode("127.0.0.1", 5556, decode_port)
    ctx = RouterCtx(f"http://127.0.0.1:{vllm_port}", Pool([node]), _Tok(), "none")
    return f"http://127.0.0.1:{_serve(build_app(ctx))}"


def test_missing_candidate_rows_are_refused_not_padded(no_candidates_router):
    """A client that asked for `top_logprobs: 1` and got empty rows was told the request succeeded.

    An absent positional row also loses the alignment, so it is not the same thing as a
    genuinely empty row.
    """
    r = _post(no_candidates_router, stop=["STOP"], logprobs=True, top_logprobs=1, temperature=0.6)
    assert r.status_code == 501, r.text[:200]
    assert r.json()["error_type"] == "logprobs_unavailable"


def test_no_candidates_asked_for_means_empty_rows_are_correct(no_candidates_router):
    """`logprobs: true` without `top_logprobs` resolves to `top_n == 0`, and a
    node that sends no `tp` is then answering exactly what was asked.
    """
    r = _post(no_candidates_router, stop=["STOP"], logprobs=True, temperature=0.6)
    assert r.status_code == 200, r.text[:200]
    ents = r.json()["choices"][0]["logprobs"]["content"]
    assert ents and all(e["top_logprobs"] == [] for e in ents)


# --------------------------------------------------------------------------- #
# The constraint: the one shape that is refused rather than guessed
# --------------------------------------------------------------------------- #
class _Session:
    """The minimum an output parser session has to be: everything is content."""

    def feed(self, text):
        return [{"kind": "content", "text": text}]

    def finish(self):
        return []


class _Parsing:
    """A router context that has an output parser, like `--parser glm47`."""

    def stream(self):
        return _Session()


@pytest.fixture(scope="module")
def parsing_router(_proxy_off):
    vllm_port = _serve(_make_vllm())
    decode_port = _serve(_make_decode())
    node = DecodeNode("127.0.0.1", 5556, decode_port)
    ctx = RouterCtx(f"http://127.0.0.1:{vllm_port}", Pool([node]), _Tok(), "none")
    ctx._parsers = {True: _Parsing(), False: _Parsing()}  # pretend --parser glm47
    return f"http://127.0.0.1:{_serve(build_app(ctx))}"


@pytest.mark.parametrize(
    "over,want,why",
    [
        (
            {"stop": ["STOP"], "logprobs": True, "top_logprobs": 1, "temperature": 0.6},
            501,
            "all three: undecidable, so refused",
        ),
        ({"stop": ["STOP"]}, 200, "stop alone"),
        (
            {"logprobs": True, "top_logprobs": 1, "temperature": 0.6},
            200,
            "parser with logprobs but no stop: nothing is held back",
        ),
    ],
)
def test_only_stop_with_a_parser_and_logprobs_is_refused(parsing_router, over, want, why):
    """`logprobs` covers `message.content`, so an entry has to be attributed to a
    channel, and that is exact only while the parser is fed one token's text at a
    time. A stop makes the router hold back, so what reaches the parser spans
    token boundaries -- and for a token whose text the parser did not emit there
    is no signal to decide the channel.

    Refusing is the design decision, and the other rows are why it is narrow: the
    combinations that need no arithmetic are all served.
    """
    r = _post(parsing_router, **over)
    assert r.status_code == want, f"{why}: got {r.status_code} {r.text[:160]}"
    if want == 501:
        assert r.json()["error_type"] == "capability_unavailable"


def test_the_refusal_happens_before_any_backend_work(parsing_router):
    """501 costs nothing: no prefill, no KV transfer, no decode slot."""
    _SEEN["stream"] = None
    r = _post(parsing_router, stop=["STOP"], logprobs=True, top_logprobs=1, temperature=0.6)
    assert r.status_code == 501
    assert _SEEN["stream"] is None, "the decode node was contacted"


def test_a_chunks_entries_never_run_ahead_of_its_own_text(router):
    """An entry belongs to the chunk carrying its token's text, not an earlier
    one.

    With a stop configured the router holds back `len(stop)-1` characters, so a
    chunk's text can end part-way into a token -- and that token's entry has to
    wait for the chunk that finishes it. Releasing entries as soon as they exist
    keeps the totals right and still tells the client that a token's probability
    describes text it has not been sent yet.

    Checked as a prefix relation over every chunk boundary, which is the strongest
    statement that survives a chunk ending mid-token.
    """
    text = entries_text = ""
    with httpx.Client(timeout=30, trust_env=False) as c:
        with c.stream(
            "POST",
            f"{router}/v1/chat/completions",
            json=_body(stream=True, stop=["ZZZZ"], logprobs=True, top_logprobs=1, temperature=0.6),
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line.startswith("data: ") or line[6:] == "[DONE]":
                    continue
                for ch in json.loads(line[6:]).get("choices", []):
                    text += ch["delta"].get("content", "") or ""
                    for e in (ch.get("logprobs") or {}).get("content") or []:
                        entries_text += e["token"]
                    assert text.startswith(entries_text) or entries_text.startswith(text), (
                        f"entries ran ahead: entries={entries_text!r} " f"text={text!r}"
                    )
                    assert len(entries_text) <= len(text), (
                        f"entries describe {len(entries_text)} characters but "
                        f"only {len(text)} have been sent"
                    )
    assert entries_text == text, "and they agree once the stream ends"


def test_a_streamed_logprobs_line_the_router_cannot_use_fails_closed(no_candidates_router):
    """The non-streaming path answers 501; the streaming path cannot.

    The 200 headers are already out by the time the node's first token line
    arrives, so the status is spent. Padding the missing candidate rows would
    stream invented sentinels to a client that asked for real ones, and
    truncating the stream would leave it without a terminator. The only honest
    ending is the reply's own text, a finish_reason, a typed error event, and
    `[DONE]` -- the same shape the grammar-violation branch uses.
    """
    finish, err, saw_done, entries = None, None, False, 0
    with httpx.Client(timeout=30, trust_env=False) as c:
        with c.stream(
            "POST",
            f"{no_candidates_router}/v1/chat/completions",
            json=_body(stream=True, stop=["STOP"], logprobs=True, top_logprobs=1, temperature=0.6),
        ) as r:
            assert r.status_code == 200, "the status was spent before this"
            for line in r.iter_lines():
                if not line.startswith("data: "):
                    continue
                if line[6:] == "[DONE]":
                    saw_done = True
                    continue
                payload = json.loads(line[6:])
                if payload.get("error"):
                    err = payload["error"]
                    continue
                for ch in payload.get("choices", []):
                    entries += len((ch.get("logprobs") or {}).get("content") or [])
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
    assert err is not None, "the failure is reported"
    assert err["error_type"] == "logprobs_unavailable", err
    assert finish == "stop", "the choice is closed before the error event"
    assert saw_done, "and the stream is terminated"
    assert entries == 0, "no invented sentinels reached the client"


def test_a_streamed_reply_that_ends_without_a_terminal_message_fails_closed(truncating_router):
    """The same refusal the non-streaming path makes, in SSE form.

    A body that reaches a clean EOF before `done`, `error` or a stop match is a
    truncated generation. Emitting a normal finish chunk and `[DONE]` would
    report it as complete; the non-streaming path answers 502 `decode_truncated`
    and this one says the same thing with the only means left after the 200.
    """
    finish, err, saw_done = None, None, False
    with httpx.Client(timeout=30, trust_env=False) as c:
        with c.stream(
            "POST",
            f"{truncating_router}/v1/chat/completions",
            json=_body(stream=True, stop=["ZZZZ"]),
        ) as r:
            assert r.status_code == 200
            for line in r.iter_lines():
                if not line.startswith("data: "):
                    continue
                if line[6:] == "[DONE]":
                    saw_done = True
                    continue
                payload = json.loads(line[6:])
                if payload.get("error"):
                    err = payload["error"]
                    continue
                for ch in payload.get("choices", []):
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
    assert err is not None and err["error_type"] == "decode_truncated", err
    assert finish == "stop", "the choice is closed before the error event"
    assert saw_done


def test_a_streamed_stop_match_is_still_a_success(truncating_router):
    """The stop leaves the loop early on purpose, so it stays a normal stream."""
    err, text = None, ""
    with httpx.Client(timeout=30, trust_env=False) as c:
        with c.stream(
            "POST",
            f"{truncating_router}/v1/chat/completions",
            json=_body(stream=True, stop=["STOP"]),
        ) as r:
            for line in r.iter_lines():
                if not line.startswith("data: ") or line[6:] == "[DONE]":
                    continue
                payload = json.loads(line[6:])
                if payload.get("error"):
                    err = payload["error"]
                for ch in payload.get("choices", []):
                    text += ch["delta"].get("content", "") or ""
    assert err is None, f"a matched stop is not a failure: {err}"
    assert text == "Hello, world. "


def test_a_cancelled_finish_reason_is_normalised_on_both_protocol_forms():
    """`cancelled` is the router's own doing, not a client-visible outcome.

    The streaming form normalised it and the blocking form did not, so a cancel
    landing while a non-streaming request was in flight surfaced the decode
    protocol's internal reason as `choices[0].finish_reason`.
    """
    from tilert.pd_vllm.decode_response import DecodeReader

    blocking = DecodeReader()
    blocking.feed_blocking({"token_ids": [], "timing_ms": {"finish_reason": "cancelled"}})
    assert blocking.finish_reason == "stop"

    streamed = DecodeReader()
    streamed.feed(json.dumps({"done": True, "finish_reason": "cancelled"}))
    assert streamed.finish_reason == "stop"

    kept = DecodeReader()
    kept.feed_blocking({"token_ids": [], "timing_ms": {"finish_reason": "length"}})
    assert kept.finish_reason == "length", "only `cancelled` is translated"


# --------------------------------------------------------------------------- #
# Failing closed: every way out after the 200 is spent
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def garbage_router(_proxy_off):
    """A node that sends a line `DecodeReader.feed` cannot parse."""
    vllm_port = _serve(_make_vllm())
    decode_port = _serve(_make_decode(garbage_after=4))
    node = DecodeNode("127.0.0.1", 5556, decode_port)
    ctx = RouterCtx(f"http://127.0.0.1:{vllm_port}", Pool([node]), _Tok(), "none")
    return f"http://127.0.0.1:{_serve(build_app(ctx))}"


def test_a_stream_that_raises_mid_flight_still_terminates(garbage_router):
    """A malformed line reaches the generator's `except`, not the EOF check.

    The 200 is spent by then, so exiting on the exception leaves the client with
    a partial response and no terminator -- the same failure the clean-EOF check
    refuses, reached by a different route.
    """
    finish, err, saw_done = None, None, False
    with httpx.Client(timeout=30, trust_env=False) as c:
        with c.stream(
            "POST", f"{garbage_router}/v1/chat/completions", json=_body(stream=True, stop=["ZZZZ"])
        ) as r:
            assert r.status_code == 200
            for line in r.iter_lines():
                if not line.startswith("data: "):
                    continue
                if line[6:] == "[DONE]":
                    saw_done = True
                    continue
                payload = json.loads(line[6:])
                if payload.get("error"):
                    err = payload["error"]
                    continue
                for ch in payload.get("choices", []):
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
    assert err is not None, "the failure is reported"
    assert err["error_type"] == "decode_stream_failed", err
    assert finish == "stop", "the choice is closed first"
    assert saw_done, "and the stream is terminated"


def test_a_decode_post_that_raises_still_cancels_the_node(monkeypatch, _proxy_off):
    """The node may have admitted the request before the call failed.

    A timeout or a reset while waiting for headers never reaches the reader's
    `finally`, so without covering the POST phase the node holds its slot until
    its own timeout -- and #41 exists because a node holding a slot surfaces as a
    429 for whoever comes next.
    """
    from fastapi.testclient import TestClient

    cancelled = []
    real_post = pd_router.requests.post

    def flaky_post(url, **kw):
        if url.endswith("/pd/decode"):
            raise ConnectionResetError("reset while waiting for headers")
        if url.endswith("/pd/cancel"):
            cancelled.append(kw.get("json", {}).get("rid"))
            return real_post(url, **kw)
        return real_post(url, **kw)

    vllm_port = _serve(_make_vllm())
    decode_port = _serve(_make_decode())
    node = DecodeNode("127.0.0.1", 5556, decode_port)
    ctx = RouterCtx(f"http://127.0.0.1:{vllm_port}", Pool([node]), _Tok(), "none")
    client = TestClient(build_app(ctx))
    monkeypatch.setattr(pd_router.requests, "post", flaky_post)
    r = client.post("/v1/chat/completions", json=_body(stop=["STOP"]))
    assert r.status_code == 502, r.text[:200]
    for _ in range(200):
        if cancelled:
            break
        time.sleep(0.02)
    assert cancelled, "the node was left holding its slot"


@pytest.fixture(scope="module")
def lossy_router(_proxy_off):
    """A node whose `done` count exceeds the token lines that arrived."""
    vllm_port = _serve(_make_vllm())
    decode_port = _serve(_make_decode(filler=6, lose_lines_after=3))
    node = DecodeNode("127.0.0.1", 5556, decode_port)
    ctx = RouterCtx(f"http://127.0.0.1:{vllm_port}", Pool([node]), _Tok(), "none")
    return f"http://127.0.0.1:{_serve(build_app(ctx))}"


def test_a_terminal_count_that_disagrees_is_a_truncated_decode(lossy_router):
    """The node states its own count on the terminal line; a mismatch means
    token lines were lost.

    It still sends a normal `done`, so nothing else in the exchange looks wrong:
    accepting it reports a shortened generation with a successful finish reason
    and an understated `usage`.
    """
    r = _post(lossy_router, stop=["ZZZZ"])
    assert r.status_code == 502, r.text[:200]
    assert r.json()["error_type"] == "decode_truncated"


def test_the_two_refusals_do_not_share_a_status(lossy_router, no_candidates_router):
    """501 is "the router cannot serve this"; 502 is "the node sent something incomplete".

    Collapsing them would tell a caller to change its request when the backend is at fault.
    """
    incomplete = _post(lossy_router, stop=["ZZZZ"])
    unusable = _post(
        no_candidates_router, stop=["STOP"], logprobs=True, top_logprobs=1, temperature=0.6
    )
    assert incomplete.status_code == 502, incomplete.text[:120]
    assert unusable.status_code == 501, unusable.text[:120]


@pytest.mark.parametrize("value", ["bad", 5, [1, 2]])
def test_a_malformed_chat_template_kwargs_is_a_400(router, value):
    """vLLM declares it `dict[str, Any] | None` and answers 422 for anything else.

    Reading `.get` off a string raised AttributeError instead, which the typed handlers do not
    catch -- so a client mistake surfaced as a 500.
    """
    r = _post(router, chat_template_kwargs=value)
    assert r.status_code == 400, f"got {r.status_code} {r.text[:160]}"


def test_a_well_formed_chat_template_kwargs_still_passes(router):
    for value in ({}, {"enable_thinking": False}, None):
        r = _post(router, chat_template_kwargs=value)
        assert r.status_code == 200, f"{value!r}: {r.status_code} {r.text[:120]}"


# --------------------------------------------------------------------------- #
# A typed node error keeps its status on both protocols
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "error_type,want",
    [
        ("logprobs_unavailable", 501),
        ("capability_unavailable", 501),
        ("invalid_grammar", 400),
        ("invalid_parameter", 400),
        ("request_cancelled", 499),
        ("grammar_backend_unavailable", 500),
        (None, 502),
        ("something_new", 502),
    ],
)
def test_a_typed_node_error_keeps_the_status_it_deserves(_proxy_off, error_type, want):
    """The two protocols carry the type differently and must not disagree.

    Over the blocking protocol the node answers an HTTP status and the router
    forwards it. Over the streaming one the error arrives inside a 200 body --
    the status is already spent -- so the router reconstructs it from the type.
    Mapping every propagated type to one status is how `logprobs_unavailable`
    came back 400 here and 501 there for the same inability, and adding a `stop`
    string is what moves a non-streaming request onto this protocol.

    An unclassified error stays a 502: the node did not say what went wrong, so
    a component fault is the honest answer.
    """
    vllm_port = _serve(_make_vllm())
    decode_port = _serve(_make_decode(error_after=3, error_type=error_type))
    node = DecodeNode("127.0.0.1", 5556, decode_port)
    ctx = RouterCtx(f"http://127.0.0.1:{vllm_port}", Pool([node]), _Tok(), "none")
    url = f"http://127.0.0.1:{_serve(build_app(ctx))}"
    r = _post(url, stop=["ZZZZ"])  # stop -> the streaming protocol
    assert r.status_code == want, f"{error_type}: got {r.status_code}"
    if error_type is not None:
        assert r.json().get("error_type") == error_type


@pytest.mark.parametrize(
    "error_type,inline",
    [
        ("logprobs_unavailable", False),
        ("capability_unavailable", False),
        ("grammar_violation", False),
        (None, True),
    ],
)
def test_a_streamed_node_error_fails_closed_when_it_was_classified(_proxy_off, error_type, inline):
    """A typed error is the node's answer about the contract, not more content.

    Emitting it as a `[decode error: ...]` chunk and then a normal finish reports
    a successful completion that broke what the request asked for -- unconstrained
    output for a grammar, or a reply without the logprobs it requested. An
    UNCLASSIFIED error keeps the inline marker: the node did not say what went
    wrong, and a visible marker beats a bare error event for a client that is
    already rendering text.
    """
    vllm_port = _serve(_make_vllm())
    decode_port = _serve(_make_decode(error_after=3, error_type=error_type))
    node = DecodeNode("127.0.0.1", 5556, decode_port)
    ctx = RouterCtx(f"http://127.0.0.1:{vllm_port}", Pool([node]), _Tok(), "none")
    url = f"http://127.0.0.1:{_serve(build_app(ctx))}"

    text, err, saw_done = "", None, False
    with httpx.Client(timeout=30, trust_env=False) as c:
        with c.stream("POST", f"{url}/v1/chat/completions", json=_body(stream=True)) as r:
            assert r.status_code == 200
            for line in r.iter_lines():
                if not line.startswith("data: "):
                    continue
                if line[6:] == "[DONE]":
                    saw_done = True
                    continue
                payload = json.loads(line[6:])
                if payload.get("error"):
                    err = payload["error"]
                    continue
                for ch in payload.get("choices", []):
                    text += ch["delta"].get("content", "") or ""
    assert saw_done, "the stream terminates either way"
    if inline:
        assert "[decode error:" in text, "an unclassified error stays inline"
        assert err is None
    else:
        assert err is not None and err.get("error_type") == error_type, err
        assert "[decode error:" not in text, "a classified error must not be dressed up as content"
