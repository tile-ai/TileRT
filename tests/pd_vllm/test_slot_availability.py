"""A decode node's single slot must free promptly, and "busy" must reach the
client as 429.

Two failures that share a cause -- the slot outliving the request that needed it:

* **Cancel during the KV transfer was a no-op.** `cancel_event` was created only
  once decoding began, so a cancel arriving during the wire-wait found it `None`
  and answered 404 while the wait ran to `timeout_s` (120 s by default). With
  bs=1 that took the whole node out of service for two minutes every time a
  client hung up early.
* **A busy decode node reached the client as 502.** `raise_for_status` turned
  "retry shortly" into "a component is broken", which is also what the release
  notes then told the operator to go and do.

CPU only -- no GPU, no tilert, no vLLM.

Run:
  CUDA_VISIBLE_DEVICES= python -m pytest \
      tests/pd_vllm/test_slot_availability.py -v
"""

import json
import queue
import threading
import time
import types
from unittest import mock

import pytest
from fastapi.testclient import TestClient

from tilert.pd_vllm import decode_pool, decode_server, pd_router
from tilert.pd_vllm.decode_pool import DecodeNode, NodeLease, Pool
from tilert.pd_vllm.decode_response import (
    TYPED_ERROR,
    DecodeReader,
    terminal_verdict,
)
from tilert.pd_vllm.decode_server import build_app as build_decode_app
from tilert.pd_vllm.engine_iface import StubEngine


# --------------------------------------------------------------------------- #
# decode server: the wire-wait observes the cancel
# --------------------------------------------------------------------------- #
class _FakeReq:
    rid = "rid-1"
    seq_len = 8
    last_prompt_token = 5


class _SlowServer:
    """A receive server whose KV never arrives, so /pd/decode sits in the wait."""

    def __init__(self, deliver_after=None):
        self.completed: queue.Queue = queue.Queue()
        self.profile = types.SimpleNamespace(
            convert=lambda *a, **k: "converted", num_ranks=8, name="stub"
        )
        self.buffer = None
        self.base_ptr = 0
        self.max_seq_len = 4096
        self.released = 0
        if deliver_after is not None:
            threading.Timer(deliver_after, lambda: self.completed.put(_FakeReq())).start()

    def expect(self, rid=None):
        # /pd/decode announces its rid so the real ReceiveServer can drop a
        # tombstone left by a previous attempt at the same request. Recorded,
        # so a test can assert the announcement happened.
        self.expected_rids = getattr(self, "expected_rids", [])
        self.expected_rids.append(rid)

    def release(self, rid=None):
        # Scoped like the real ReceiveServer.release: the decode server
        # names the rid it owns, because the slot may since have been
        # handed to a later request.
        self.released_rids = getattr(self, "released_rids", [])
        self.released_rids.append(rid)
        self.released += 1


def _decode_client(server=None, engine=None):
    return TestClient(build_decode_app(server or _SlowServer(), engine or StubEngine()))


def test_a_cancel_during_the_wire_wait_returns_promptly():
    """The whole point: this used to block for timeout_s."""
    server = _SlowServer()
    client = _decode_client(server)
    result = {}

    def _decode():
        t0 = time.time()
        r = client.post(
            "/pd/decode",
            json={"rid": "rid-1", "first_token_id": 7, "max_tokens": 8, "timeout_s": 30.0},
        )
        result["status"] = r.status_code
        result["body"] = r.json()
        result["elapsed"] = time.time() - t0

    t = threading.Thread(target=_decode)
    t.start()
    # Let it reach the wire-wait, then hang up.
    time.sleep(0.3)
    assert client.post("/pd/cancel", json={"rid": "rid-1"}).status_code == 200
    t.join(timeout=10)
    assert not t.is_alive(), "the wire-wait ignored the cancel"
    assert (
        result["elapsed"] < 5
    ), f"took {result['elapsed']:.1f}s; the cancel should land within one poll"
    assert result["status"] == 499
    assert result["body"]["error_type"] == "request_cancelled"


def test_a_cancel_during_the_wire_wait_frees_the_slot():
    """Otherwise the next request is turned away with 429 for two minutes."""
    server = _SlowServer()
    client = _decode_client(server)
    t = threading.Thread(
        target=lambda: client.post(
            "/pd/decode",
            json={"rid": "rid-1", "first_token_id": 7, "max_tokens": 8, "timeout_s": 30.0},
        )
    )
    t.start()
    time.sleep(0.3)
    client.post("/pd/cancel", json={"rid": "rid-1"})
    t.join(timeout=10)
    assert client.get("/decode_status").json()["status"] == "idle"
    assert server.released >= 1


def test_cancel_is_accepted_from_the_moment_the_request_is_admitted():
    """It used to 404 until decoding began, which is after the wire-wait."""
    server = _SlowServer()
    client = _decode_client(server)
    t = threading.Thread(
        target=lambda: client.post(
            "/pd/decode",
            json={"rid": "rid-1", "first_token_id": 7, "max_tokens": 8, "timeout_s": 30.0},
        )
    )
    t.start()
    time.sleep(0.3)
    r = client.post("/pd/cancel", json={"rid": "rid-1"})
    assert r.status_code == 200, r.json()
    assert r.json()["cancelled"] == "rid-1"
    t.join(timeout=10)


def test_a_cancel_for_another_rid_is_still_a_404():
    server = _SlowServer()
    client = _decode_client(server)
    t = threading.Thread(
        target=lambda: client.post(
            "/pd/decode",
            json={"rid": "rid-1", "first_token_id": 7, "max_tokens": 8, "timeout_s": 30.0},
        )
    )
    t.start()
    time.sleep(0.3)
    assert client.post("/pd/cancel", json={"rid": "other"}).status_code == 404
    client.post("/pd/cancel", json={"rid": "rid-1"})
    t.join(timeout=10)


def test_an_uncancelled_wait_still_times_out_as_504():
    """The cancel path must not swallow the timeout it shares a loop with: a
    504 points at the RDMA path, a cancel means the client left.
    """
    r = _decode_client(_SlowServer()).post(
        "/pd/decode", json={"rid": "rid-1", "first_token_id": 7, "max_tokens": 8, "timeout_s": 0.6}
    )
    assert r.status_code == 504
    assert r.json()["error"] == "kv_transfer_timeout"


def test_kv_that_arrives_before_any_cancel_is_served():
    server = _SlowServer(deliver_after=0.2)
    r = _decode_client(server).post(
        "/pd/decode", json={"rid": "rid-1", "first_token_id": 7, "max_tokens": 4, "timeout_s": 30.0}
    )
    assert r.status_code == 200
    assert r.json()["token_ids"][0] == 7


def test_the_cancel_sentinel_is_distinct_from_a_timeout():
    """Two different answers from one loop; conflating them would report a
    client hang-up as an RDMA fault.
    """
    assert decode_server._CANCELLED is not None
    assert decode_server._CANCELLED is not False


# --------------------------------------------------------------------------- #
# router: a busy decode node is retried once, then surfaced as 429
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
    "id": "cmpl-abc",
    "choices": [{"logprobs": {"content": [{"token": "token_id:7"}]}}],
    "usage": {"prompt_tokens": 3},
    "model": "m",
}
DECODED = {"rid": "x", "token_ids": [7], "seq_len": 8, "timing_ms": {"finish_reason": "stop"}}
BODY = {"messages": [{"role": "user", "content": "hi"}]}


def _router(monkeypatch, decode_statuses):
    """Router whose decode node answers with each status in turn.

    Returns (client, calls) where calls counts /pd/decode attempts.
    """
    statuses = list(decode_statuses)
    calls = {"decode": 0}

    monkeypatch.setattr(
        pd_router.requests,
        "get",
        lambda url, timeout=None, **kw: _Resp(
            {"capabilities": {"penalties": True, "ignore_eos": True}}
        ),
    )
    monkeypatch.setattr(pd_router.time, "sleep", lambda s: None)

    def fake_post(url, json=None, timeout=None, **kw):
        if url.endswith("/pd/decode"):
            calls["decode"] += 1
            status = statuses.pop(0) if statuses else 200
            if status == 200:
                return _Resp(DECODED)
            return _Resp({"error": "busy", "current_rid": "other"}, status)
        return _Resp(PREFILL)

    monkeypatch.setattr(pd_router.requests, "post", fake_post)
    pool = pd_router.Pool([pd_router.DecodeNode("127.0.0.1", 5556, 5557)])
    ctx = pd_router.RouterCtx("http://vllm.invalid", pool, tokenizer=None, parser_name="none")
    return TestClient(pd_router.build_app(ctx)), calls, pool


def test_a_transient_busy_is_retried_and_succeeds(monkeypatch):
    """The router frees its own reservation as soon as it stops reading, while
    the node's slot unwinds a little later -- a request dispatched into that
    window meets the node's admission.
    """
    client, calls, _ = _router(monkeypatch, [429, 200])
    r = client.post("/v1/chat/completions", json=BODY)
    assert r.status_code == 200
    assert calls["decode"] == 2


def test_a_node_still_busy_after_the_retry_is_429_not_502(monkeypatch):
    """429 is retryable and truthful; 502 sends the operator to restart a
    healthy component.
    """
    client, calls, _ = _router(monkeypatch, [429, 429])
    r = client.post("/v1/chat/completions", json=BODY)
    assert r.status_code == 429
    assert r.json()["error_type"] == "decode_busy"
    assert calls["decode"] == pd_router._DECODE_BUSY_ATTEMPTS


def test_the_retry_is_bounded(monkeypatch):
    """One short retry absorbs the handover window; more would mask a node that
    is genuinely stuck behind a status the client can act on.
    """
    client, calls, _ = _router(monkeypatch, [429] * 10)
    client.post("/v1/chat/completions", json=BODY)
    assert calls["decode"] == 2


def test_a_busy_node_is_returned_to_the_pool(monkeypatch):
    """A 429 must not leak the reservation, or the pool drains one node per
    busy reply.
    """
    client, _, pool = _router(monkeypatch, [429, 429])
    client.post("/v1/chat/completions", json=BODY)
    assert all(not n.busy for n in pool.nodes)


def test_a_healthy_node_is_not_retried(monkeypatch):
    client, calls, _ = _router(monkeypatch, [200])
    assert client.post("/v1/chat/completions", json=BODY).status_code == 200
    assert calls["decode"] == 1


def test_a_non_busy_decode_failure_is_still_502(monkeypatch):
    """The retry is for 429 only; an untyped 500 is a component call that did
    not work, and saying so is correct.
    """
    client, calls, _ = _router(monkeypatch, [500])
    r = client.post("/v1/chat/completions", json=BODY)
    assert r.status_code == 502
    assert calls["decode"] == 1


def test_a_typed_decode_error_still_propagates(monkeypatch):
    """The busy branch must not swallow the classified statuses."""

    def fake_post(url, json=None, timeout=None, **kw):
        if url.endswith("/pd/decode"):
            return _Resp({"error": "no", "error_type": "invalid_grammar"}, 400)
        return _Resp(PREFILL)

    monkeypatch.setattr(
        pd_router.requests, "get", lambda url, timeout=None, **kw: _Resp({"capabilities": {}})
    )
    monkeypatch.setattr(pd_router.requests, "post", fake_post)
    pool = pd_router.Pool([pd_router.DecodeNode("127.0.0.1", 5556, 5557)])
    ctx = pd_router.RouterCtx("http://vllm.invalid", pool, tokenizer=None, parser_name="none")
    client = TestClient(pd_router.build_app(ctx))
    r = client.post("/v1/chat/completions", json=BODY)
    assert r.status_code == 400
    assert r.json()["error_type"] == "invalid_grammar"


def test_request_cancelled_is_propagated_not_masked():
    """The decode node's 499 says the caller asked it to stop; flattening that
    into a 502 would read as a component fault.

    Asserted through the verdict both response paths take, rather than by
    membership in a table: the table is where the answer comes from, but the
    verdict is what the handlers act on.
    """
    reader = DecodeReader(stream=None, logprobs_req=None, rid="rid-1")
    reader.feed(json.dumps({"error": "client left", "error_type": "request_cancelled"}))
    verdict, payload, status = terminal_verdict(reader)
    assert verdict == TYPED_ERROR, verdict
    assert status == 499, status
    assert payload["error_type"] == "request_cancelled"


# --------------------------------------------------------------------------- #
# Streaming: the status is decided before the response begins
# --------------------------------------------------------------------------- #
#
# The decode request is now sent BEFORE StreamingResponse is returned. Inside the
# generator the response has already begun and the status is spent, so a busy node
# could only have been reported as an SSE error inside a 200 -- or, as it was, a
# stream truncated with no terminator at all. These run against real servers so
# the httpx streaming path is the real one.
import json as _json  # noqa: E402

import httpx as _httpx  # noqa: E402
import uvicorn as _uvicorn  # noqa: E402
from fastapi import FastAPI as _FastAPI  # noqa: E402


def _serve(app):
    cfg = _uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error")
    server = _uvicorn.Server(cfg)
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(200):
        if server.started:
            return server.servers[0].sockets[0].getsockname()[1]
        time.sleep(0.05)
    raise RuntimeError("stub server did not start")


@pytest.fixture
def no_proxy():
    mp = pytest.MonkeyPatch()
    for var in ("no_proxy", "NO_PROXY"):
        mp.setenv(var, "127.0.0.1,localhost")
    for var in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        mp.delenv(var, raising=False)
    yield
    mp.undo()


class _StubTokenizer:
    """The streaming path detokenises every token, so it needs one."""

    _VOCAB = {7: "hi", 11: " there"}

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self._VOCAB.get(i, "") for i in ids)


def _stream_stack(decode_statuses):
    """vLLM stub + a decode stub answering `decode_statuses` in turn + router."""
    statuses = list(decode_statuses)
    calls = {"decode": 0}

    vllm = _FastAPI()

    @vllm.post("/v1/chat/completions")
    async def _prefill():
        return PREFILL

    decode = _FastAPI()

    @decode.get("/capabilities")
    def _caps():
        return {"capabilities": {"penalties": True, "ignore_eos": True, "logprobs": True}}

    @decode.post("/pd/decode")
    def _dec():
        from fastapi.responses import JSONResponse, StreamingResponse

        calls["decode"] += 1
        status = statuses.pop(0) if statuses else 200
        if status != 200:
            return JSONResponse({"error": "busy", "current_rid": "other"}, status_code=status)

        def _body():
            yield _json.dumps({"t": [11]}) + "\n"
            yield _json.dumps(
                {"done": True, "n": 1, "seq_len": 8, "finish_reason": "stop", "timing_ms": {}}
            ) + "\n"

        return StreamingResponse(_body(), media_type="application/x-ndjson")

    @decode.post("/pd/cancel")
    def _cancel(b: dict):
        return {"cancelled": b.get("rid")}

    vllm_port = _serve(vllm)
    decode_port = _serve(decode)
    node = pd_router.DecodeNode("127.0.0.1", 5556, decode_port)
    pool = pd_router.Pool([node])
    ctx = pd_router.RouterCtx(
        f"http://127.0.0.1:{vllm_port}", pool, tokenizer=_StubTokenizer(), parser_name="none"
    )
    router_port = _serve(pd_router.build_app(ctx))
    return f"http://127.0.0.1:{router_port}", calls, pool


STREAM_BODY = {"messages": [{"role": "user", "content": "hi"}], "stream": True}


def test_a_streamed_request_gets_a_real_429_not_a_truncated_stream(no_proxy):
    """It used to raise inside the generator, so the client got a 200 whose body
    simply stopped -- no finish_reason, no [DONE], nothing to distinguish it from
    a network cut.
    """
    url, calls, _ = _stream_stack([429, 429])
    with _httpx.Client(timeout=30, trust_env=False) as c:
        r = c.post(f"{url}/v1/chat/completions", json=STREAM_BODY)
    assert r.status_code == 429
    assert r.json()["error_type"] == "decode_busy"
    assert calls["decode"] == 2


def test_a_streamed_request_retries_a_transient_busy(no_proxy):
    url, calls, _ = _stream_stack([429, 200])
    chunks = []
    with _httpx.Client(timeout=30, trust_env=False) as c:
        with c.stream("POST", f"{url}/v1/chat/completions", json=STREAM_BODY) as r:
            assert r.status_code == 200
            for line in r.iter_lines():
                if line.startswith("data: "):
                    chunks.append(line[6:])
    assert chunks[-1] == "[DONE]"
    assert calls["decode"] == 2


def test_a_streamed_request_still_streams_normally(no_proxy):
    """The restructure moved the send out of the generator; the happy path must
    be untouched.
    """
    url, calls, _ = _stream_stack([200])
    seen = []
    with _httpx.Client(timeout=30, trust_env=False) as c:
        with c.stream("POST", f"{url}/v1/chat/completions", json=STREAM_BODY) as r:
            assert r.status_code == 200
            for line in r.iter_lines():
                if line.startswith("data: ") and line[6:] != "[DONE]":
                    seen.append(_json.loads(line[6:]))
    assert seen, "no chunks at all"
    assert any(
        c["choices"] and c["choices"][0].get("finish_reason") for c in seen
    ), "no terminating chunk"
    assert calls["decode"] == 1


def test_a_busy_streamed_request_returns_the_node_to_the_pool(no_proxy):
    url, _, pool = _stream_stack([429, 429])
    with _httpx.Client(timeout=30, trust_env=False) as c:
        c.post(f"{url}/v1/chat/completions", json=STREAM_BODY)
    assert all(not n.busy for n in pool.nodes)


def test_a_streamed_typed_decode_error_keeps_its_status(no_proxy):
    """A classified 400 must survive the new pre-flight, not become a 502."""
    statuses = [400]
    vllm = _FastAPI()

    @vllm.post("/v1/chat/completions")
    async def _prefill():
        return PREFILL

    decode = _FastAPI()

    @decode.get("/capabilities")
    def _caps():
        return {"capabilities": {}}

    @decode.post("/pd/decode")
    def _dec():
        from fastapi.responses import JSONResponse

        statuses.pop(0)
        return JSONResponse({"error": "bad", "error_type": "invalid_grammar"}, status_code=400)

    vllm_port = _serve(vllm)
    decode_port = _serve(decode)
    ctx = pd_router.RouterCtx(
        f"http://127.0.0.1:{vllm_port}",
        pd_router.Pool([pd_router.DecodeNode("127.0.0.1", 5556, decode_port)]),
        tokenizer=_StubTokenizer(),
        parser_name="none",
    )
    port = _serve(pd_router.build_app(ctx))
    with _httpx.Client(timeout=30, trust_env=False) as c:
        r = c.post(f"http://127.0.0.1:{port}/v1/chat/completions", json=STREAM_BODY)
    assert r.status_code == 400
    assert r.json()["error_type"] == "invalid_grammar"


# --------------------------------------------------------------------------- #
# Review findings on PR #41 (codex)
# --------------------------------------------------------------------------- #
def test_release_is_scoped_to_the_rid_that_owns_the_slot():
    """A caller only knows about its own request, and the slot may have moved on.

    Reachable whenever a transfer arrives after its consumer gave up: the entry
    is enqueued with nobody waiting, the NEXT request drains it as unmatched, and
    an unscoped release there would free that next request's own tenancy —
    which then never completes, because a cancelled tenancy drops its ranks'
    `done` messages.
    """
    import types as _t

    from tilert.pd_vllm import receive_server as rs

    srv = object.__new__(rs.ReceiveServer)
    srv._lock = threading.Lock()
    srv._cancelled = {}
    srv.request_timeout = 120.0
    srv._current = rs.ReceivedRequest(
        rid="current",
        seq_len=8,
        last_prompt_token=5,
        first_token_id=None,
        sampling=None,
        state=rs.TRANSFERRING,
        active_writers=1,
    )

    srv.release("someone-else")
    assert srv._current is not None, "released a tenancy it does not own"
    assert srv._current.state == rs.TRANSFERRING, "cancelled the wrong tenancy"

    srv.release("current")
    assert srv._current.state == rs.CANCELLING, "its own release had no effect"
    del _t


def test_a_late_transfer_does_not_disturb_the_next_request():
    """End-to-end shape of the same bug, through /pd/decode.

    rid-A is cancelled before any sender arrives, so nothing is released. Its KV
    then lands with no consumer. rid-B is in flight when the next request drains
    that stale entry — and must survive it.
    """
    server = _SlowServer()
    client = _decode_client(server)

    t = threading.Thread(
        target=lambda: client.post(
            "/pd/decode",
            json={"rid": "rid-A", "first_token_id": 7, "max_tokens": 4, "timeout_s": 30.0},
        )
    )
    t.start()
    time.sleep(0.3)
    client.post("/pd/cancel", json={"rid": "rid-A"})
    t.join(timeout=10)

    # rid-A's transfer arrives late, with nobody waiting for it.
    server.completed.put(_FakeReq())  # _FakeReq.rid == "rid-1"
    # The next request drains it as unmatched; the release must name rid-1, not
    # whatever the slot holds now.
    r = client.post(
        "/pd/decode", json={"rid": "rid-B", "first_token_id": 7, "max_tokens": 4, "timeout_s": 1.0}
    )
    assert r.status_code == 504, r.text
    assert server.released_rids, "nothing was released at all"
    assert "rid-1" in server.released_rids, (
        f"the stale entry was not released by its own rid: " f"{server.released_rids}"
    )


def test_the_streaming_preflight_watches_for_a_disconnect():
    """The decode node holds its headers through the whole KV wire-wait, so the
    preflight `send` can sit for `timeout_s`.

    At that point StreamingResponse does not exist, so neither the generator's
    `finally` nor its `is_disconnected` poll is running — a client that hangs up
    there would hold the router's reservation and the decode slot for the full
    wait, which is the failure this endpoint is supposed to have stopped having.
    """
    import pathlib as _p

    src = _p.Path(pd_router.__file__).read_text()
    # Defined AND used: a helper that exists but is not called on the preflight
    # path leaves the disconnect unwatched, which is the whole bug.
    assert (
        src.count("_send_watching_client") >= 2
    ), "the disconnect-aware helper is defined but never called"
    assert (
        "await _send_watching_client(" in src
    ), "the preflight does not go through the disconnect-aware helper"
    assert (
        "client.send(" not in src.split("for attempt in range")[1][:600]
    ), "the preflight still sends directly, bypassing the disconnect watch"
    assert (
        "request.is_disconnected()" in src.split("async def _send_watching_client")[1][:1200]
    ), "the helper does not poll for a disconnect"
    # CancelledError is a BaseException; suppress(Exception) would let it escape
    # and turn a clean 499 into a 500.
    assert "asyncio.CancelledError" in src


@pytest.mark.parametrize(
    "dispatched,terminated,want_cancel,why",
    [
        (True, False, True, "a request went out and the node never said done"),
        (
            True,
            True,
            False,
            "the node reported done; cancelling could land on the " "NEXT request for that slot",
        ),
        (False, False, False, "nothing went out, so there is nothing to cancel"),
    ],
)
def test_a_lease_cancels_exactly_when_the_node_may_still_be_working(
    dispatched, terminated, want_cancel, why
):
    """Releasing the router's own reservation is not enough -- the decode node is
    a separate process still holding its slot.

    This was spelled out at five call sites across the two handlers, and the
    fixes for it landed at some of them; `NodeLease` is the rule once. It used to
    be checked by grepping pd_router.py for a cancel call, which passed for any
    spelling and failed for a correct rename.
    """
    fired = []
    node = DecodeNode("127.0.0.1", 5556, 5557)
    pool = Pool([node])
    assert pool.acquire() is node
    lease = NodeLease(pool, node)
    lease.rid = "rid-1"
    lease.dispatched = dispatched
    with mock.patch.object(decode_pool, "cancel_decode", lambda n, rid: fired.append(rid)):
        lease.release(terminated=terminated)
        lease.release(terminated=terminated)  # idempotent
        for _ in range(50):
            if fired:
                break
            time.sleep(0.02)
    assert fired == (["rid-1"] if want_cancel else []), why
    assert not node.busy, "the slot goes back exactly once, on every exit"
    assert pool.acquire() is node, "and the node is reusable"


# --------------------------------------------------------------------------- #
# the abandon drain observes the cancel too
# --------------------------------------------------------------------------- #
def test_a_cancel_during_the_abandon_drain_returns_promptly():
    """A rejected request's drain used to ignore the cancel it reported taking.

    When decode-side validation refuses a request its KV may never arrive (the
    prefill leg died, say), so _abandon_pending_kv drains before releasing. That
    drain ran to _ABANDON_DRAIN_S while /pd/cancel answered 200 -- the event is
    armed and this rid is still current -- so the client was told the request was
    cancelled and the next one was refused 429 for another 30 s.
    """
    server = _SlowServer()
    client = _decode_client(server)
    result = {}

    def _decode():
        t0 = time.time()
        # An unknown grammar type is refused post-admission, which is what
        # sends us down the abandon path.
        r = client.post(
            "/pd/decode",
            json={
                "rid": "rid-1",
                "first_token_id": 7,
                "max_tokens": 8,
                "grammar_spec": {"type": "__nope__"},
                "timeout_s": 30.0,
            },
        )
        result["status"] = r.status_code
        result["elapsed"] = time.time() - t0

    t = threading.Thread(target=_decode)
    t.start()
    time.sleep(0.3)  # let it reach the abandon drain
    assert (
        client.post("/pd/cancel", json={"rid": "rid-1"}).status_code == 200
    ), "the cancel was refused, so this test is not exercising the drain"
    t.join(timeout=15)
    assert not t.is_alive(), "the abandon drain ignored the cancel"
    assert result["elapsed"] < 5, (
        f"took {result['elapsed']:.1f}s; a cancelled abandon drain must land "
        f"within one poll, not at _ABANDON_DRAIN_S "
        f"({decode_server._ABANDON_DRAIN_S:.0f}s)"
    )
    # The request is still refused for its own reason -- the cancel only ends
    # the wait, it does not change the answer.
    assert result["status"] == 400, result


def test_the_slot_is_free_after_a_cancelled_abandon_drain():
    server = _SlowServer()
    client = _decode_client(server)
    t = threading.Thread(
        target=lambda: client.post(
            "/pd/decode",
            json={
                "rid": "rid-1",
                "first_token_id": 7,
                "max_tokens": 8,
                "grammar_spec": {"type": "__nope__"},
                "timeout_s": 30.0,
            },
        )
    )
    t.start()
    time.sleep(0.3)
    client.post("/pd/cancel", json={"rid": "rid-1"})
    t.join(timeout=15)
    assert client.get("/decode_status").json()["status"] == "idle"


# --------------------------------------------------------------------------- #
# a rid nobody is waiting for must not open a tenancy
# --------------------------------------------------------------------------- #
def _bare_receive_server(request_timeout=120.0):
    """A ReceiveServer with just the admission state, no sockets or buffer."""
    from tilert.pd_vllm import receive_server as rs

    srv = object.__new__(rs.ReceiveServer)
    srv._lock = threading.Lock()
    srv._current = None
    srv._cancelled = {}
    srv._generation = 0
    srv.request_timeout = request_timeout
    return srv


def _admit(srv, rid, rank=0, seq_len=8):
    return srv._admit({"seq_len": seq_len, "last_prompt_token": 5}, rid, rank)


def test_a_rank_arriving_after_its_request_was_released_is_refused():
    """The hole rid-scoping alone left open.

    release() with no writer yet was a plain return: nothing recorded that the
    rid was dead. The straggler then found a FREE buffer, was admitted, and held
    it while the next request's ranks were turned away "busy" until they
    exhausted their retries -- that request then waited out its whole
    kv_transfer_timeout.
    """
    srv = _bare_receive_server()
    srv.release("rid-dead")  # cancelled before any sender arrived
    assert srv._current is None, "precondition: nothing holds the slot"

    reply = _admit(srv, "rid-dead")
    assert reply["accepted"] is False, "the straggler opened a tenancy"
    # `cancelling`, which every connector already retries; a new reason would
    # be permanent to an older prefill and drop the shard during a rolling
    # upgrade. The distinction lives in `detail`.
    assert reply["error"] == "cancelling"
    assert reply["detail"] == "no_consumer"
    assert srv._current is None, "a refused straggler must claim nothing"


def test_a_released_rid_does_not_block_the_next_request():
    """The refusal has to be scoped to the dead rid, or it is a worse bug."""
    srv = _bare_receive_server()
    srv.release("rid-dead")

    reply = _admit(srv, "rid-next")
    assert reply["accepted"] is True, "a different rid was caught by the tombstone"
    assert srv._current.rid == "rid-next"


def test_a_re_announced_rid_is_admitted_again():
    """vLLM reuses the request id when it reschedules a preempted request.

    So a tombstone must not outlive the request coming back. /pd/decode calling
    expect() is the announcement, and it is the only thing that separates the
    retry from a straggler of the attempt before it.
    """
    srv = _bare_receive_server()
    srv.release("rid-1")
    assert _admit(srv, "rid-1")["accepted"] is False

    srv.expect("rid-1")  # what /pd/decode does on admission
    reply = _admit(srv, "rid-1")
    assert reply["accepted"] is True, "the retry lost its shard to a tombstone"
    assert srv._current.rid == "rid-1"


def test_a_tombstone_ages_out():
    """The backstop, in case no consumer ever re-announces the rid.

    Bounded by request_timeout because that is the senders' socket timeout: past
    it no rank can still be trying to join, so keeping the entry only leaks it.
    """
    srv = _bare_receive_server(request_timeout=0.05)
    srv.release("rid-1")
    assert _admit(srv, "rid-1")["accepted"] is False
    time.sleep(0.1)
    assert _admit(srv, "rid-1")["accepted"] is True


def test_a_straggler_is_refused_transiently_so_a_retry_can_still_land():
    """The tombstone refusal must be a reason senders already retry.

    A permanent one drops the shard of a rescheduled request whose /pd/decode
    has not arrived yet -- the same 120 s stall this fix is for, caused by the
    fix. A NEW reason is permanent to every connector built before it, so during
    a rolling upgrade a new decode node would do exactly that to an old prefill;
    the tombstone therefore reuses one the sender already knows.

    The reason set is read from the connector's source: it pulls in vLLM, and
    this file is CPU-only. Parsed, not pattern-matched.
    """
    import ast as _ast
    import pathlib as _p

    src = _p.Path(pd_router.__file__).with_name("prefill_connector.py").read_text()
    for node in _ast.walk(_ast.parse(src)):
        if isinstance(node, _ast.Assign) and any(
            getattr(t, "id", None) == "_TRANSIENT_REJECTS" for t in node.targets
        ):
            value = node.value
            if isinstance(value, _ast.Call):  # frozenset({...})
                value = value.args[0]
            reasons = _ast.literal_eval(value)
            break
    else:
        raise AssertionError("_TRANSIENT_REJECTS not found in the connector")

    srv = _bare_receive_server()
    srv.release("rid-1")
    assert (
        _admit(srv, "rid-1")["error"] in reasons
    ), "the tombstone refusal is not one the sender comes back from"


def test_pd_decode_announces_its_rid():
    """Otherwise a rescheduled request's senders meet the old tombstone."""
    server = _SlowServer()
    client = _decode_client(server)
    client.post(
        "/pd/decode", json={"rid": "rid-1", "first_token_id": 7, "max_tokens": 8, "timeout_s": 0.2}
    )
    assert getattr(server, "expected_rids", []) == ["rid-1"]


# --------------------------------------------------------------------------- #
# The two ways the retry budgets can disagree
# --------------------------------------------------------------------------- #
def test_the_send_never_runs_inside_the_forward_window():
    """Sending from the forward window cannot work with admission.

    The admission retries would all run before the prefill response returns,
    and the router cannot call /pd/decode -- the only thing that clears a
    tombstone for a rescheduled rid -- until it has. A sender there is
    therefore guaranteed to spend its whole budget against a tombstone it
    cannot outlast, and then drop the shard.

    So `wait_for_save` only ever queues, and the selectable synchronous path
    that used to bypass the queue is gone.
    """
    import ast as _ast
    import pathlib as _p

    src = _p.Path(pd_router.__file__).with_name("prefill_connector.py").read_text()
    assert "self._sync_send" not in src, (
        "the synchronous send path is back; its retries cannot outlast a "
        "tombstone, because /pd/decode comes after the prefill response"
    )

    tree = _ast.parse(src)
    fn = next(
        n for n in _ast.walk(tree) if isinstance(n, _ast.FunctionDef) and n.name == "wait_for_save"
    )
    called = {
        n.func.attr
        for n in _ast.walk(fn)
        if isinstance(n, _ast.Call) and isinstance(n.func, _ast.Attribute)
    }
    assert (
        "_send" not in called and "_send_with_retry" not in called
    ), f"wait_for_save sends inline: {sorted(called)}"


def test_wait_for_save_queues_a_complete_job():
    """Run it, do not read it.

    Reading the source could tell that `put` is called but not what it is
    called with -- and a `job` that is never built raises NameError on every
    single transfer, which no assertion about the source would have noticed.
    """
    import queue as _q
    import types as _t

    from tilert.pd_vllm import prefill_connector as pc

    meta = pc._ReqMeta(
        req_id="r",
        rid="rid-1",
        num_tokens=8,
        last_prompt_token=5,
        block_ids_per_group=[],
        tilert_host="127.0.0.1",
        tilert_ctrl_port=1,
    )

    conn = object.__new__(pc.TileRTConnector)
    conn._send_q = _q.Queue()
    conn._tp_rank = 0
    conn._reg = object()
    conn._staging = _t.SimpleNamespace(data_ptr=lambda: 0x1000)
    conn._max_seq = 4096
    conn._profile = _t.SimpleNamespace(
        sender_ranks=(0,), extract=lambda reg, m, rank, staging, max_seq: {"seq": 8, "x": 1}
    )
    conn._ensure_worker_ready = lambda: None
    conn._get_connector_metadata = lambda: _t.SimpleNamespace(requests=[meta])
    # The isinstance check in wait_for_save is against the real metadata type.
    md = pc.TileRTMetadata()
    md.requests = [meta]
    conn._get_connector_metadata = lambda: md

    conn.wait_for_save()

    job = conn._send_q.get_nowait()
    assert job["meta"] is meta
    assert job["seq"] == 8
    assert job["sections"] == {"seq": 8, "x": 1}


def test_a_tombstone_outlasts_the_senders_own_retry_budget():
    """request_timeout bounds ONE connection, not the sequence of them.

    Each retry opens a new socket, so a sender configured with enough attempts
    is still trying after a tombstone sized to one socket timeout has expired
    -- and is then admitted for a request nobody wants. The sender declares its
    budget; the receiver sizes the tombstone to outlast it.
    """
    srv = _bare_receive_server(request_timeout=0.05)
    srv.release("rid-1")
    # Without the declaration, this tombstone expires almost immediately.
    time.sleep(0.1)
    assert _admit(srv, "rid-1")["accepted"] is True, "precondition"

    srv = _bare_receive_server(request_timeout=0.05)
    srv.release("rid-2")
    reply = srv._admit(
        {"seq_len": 8, "last_prompt_token": 5, "admission_window_s": 30.0}, "rid-2", 0
    )
    assert reply["accepted"] is False
    time.sleep(0.1)  # past request_timeout, inside the window
    assert (
        srv._admit({"seq_len": 8, "last_prompt_token": 5, "admission_window_s": 30.0}, "rid-2", 1)[
            "accepted"
        ]
        is False
    ), "the tombstone expired while the sender was still retrying"


def test_a_second_rank_cannot_shorten_the_tombstone():
    """Never take the smaller of two declared budgets."""
    srv = _bare_receive_server(request_timeout=0.05)
    srv.release("rid-1")
    srv._admit({"seq_len": 8, "last_prompt_token": 5, "admission_window_s": 30.0}, "rid-1", 0)
    long_deadline = srv._cancelled["rid-1"]
    srv._admit({"seq_len": 8, "last_prompt_token": 5, "admission_window_s": 0.01}, "rid-1", 1)
    assert srv._cancelled["rid-1"] == long_deadline


def test_a_sender_that_declares_nothing_keeps_the_default():
    """An older connector sends no budget; the default must still apply."""
    srv = _bare_receive_server(request_timeout=60.0)
    srv.release("rid-1")
    before = srv._cancelled["rid-1"]
    srv._admit({"seq_len": 8, "last_prompt_token": 5}, "rid-1", 0)
    assert srv._cancelled["rid-1"] == before
