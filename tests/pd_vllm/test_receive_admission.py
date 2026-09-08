"""No RDMA write without an admission, and no buffer reuse under a live writer.

The decode node's receive buffer holds one request at a time. The sender used to
send its request metadata and then write immediately, never reading the reply --
so a receiver that answered "busy" was overruled, and this request's KV landed
inside the request the node was already serving. Nothing downstream detects that:
the victim decodes from a mix of two prompts' state and answers confidently.

Two invariants are pinned here, and they are separate:

1. **Admission gates the write.** The sender performs zero ``transport.write``
   calls unless it received an accept whose rid, rank and generation match what
   it asked for.
2. **A tenancy is not replaced while its writers are live.** A request that timed
   out or was abandoned holds the slot until every admitted rank has stopped, so
   the buffer is never handed over underneath an in-flight RDMA.

The receive server is exercised over a real TCP socket pair with a fake transport
and a fake profile, so the framing and the multi-rank ordering are real. No GPU,
no mooncake, no vLLM.

Run:
  CUDA_VISIBLE_DEVICES= python -m pytest \
      tests/pd_vllm/test_receive_admission.py -v
"""

import socket
import threading
import time
import types

import pytest

from tilert.pd_vllm import receive_server as rs
from tilert.pd_vllm import wire

SENDER_RANKS = frozenset({0, 1})
MAX_SEQ = 128


# --------------------------------------------------------------------------- #
# A ReceiveServer with the GPU/RDMA parts replaced, but the real control plane
# --------------------------------------------------------------------------- #
class _FakeTransport:
    name = "mooncake"

    def init(self, hostname):
        pass

    def register(self, ptr, size, dev):
        pass

    def local_meta(self):
        return {"mooncake_session_id": "fake-session"}


@pytest.fixture
def server(monkeypatch):
    """Real ReceiveServer: real socket, real framing, no torch and no RDMA."""
    monkeypatch.setattr(rs, "make_transport", lambda name: _FakeTransport(), raising=False)
    monkeypatch.setattr("tilert.pd_vllm.transport.make_transport", lambda name: _FakeTransport())
    monkeypatch.setattr(
        rs.torch, "zeros", lambda *a, **k: types.SimpleNamespace(data_ptr=lambda: 0x1000)
    )
    monkeypatch.setattr(rs.wire, "local_ip", lambda *a, **k: "127.0.0.1")

    profile = types.SimpleNamespace(
        name="fake",
        layout_version=7,
        sender_ranks=SENDER_RANKS,
        buffer_bytes=lambda max_seq_len: 4096,
        hello_layout=lambda base_ptr, max_seq_len: {"kv_base": base_ptr},
    )
    srv = rs.ReceiveServer(
        profile,
        max_seq_len=MAX_SEQ,
        ctrl_port=0,
        hostname="127.0.0.1",
        device="cpu",
        request_timeout=5.0,
    )
    yield srv
    srv.close()


class _Rank:
    """One prefill rank's side of the control channel."""

    def __init__(self, server: rs.ReceiveServer, rank: int):
        self.rank = rank
        self.sock = socket.create_connection(("127.0.0.1", server._srv.getsockname()[1]), timeout=5)
        self.hello = wire.recv_msg(self.sock)

    def request(self, rid="rid-1", seq_len=8, **extra):
        wire.send_msg(
            self.sock,
            {"rid": rid, "rank": self.rank, "seq_len": seq_len, "last_prompt_token": 5, **extra},
        )
        return wire.recv_msg(self.sock)

    def done(self, rid, generation):
        wire.send_msg(self.sock, wire.done_msg(rid, self.rank, generation))

    def close(self):
        self.sock.close()


def _settle():
    """Let the server's per-connection threads reach their next lock step."""
    for _ in range(200):
        if not any(t.name.startswith("Thread-") and t.is_alive() for t in threading.enumerate()):
            break
        threading.Event().wait(0.005)
    threading.Event().wait(0.05)


# --------------------------------------------------------------------------- #
# The hello advertises the control-plane version
# --------------------------------------------------------------------------- #
def test_hello_carries_the_protocol_version(server):
    r = _Rank(server, 0)
    assert r.hello["protocol_version"] == wire.PROTOCOL_VERSION
    assert wire.PROTOCOL_VERSION >= 2, "admission arrived in v2"
    r.close()


def test_the_protocol_version_is_separate_from_the_layout_version(server):
    """They version different things: one the control flow, one the buffer geometry.

    Folding them together would make every layout bump look like a control-plane bump to the
    sender's assertion.
    """
    r = _Rank(server, 0)
    assert r.hello["layout_version"] == 7
    assert r.hello["protocol_version"] != 7
    r.close()


# --------------------------------------------------------------------------- #
# Admission: accept carries rid/rank/generation
# --------------------------------------------------------------------------- #
def test_a_first_rank_is_admitted_with_a_generation(server):
    r = _Rank(server, 0)
    ack = r.request()
    assert ack["accepted"] is True
    assert ack["rid"] == "rid-1"
    assert ack["rank"] == 0
    assert isinstance(ack["generation"], int)
    r.close()


def test_same_rid_ranks_are_admitted_in_arbitrary_order(server):
    """Ranks connect in no particular order and only one of them creates the
    tenancy, so every other rank must still be admitted to the same generation.
    """
    r1 = _Rank(server, 1)
    a1 = r1.request()
    r0 = _Rank(server, 0)
    a0 = r0.request()
    assert a1["accepted"] and a0["accepted"]
    assert a1["generation"] == a0["generation"]
    r0.close()
    r1.close()


def test_a_duplicate_rank_is_refused(server):
    """Counting one rank twice could complete the request before every shard has
    actually landed.
    """
    r0 = _Rank(server, 0)
    ack = r0.request()
    r0.done("rid-1", ack["generation"])
    _settle()
    dup = _Rank(server, 0)
    assert dup.request()["accepted"] is False
    r0.close()
    dup.close()


def test_a_second_rid_is_refused_while_the_first_is_transferring(server):
    """The reply the sender used to ignore."""
    held = _Rank(server, 0)
    held.request(rid="rid-1")
    other = _Rank(server, 0)
    ack = other.request(rid="rid-2")
    assert ack["accepted"] is False
    assert ack["error"] == "busy"
    assert ack["busy_rid"] == "rid-1"
    held.close()
    other.close()


def test_an_oversized_seq_len_is_refused(server):
    r = _Rank(server, 0)
    ack = r.request(seq_len=MAX_SEQ + 1)
    assert ack["accepted"] is False
    assert "seq_len" in ack["error"]
    r.close()


def test_an_oversized_request_claims_no_tenancy(server):
    """A refusal before admission must leave the slot free for the next request."""
    r = _Rank(server, 0)
    r.request(seq_len=MAX_SEQ + 1)
    r.close()
    _settle()
    assert server.state_snapshot()["state"] == rs.FREE


# --------------------------------------------------------------------------- #
# Completion and generations
# --------------------------------------------------------------------------- #
def test_the_request_completes_when_every_sender_rank_is_done(server):
    ranks = [_Rank(server, i) for i in sorted(SENDER_RANKS)]
    acks = [r.request() for r in ranks]
    gen = acks[0]["generation"]
    for r in ranks:
        r.done("rid-1", gen)
    got = server.completed.get(timeout=5)
    assert got.rid == "rid-1"
    assert got.done_ranks == set(SENDER_RANKS)
    assert got.state == rs.COMPLETE
    for r in ranks:
        r.close()


def test_a_partially_done_request_does_not_complete(server):
    r0 = _Rank(server, 0)
    ack = r0.request()
    r0.done("rid-1", ack["generation"])
    _settle()
    assert server.completed.empty()
    assert server.state_snapshot()["state"] == rs.TRANSFERRING
    r0.close()


def test_a_rejected_straggler_cannot_disturb_the_current_tenancy(server):
    """A rank turned away at admission must not affect whoever holds the slot,
    even if it goes on to send a done anyway.
    """
    ranks = [_Rank(server, i) for i in sorted(SENDER_RANKS)]
    acks = [r.request() for r in ranks]
    gen = acks[0]["generation"]
    for r in ranks:
        r.done("rid-1", gen)
    first = server.completed.get(timeout=5)
    server.release("rid-1")
    _settle()

    # A second tenancy takes the slot ...
    nxt = [_Rank(server, i) for i in sorted(SENDER_RANKS)]
    new_acks = [r.request(rid="rid-2") for r in nxt]
    assert new_acks[0]["generation"] != first.generation

    # ... and a straggler from the first one is refused and then ignored.
    straggler = _Rank(server, 0)
    assert straggler.request(rid="rid-1")["accepted"] is False
    wire.send_msg(straggler.sock, wire.done_msg("rid-1", 0, first.generation))
    _settle()
    snap = server.state_snapshot()
    assert snap["rid"] == "rid-2"
    assert snap["state"] == rs.TRANSFERRING
    for r in ranks + nxt + [straggler]:
        r.close()


def test_a_done_from_a_superseded_generation_of_the_same_rid_is_ignored(server):
    """The rid alone is not enough to identify a tenancy.

    ``rid`` is derived from the vLLM request id, so a retry of the same request
    carries the SAME rid against a NEW receive-buffer tenancy. A done left over
    from the previous attempt therefore matches on rid and must be rejected on
    generation -- otherwise it credits a rank that has written nothing for the
    current tenancy, and the request can be handed to the engine before that
    rank's shard has actually landed.

    Driven through the real socket path: the extra rank-0 connection is admitted
    to generation 1 (nothing has marked rank 0 done yet) and deliberately outlives
    it, which is how a stale done reaches a live tenancy.
    """
    a = _Rank(server, 0)
    gen1 = a.request()["generation"]
    stale = _Rank(server, 0)  # second rank-0 channel
    assert stale.request()["generation"] == gen1
    c = _Rank(server, 1)
    c.request()

    a.done("rid-1", gen1)
    c.done("rid-1", gen1)
    assert server.completed.get(timeout=5).done_ranks == set(SENDER_RANKS)
    a.close()
    c.close()
    server.release("rid-1")
    _settle()

    # The retry announces itself, as /pd/decode does in production: release()
    # leaves a tombstone so a straggler from the finished attempt cannot open a
    # tenancy nobody is waiting for, and only a consumer saying "I want this
    # rid" distinguishes the retry from that straggler.
    server.expect("rid-1")

    # Same rid, new tenancy: rank 0 is admitted but has NOT written yet, while
    # rank 1 has finished. The tenancy is one rank short of complete.
    d = _Rank(server, 0)
    gen2 = d.request()["generation"]
    assert gen2 != gen1
    e = _Rank(server, 1)
    e.request()
    e.done("rid-1", gen2)
    _settle()
    assert server.completed.empty(), "precondition: gen2 is not complete yet"

    # The leftover channel reports done for the OLD generation. Counted, it
    # supplies the missing rank and the request is handed to the engine while
    # rank 0's shard for THIS tenancy has never been written.
    stale.done("rid-1", gen1)
    _settle()

    assert server.completed.empty(), (
        "a stale done completed the tenancy: the engine would decode from a "
        "buffer whose rank-0 shard was never written for this request"
    )
    snap = server.state_snapshot()
    assert snap["generation"] == gen2
    assert snap["state"] == rs.TRANSFERRING
    for r in (stale, d, e):
        r.close()


def test_generations_are_never_reused(server):
    seen = set()
    for i in range(3):
        r = _Rank(server, 0)
        ack = r.request(rid=f"rid-{i}")
        assert ack["accepted"], ack
        seen.add(ack["generation"])
        r.close()
        _settle()
        server.release(f"rid-{i}")  # scoped to the rid this round created
        _settle()
    assert len(seen) == 3


# --------------------------------------------------------------------------- #
# The tenancy is not replaced under a live writer
# --------------------------------------------------------------------------- #
def test_release_under_a_live_writer_cancels_instead_of_freeing(server):
    """The abandon path (rejected request, drained stale entry) must not hand the
    buffer over while a rank is still writing into it.
    """
    r0 = _Rank(server, 0)
    r0.request()
    _settle()
    server.release("rid-1")
    snap = server.state_snapshot()
    assert snap["state"] == rs.CANCELLING
    assert snap["rid"] == "rid-1"
    assert snap["active_writers"] == 1
    r0.close()


def test_a_cancelling_tenancy_still_refuses_a_new_rid(server):
    r0 = _Rank(server, 0)
    r0.request()
    _settle()
    server.release("rid-1")
    other = _Rank(server, 0)
    assert other.request(rid="rid-2")["accepted"] is False
    r0.close()
    other.close()


def test_the_slot_frees_once_the_writer_goes_away(server):
    """Every sender connection carries request_timeout as its socket timeout, so
    the drain is bounded -- a disconnect just gets there sooner.
    """
    r0 = _Rank(server, 0)
    r0.request()
    _settle()
    server.release("rid-1")
    assert server.state_snapshot()["state"] == rs.CANCELLING
    r0.close()
    _settle()
    assert server.state_snapshot()["state"] == rs.FREE


def test_a_new_rid_is_admitted_after_the_drain_completes(server):
    r0 = _Rank(server, 0)
    r0.request()
    _settle()
    server.release("rid-1")
    r0.close()
    _settle()
    nxt = _Rank(server, 0)
    assert nxt.request(rid="rid-2")["accepted"] is True
    nxt.close()


def test_a_dead_writer_releases_its_claim(server):
    """A sender that dies mid-RDMA must not pin the slot forever."""
    r0 = _Rank(server, 0)
    r0.request()
    _settle()
    assert server.state_snapshot()["active_writers"] == 1
    r0.close()
    _settle()
    assert server.state_snapshot()["active_writers"] == 0


# --------------------------------------------------------------------------- #
# Timeout: bounded, but never at the cost of the live-writer rule
# --------------------------------------------------------------------------- #
def test_a_tenancy_whose_senders_all_died_ages_out(server):
    """Otherwise the node is out of service for good.

    Every rank is admitted and then dies without reporting done, so the tenancy
    stays TRANSFERRING with no writers. Nothing will call ``release()`` for it --
    the router gave up on the prefill leg, so ``/pd/decode`` never arrives for
    that rid. The age-out is the only way back.
    """
    server.request_timeout = 0.05
    r0 = _Rank(server, 0)
    r0.request()
    r0.close()
    _settle()
    snap = server.state_snapshot()
    assert snap["state"] == rs.TRANSFERRING and snap["active_writers"] == 0

    threading.Event().wait(0.1)
    nxt = _Rank(server, 0)
    assert nxt.request(rid="rid-2")["accepted"] is True
    nxt.close()


def _aged(state, writers, server):
    """A tenancy whose first connection is far older than request_timeout."""
    return rs.ReceivedRequest(
        rid="rid-1",
        seq_len=8,
        last_prompt_token=5,
        first_token_id=None,
        sampling=None,
        state=state,
        active_writers=writers,
        t_first_conn=time.time() - server.request_timeout - 60,
    )


@pytest.mark.parametrize("state", [rs.RESERVED, rs.TRANSFERRING])
def test_the_age_out_never_overrides_a_live_writer(server, state):
    """The original code aged out on wall clock ALONE, which is exactly how a
    request got replaced while its ranks were still writing. No amount of
    elapsed time may make that reusable.

    Tested on ``_reusable`` directly rather than over sockets: ``request_timeout``
    is also each sender connection's socket timeout, so shrinking it to force an
    age-out kills the very live writer the case is about.
    """
    assert server._reusable(_aged(state, writers=1, server=server)) is False


@pytest.mark.parametrize("state", [rs.RESERVED, rs.TRANSFERRING])
def test_an_aged_out_tenancy_with_no_writer_is_reusable(server, state):
    assert server._reusable(_aged(state, writers=0, server=server)) is True


@pytest.mark.parametrize(
    "state,writers,expected",
    [
        (rs.COMPLETE, 0, True),  # finished; the decode server owns it now
        (rs.COMPLETE, 1, True),  # all ranks reported done, sockets still closing
        (rs.CANCELLING, 0, True),  # abandoned and drained
        (rs.CANCELLING, 1, False),  # abandoned, still draining
    ],
)
def test_the_reuse_rule_by_state(server, state, writers, expected):
    cur = rs.ReceivedRequest(
        rid="rid-1",
        seq_len=8,
        last_prompt_token=5,
        first_token_id=None,
        sampling=None,
        state=state,
        active_writers=writers,
        t_first_conn=time.time(),
    )
    assert server._reusable(cur) is expected


def test_a_fresh_tenancy_is_not_aged_out_immediately(server):
    """The age-out must not turn into "first come, first served, briefly"."""
    r0 = _Rank(server, 0)
    r0.request(rid="rid-1")
    r0.close()
    _settle()
    # request_timeout is 5.0s in the fixture, so this is still young.
    other = _Rank(server, 0)
    assert other.request(rid="rid-2")["accepted"] is False
    other.close()


# --------------------------------------------------------------------------- #
# The sender side: zero RDMA writes on any rejection
# --------------------------------------------------------------------------- #
class _RecordingTransport:
    name = "mooncake"

    def __init__(self):
        self.writes = []

    def write(self, hello, srcs, dsts, lens):
        self.writes.append((srcs, dsts, lens))


def _import_connector(monkeypatch):
    """Import ``prefill_connector`` without a real vLLM.

    It subclasses vLLM's connector base at module level, so a serve-only
    environment cannot import it -- and ``importorskip`` would SKIP the most
    important tests in this file (zero RDMA writes on a rejection) exactly where
    they run: CI has no vLLM. So the base classes are stubbed instead, the way
    the engine-free tests in this directory stub ``tilert``.

    The stubs are installed through ``monkeypatch`` rather than at import time so
    the fake ``vllm`` entry disappears at teardown: left in ``sys.modules`` it
    would make ``pytest.importorskip("vllm")`` succeed elsewhere in the session
    (test_oai_parser.py) and those tests would fail on a module that has no
    ``vllm.parser``.
    """
    import dataclasses
    import sys

    base_path = "vllm.distributed.kv_transfer.kv_connector.v1.base"
    try:  # a real vLLM (the router's own environment) is used as-is
        __import__(base_path)
    except Exception:
        for name in (
            "vllm",
            "vllm.distributed",
            "vllm.distributed.kv_transfer",
            "vllm.distributed.kv_transfer.kv_connector",
            "vllm.distributed.kv_transfer.kv_connector.v1",
        ):
            if name not in sys.modules:
                monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
        base = types.ModuleType(base_path)

        class KVConnectorBase_V1:  # noqa: N801  (mirrors vLLM's own name)
            def __init__(self, vllm_config=None, role=None, kv_cache_config=None):
                pass

        @dataclasses.dataclass
        class KVConnectorMetadata:
            pass

        class SupportsHMA:
            pass

        base.KVConnectorBase_V1 = KVConnectorBase_V1
        base.KVConnectorMetadata = KVConnectorMetadata
        base.SupportsHMA = SupportsHMA
        monkeypatch.setitem(sys.modules, base_path, base)

    from tilert.pd_vllm import prefill_connector as pc

    return pc


def _sender(monkeypatch, replies, *, protocol_version=wire.PROTOCOL_VERSION):
    """Drive ``TileRTConnector._send`` against a scripted receiver.

    Returns (transport, sent_messages). ``replies`` are the messages the fake
    receiver returns after the hello, in order.
    """
    pc = _import_connector(monkeypatch)

    hello = {
        "magic": wire.MAGIC,
        "protocol_version": protocol_version,
        "layout_version": 7,
        "transport": "mooncake",
        "max_seq_len": MAX_SEQ,
        "busy": False,
        "kv_base": 0x1000,
    }
    inbox = [hello, *replies]
    sent = []

    monkeypatch.setattr(pc.wire, "recv_msg", lambda conn: inbox.pop(0))
    monkeypatch.setattr(pc.wire, "send_msg", lambda conn, obj: sent.append(obj))

    class _Sock:
        def __init__(self, *a, **k):
            pass

        def setsockopt(self, *a):
            pass

        def settimeout(self, *a):
            pass

        def connect(self, *a):
            pass

        def close(self):
            pass

    monkeypatch.setattr("socket.socket", _Sock)

    transport = _RecordingTransport()
    conn_obj = object.__new__(pc.TileRTConnector)
    conn_obj._transport = transport
    conn_obj._tp_rank = 0
    # _send declares this sender's remaining retry budget so the receiver can
    # size a tombstone to outlast it. The retry test overrides it below.
    conn_obj._admission_attempts = pc._ADMISSION_ATTEMPTS
    conn_obj._staging = types.SimpleNamespace(data_ptr=lambda: 0x2000)
    conn_obj._profile = types.SimpleNamespace(
        layout_version=7, rdma_plan=lambda hello, sections, rank, seq, base: ([1], [2], [3])
    )

    meta = pc._ReqMeta(
        req_id="r",
        rid="rid-1",
        num_tokens=8,
        last_prompt_token=5,
        block_ids_per_group=[],
        tilert_host="127.0.0.1",
        tilert_ctrl_port=1,
    )
    conn_obj._send({"meta": meta, "sections": {"seq": 8}, "seq": 8})
    return transport, sent


def test_the_sender_writes_after_an_accept(monkeypatch):
    transport, sent = _sender(monkeypatch, [wire.accept_msg("rid-1", 0, 12)])
    assert len(transport.writes) == 1
    # ... and the done echoes the generation it was admitted under.
    assert sent[-1] == {"done": True, "rid": "rid-1", "rank": 0, "generation": 12}


def test_the_sender_writes_nothing_on_a_busy_rejection(monkeypatch):
    """The original bug: this reply existed and was never read."""
    transport, sent = _sender(monkeypatch, [wire.reject_msg("busy", busy_rid="other")])
    assert transport.writes == []
    assert not any("done" in m for m in sent)


def test_the_sender_writes_nothing_on_a_seq_len_rejection(monkeypatch):
    transport, _ = _sender(monkeypatch, [wire.reject_msg("seq_len exceeds max_seq_len")])
    assert transport.writes == []


@pytest.mark.parametrize(
    "ack",
    [
        {},  # empty
        {"accepted": False},  # bare refusal
        {"rid": "rid-1", "rank": 0, "generation": 1},  # no accepted flag
        {"accepted": True, "rid": "other", "rank": 0, "generation": 1},
        {"accepted": True, "rid": "rid-1", "rank": 3, "generation": 1},
        {"accepted": True, "rid": "rid-1", "rank": 0},  # no generation
        {"accepted": True, "rid": "rid-1", "rank": 0, "generation": "x"},
    ],
)
def test_only_a_fully_matching_admission_permits_a_write(monkeypatch, ack):
    """Checked field by field, not just for an ``error`` key: an admission for a
    different rid or rank is an admission for a different tenancy, and writing on
    it corrupts exactly the same way a busy rejection would.
    """
    transport, _ = _sender(monkeypatch, [ack])
    assert transport.writes == []


def test_a_protocol_version_mismatch_fails_before_writing(monkeypatch):
    """A v1 receiver never sends an accept, so waiting for one would hang every
    request; failing loud names the actual problem.
    """
    with pytest.raises(AssertionError, match="protocol mismatch"):
        _sender(monkeypatch, [wire.accept_msg("rid-1", 0, 1)], protocol_version=1)


def test_a_layout_version_mismatch_still_fails_before_writing(monkeypatch):
    pc = _import_connector(monkeypatch)

    hello = {
        "magic": wire.MAGIC,
        "protocol_version": wire.PROTOCOL_VERSION,
        "layout_version": 99,
        "transport": "mooncake",
        "max_seq_len": MAX_SEQ,
        "busy": False,
    }
    monkeypatch.setattr(pc.wire, "recv_msg", lambda conn: hello)
    monkeypatch.setattr(pc.wire, "send_msg", lambda conn, obj: None)

    class _Sock:
        def __init__(self, *a, **k):
            pass

        def setsockopt(self, *a):
            pass

        def settimeout(self, *a):
            pass

        def connect(self, *a):
            pass

        def close(self):
            pass

    monkeypatch.setattr("socket.socket", _Sock)
    transport = _RecordingTransport()
    conn_obj = object.__new__(pc.TileRTConnector)
    conn_obj._transport = transport
    conn_obj._tp_rank = 0
    # _send declares this sender's remaining retry budget so the receiver can
    # size a tombstone to outlast it. The retry test overrides it below.
    conn_obj._admission_attempts = pc._ADMISSION_ATTEMPTS
    conn_obj._staging = types.SimpleNamespace(data_ptr=lambda: 0x2000)
    conn_obj._profile = types.SimpleNamespace(layout_version=7)
    meta = pc._ReqMeta(
        req_id="r",
        rid="rid-1",
        num_tokens=8,
        last_prompt_token=5,
        block_ids_per_group=[],
        tilert_host="127.0.0.1",
        tilert_ctrl_port=1,
    )
    with pytest.raises(AssertionError, match="layout version"):
        conn_obj._send({"meta": meta, "sections": {"seq": 8}, "seq": 8})
    assert transport.writes == []


# --------------------------------------------------------------------------- #
# A rejected admission must not silently drop the shard (codex, PR #40)
# --------------------------------------------------------------------------- #
def _retrying_sender(monkeypatch, reply_sequence, attempts=3):
    """Drive ``_send_with_retry`` against a receiver that answers in sequence.

    Returns (transport, attempt_count). Sleeping is stubbed out so the backoff
    does not slow the suite.
    """
    pc = _import_connector(monkeypatch)

    hello = {
        "magic": wire.MAGIC,
        "protocol_version": wire.PROTOCOL_VERSION,
        "layout_version": 7,
        "transport": "mooncake",
        "max_seq_len": MAX_SEQ,
        "busy": False,
        "kv_base": 0x1000,
    }
    replies = list(reply_sequence)
    calls = {"n": 0}

    def recv(conn):
        # Each attempt reopens the channel: hello, then that attempt's verdict.
        if calls["hello_pending"]:
            calls["hello_pending"] = False
            return hello
        calls["hello_pending"] = True
        return replies.pop(0)

    calls["hello_pending"] = True
    monkeypatch.setattr(pc.wire, "recv_msg", recv)
    monkeypatch.setattr(pc.wire, "send_msg", lambda conn, obj: None)
    monkeypatch.setattr(
        pc._time if hasattr(pc, "_time") else pc, "sleep", lambda s: None, raising=False
    )

    class _Sock:
        def __init__(self, *a, **k):
            calls["n"] += 1

        def setsockopt(self, *a):
            pass

        def settimeout(self, *a):
            pass

        def connect(self, *a):
            pass

        def close(self):
            pass

    monkeypatch.setattr("socket.socket", _Sock)
    monkeypatch.setattr("time.sleep", lambda s: None)

    transport = _RecordingTransport()
    conn_obj = object.__new__(pc.TileRTConnector)
    conn_obj._transport = transport
    conn_obj._tp_rank = 0
    # _send declares this sender's remaining retry budget so the receiver can
    # size a tombstone to outlast it. The retry test overrides it below.
    conn_obj._admission_attempts = pc._ADMISSION_ATTEMPTS
    conn_obj._staging = types.SimpleNamespace(data_ptr=lambda: 0x2000)
    conn_obj._admission_attempts = attempts
    conn_obj._profile = types.SimpleNamespace(
        layout_version=7, rdma_plan=lambda hello, sections, rank, seq, base: ([1], [2], [3])
    )
    meta = pc._ReqMeta(
        req_id="r",
        rid="rid-1",
        num_tokens=8,
        last_prompt_token=5,
        block_ids_per_group=[],
        tilert_host="127.0.0.1",
        tilert_ctrl_port=1,
    )
    conn_obj._send_with_retry({"meta": meta, "sections": {"seq": 8}, "seq": 8})
    return transport, calls["n"]


def test_a_busy_slot_is_retried_and_then_written(monkeypatch):
    """A rank turned away while a previous transfer drains would otherwise drop
    its shard for good, and nothing tells the router: the prefill response still
    succeeds and /pd/decode waits out its whole kv_transfer_timeout.
    """
    transport, attempts = _retrying_sender(
        monkeypatch,
        [
            wire.reject_msg("busy", busy_rid="other"),
            wire.accept_msg("rid-1", 0, 9),
        ],
    )
    assert len(transport.writes) == 1
    assert attempts == 2


def test_a_draining_slot_is_retried(monkeypatch):
    transport, attempts = _retrying_sender(
        monkeypatch,
        [
            wire.reject_msg("cancelling", rid="rid-1"),
            wire.accept_msg("rid-1", 0, 9),
        ],
    )
    assert len(transport.writes) == 1


def test_retries_are_bounded_and_never_write(monkeypatch):
    """Still zero writes when every attempt is refused -- the no-write rule is
    not traded away for promptness.
    """
    transport, attempts = _retrying_sender(monkeypatch, [wire.reject_msg("busy")] * 3, attempts=3)
    assert transport.writes == []
    assert attempts == 3


@pytest.mark.parametrize(
    "reject",
    [
        wire.reject_msg("seq_len exceeds max_seq_len"),
        wire.reject_msg("duplicate_rank", rank=0),
    ],
)
def test_a_permanent_rejection_is_not_retried(monkeypatch, reject):
    """These are about THIS request and would be refused again; retrying only
    delays the failure.
    """
    transport, attempts = _retrying_sender(monkeypatch, [reject], attempts=3)
    assert transport.writes == []
    assert attempts == 1


def test_the_request_declares_the_senders_admission_budget(monkeypatch):
    """The receiver sizes a tombstone from this, and cannot derive it itself.

    Its own `request_timeout` bounds one connection; each retry opens a new
    one, so only the sender knows how long this rid may keep coming back.
    """
    _, sent = _sender(monkeypatch, [wire.accept_msg("rid-1", 0, 12)])
    req = sent[0]
    assert "admission_window_s" in req, req
    # The default 5 attempts back off 0.2 + 0.4 + 0.8 + 1.6.
    assert abs(req["admission_window_s"] - 3.0) < 1e-9, req
