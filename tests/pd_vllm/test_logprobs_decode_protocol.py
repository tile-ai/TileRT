"""`/pd/decode` carries per-token logprobs, in both response branches.

Drives the real ``decode_server`` app over HTTP with ``StubEngine``, so the
protocol is pinned without a GPU: the request field, the non-streaming
``logprobs`` object, the per-line ``lp``/``tp`` fields on the NDJSON stream, and
the refusal to answer at all when the engine cannot produce logprobs.

The alignment property that matters: ``lp[i]`` and ``tp[i]`` describe ``t[i]``.
The engine writes its entry before putting the token on the queue, so a token the
generator has dequeued already has its logprob visible; the stream batches
tokens, so this is asserted across batch boundaries rather than assumed.

Run:
  CUDA_VISIBLE_DEVICES= python -m pytest \
      tests/pd_vllm/test_logprobs_decode_protocol.py -v
"""

from __future__ import annotations

import json
import queue
import time
import types

import pytest
from fastapi.testclient import TestClient

from tilert.pd_vllm.decode_server import build_app
from tilert.pd_vllm.engine_iface import StubEngine

RID = "cmpl-lp"
FIXED = (11, 22, 33)
FIRST = 7


class _FakeReq:
    def __init__(self, rid):
        self.rid = rid
        self.seq_len = 8
        self.last_prompt_token = 5


class _FakeServer:
    """Minimal ReceiveServer stand-in (same shape as test_grammar_plumbing)."""

    def __init__(self, rid):
        self.completed: queue.Queue = queue.Queue()
        self.completed.put(_FakeReq(rid))
        self.profile = types.SimpleNamespace(convert=lambda *a, **k: "converted", num_ranks=8)
        self.buffer = None
        self.base_ptr = 0
        self.max_seq_len = 4096

    def expect(self, rid=None):
        # /pd/decode announces its rid so the real ReceiveServer can drop a
        # tombstone left by a previous attempt at the same request. Recorded,
        # so a test can assert the announcement happened.
        self.expected_rids = getattr(self, "expected_rids", [])
        self.expected_rids.append(rid)

    def release(self, rid=None):
        pass


def _client(engine=None, rid=RID):
    return TestClient(build_app(_FakeServer(rid), engine or StubEngine(FIXED)))


def _body(**kw):
    return {"rid": RID, "first_token_id": FIRST, "max_tokens": 8, **kw}


class _NoLogprobsEngine(StubEngine):
    """An engine predating the capability."""

    def supports_logprobs(self) -> bool:
        return False


class _SlowEngine(StubEngine):
    """Emits with a gap so the stream generator drains one token per batch.

    Needed to exercise the batch offset: when every token arrives in a single
    batch, a wrong offset is indistinguishable from a right one.
    """

    def decode(
        self,
        first_token_id,
        max_tokens,
        sampling,
        on_token=None,
        cancel_event=None,
        grammar_session=None,
        top_logprobs=None,
    ):
        out = ([int(first_token_id)] + list(self._fixed))[:max_tokens]
        for t in out:
            if on_token:
                if top_logprobs is None:
                    on_token(t)
                else:
                    on_token(
                        t,
                        self.fake_logprob(t),
                        [(t + k, self.fake_logprob(t) - 0.5 * k) for k in range(top_logprobs)],
                    )
            time.sleep(0.03)  # > the generator's 5 ms idle poll
        self.last_stats = {"finish_reason": "stop"}
        return out


class _ShortLogprobsEngine(StubEngine):
    """Declares support but emits a bare token -- an engine bug, not a client one.

    Omits at position 1, not 0. Position 0 is ``first_token_id``, which the
    prefill instance sampled, so a bare emit there is the contract (the router
    fills that entry in from the prefill response) rather than a fault.
    """

    def decode(
        self,
        first_token_id,
        max_tokens,
        sampling,
        on_token=None,
        cancel_event=None,
        grammar_session=None,
        top_logprobs=None,
    ):
        out = ([int(first_token_id)] + list(self._fixed))[:max_tokens]
        for i, t in enumerate(out):
            if on_token:
                if top_logprobs is None or i <= 1:
                    on_token(t)
                else:
                    on_token(t, self.fake_logprob(t), [])
        self.last_stats = {"finish_reason": "stop"}
        return out


class _EchoFirstEngine(StubEngine):
    """The real engines' shape: token 0 bare, every later token with a logprob.

    A real adapter behaves this way because it cannot report a distribution
    for ``first_token_id`` -- it echoed that token rather than sampling it.
    """

    def decode(
        self,
        first_token_id,
        max_tokens,
        sampling,
        on_token=None,
        cancel_event=None,
        grammar_session=None,
        top_logprobs=None,
    ):
        out = ([int(first_token_id)] + list(self._fixed))[:max_tokens]
        for i, t in enumerate(out):
            if on_token:
                if top_logprobs is None or i == 0:
                    on_token(t)
                else:
                    on_token(t, self.fake_logprob(t), [(t, self.fake_logprob(t))])
        self.last_stats = {"finish_reason": "stop"}
        return out


# --------------------------------------------------------------------------- #
# not requested -> byte-for-byte the old response
# --------------------------------------------------------------------------- #


def test_absent_field_leaves_the_response_unchanged() -> None:
    r = _client().post("/pd/decode", json=_body())
    assert r.status_code == 200
    assert "logprobs" not in r.json()


def test_absent_field_leaves_the_stream_unchanged() -> None:
    with _client().stream("POST", "/pd/decode", json=_body(stream=True)) as resp:
        lines = [json.loads(x) for x in resp.iter_lines() if x]
    tok_lines = [x for x in lines if "t" in x]
    assert tok_lines and all("lp" not in x and "tp" not in x for x in tok_lines)


# --------------------------------------------------------------------------- #
# non-streaming
# --------------------------------------------------------------------------- #


def test_non_streaming_returns_one_logprob_per_token() -> None:
    r = _client().post("/pd/decode", json=_body(top_logprobs=0))
    body = r.json()
    ids = body["token_ids"]
    assert body["logprobs"]["lp"] == [StubEngine.fake_logprob(t) for t in ids]


def test_non_streaming_candidate_count_matches_the_request() -> None:
    r = _client().post("/pd/decode", json=_body(top_logprobs=3))
    tp = r.json()["logprobs"]["tp"]
    assert all(len(row) == 3 for row in tp)


def test_top_logprobs_zero_gives_empty_candidate_rows() -> None:
    r = _client().post("/pd/decode", json=_body(top_logprobs=0))
    assert all(row == [] for row in r.json()["logprobs"]["tp"])


def test_candidates_are_id_logprob_pairs_chosen_token_first() -> None:
    r = _client().post("/pd/decode", json=_body(top_logprobs=2))
    body = r.json()
    first_id = body["token_ids"][0]
    row = body["logprobs"]["tp"][0]
    assert row[0] == [first_id, StubEngine.fake_logprob(first_id)]
    assert row[0][1] > row[1][1], "candidates must be descending"


# --------------------------------------------------------------------------- #
# streaming: lp[i] / tp[i] line up with t[i] across batch boundaries
# --------------------------------------------------------------------------- #


@pytest.fixture
def streamed():
    with _client().stream("POST", "/pd/decode", json=_body(stream=True, top_logprobs=2)) as resp:
        return [json.loads(x) for x in resp.iter_lines() if x]


def test_every_token_line_carries_aligned_logprobs(streamed) -> None:
    for line in (x for x in streamed if "t" in x):
        assert len(line["lp"]) == len(line["t"])
        assert len(line["tp"]) == len(line["t"])


def test_streamed_logprobs_match_the_token_they_describe(streamed) -> None:
    """The alignment property, checked per token across all batches."""
    for line in (x for x in streamed if "t" in x):
        for tok, lp, cands in zip(line["t"], line["lp"], line["tp"]):
            assert lp == StubEngine.fake_logprob(tok)
            assert cands[0][0] == tok


def test_stream_covers_exactly_the_emitted_tokens(streamed) -> None:
    flat = [t for x in streamed if "t" in x for t in x["t"]]
    n_lp = sum(len(x["lp"]) for x in streamed if "t" in x)
    assert n_lp == len(flat)
    assert flat[0] == FIRST


# --------------------------------------------------------------------------- #
# an engine that cannot do it must say so, not answer without the field
# --------------------------------------------------------------------------- #


def test_unsupported_engine_returns_501_not_a_silent_omission() -> None:
    r = _client(_NoLogprobsEngine(FIXED)).post("/pd/decode", json=_body(top_logprobs=1))
    assert r.status_code == 501
    assert r.json()["error_type"] == "logprobs_unavailable"


def test_unsupported_engine_still_serves_requests_without_logprobs() -> None:
    r = _client(_NoLogprobsEngine(FIXED)).post("/pd/decode", json=_body())
    assert r.status_code == 200
    assert "logprobs" not in r.json()


def test_multi_batch_stream_keeps_logprobs_aligned() -> None:
    """One token per batch, so a wrong batch offset shows up."""
    with _client(_SlowEngine(FIXED)).stream(
        "POST", "/pd/decode", json=_body(stream=True, top_logprobs=2)
    ) as resp:
        lines = [json.loads(x) for x in resp.iter_lines() if x]
    tok_lines = [x for x in lines if "t" in x]
    assert len(tok_lines) > 1, "expected several batches"
    for line in tok_lines:
        for tok, lp, cands in zip(line["t"], line["lp"], line["tp"]):
            assert lp == StubEngine.fake_logprob(tok)
            assert cands[0][0] == tok


def test_engine_omitting_a_logprob_is_a_501_not_a_sentinel() -> None:
    r = _client(_ShortLogprobsEngine(FIXED)).post("/pd/decode", json=_body(top_logprobs=1))
    assert r.status_code == 501
    assert r.json()["error_type"] == "logprobs_unavailable"


def test_first_token_may_be_bare_and_travels_as_null() -> None:
    """Position 0 is the one legitimate omission, and it must not be a sentinel.

    The decode node never sampled ``first_token_id``, so it has no value to
    report there. Sending -9999.0 would be indistinguishable from a real
    measurement, and refusing would make every logprobs request a 501, so the
    slot is held and sent as null for the router to fill from prefill.
    """
    r = _client(_EchoFirstEngine(FIXED)).post("/pd/decode", json=_body(top_logprobs=1))
    assert r.status_code == 200
    body = r.json()
    lp = body["logprobs"]["lp"]
    assert len(lp) == len(body["token_ids"]), "one entry per token"
    assert lp[0] is None, "token 1 carries no decode-side value"
    assert body["logprobs"]["tp"][0] == [], "and no candidate row"
    assert all(v is not None for v in lp[1:]), "every later token has one"


def test_first_token_bare_is_still_refused_when_a_later_one_is_too() -> None:
    """Tolerating position 0 must not weaken the rule for the rest.

    Guards against the obvious over-correction: accepting any bare emit once the
    first has been seen would let a real engine fault through as a sentinel.
    """
    r = _client(_ShortLogprobsEngine(FIXED)).post("/pd/decode", json=_body(top_logprobs=1))
    assert r.status_code == 501


def test_the_streaming_worker_keeps_the_logprobs_type():
    """Both branches of the decode server report the same inability the same way.

    The blocking branch answers 501 `logprobs_unavailable`. The streaming worker
    caught `LogprobsUnavailable` in its generic handler and emitted only
    `{"error": ...}`, so the type was lost and the router -- which cannot see a
    status inside a 200 body -- reported 502. Adding a `stop` string is what puts
    a non-streaming request on that protocol, so the same failure changed status
    depending on an unrelated field.
    """
    import pathlib as _p

    src = _p.Path(decode_server.__file__).read_text()
    run = src[src.index("        def _run():") : src.index("worker = threading.Thread")]
    assert "except LogprobsUnavailable" in run, (
        "the streaming worker no longer classifies LogprobsUnavailable, so the "
        "router sees an untyped error and answers 502"
    )
    typed = run[run.index("except LogprobsUnavailable") :]
    assert '"error_type": "logprobs_unavailable"' in typed


from tilert.pd_vllm import decode_server  # noqa: E402
