"""Every generation field a client can send must be executed or refused.

The PD split samples token 1 on the vLLM prefill instance and tokens 2..N on the
decode node. The client body reaches vLLM almost verbatim, while the decode node
gets only the keys ``pd_router._sampling_of`` forwards. A field in the gap is
applied to token 1 and dropped for the rest of the reply, and the response is a
200 that quietly violates what was asked for.

These tests pin the closed set: for each such field, either the request is
refused before anything observable happens, or the field is forwarded to a node
that declared it can execute it.

CPU only -- no GPU, no tilert, no real vLLM.

Run:
  CUDA_VISIBLE_DEVICES= python -m pytest \
      tests/pd_vllm/test_request_capabilities.py -v
"""

import queue
import types

import pytest
from fastapi.testclient import TestClient

from tilert.pd_vllm import pd_router
from tilert.pd_vllm.capabilities import (
    PROFILE_FIELD_NAMES,
    STATIC_FIELD_NAMES,
    CapabilityUnavailable,
    InvalidParameter,
    NodeCapabilities,
    engine_capabilities,
    validate_generation_request,
)
from tilert.pd_vllm.decode_server import build_app as build_decode_app
from tilert.pd_vllm.engine_iface import StubEngine

# One non-neutral value per statically unsupported field, i.e. a value that asks
# for behaviour the decode node cannot produce.
LIVE_VALUES = {
    "stop_token_ids": [151643],
    "min_tokens": 16,
    "frequency_penalty": 0.5,
    "min_p": 0.05,
    "seed": 42,
    "logit_bias": {"151643": -100.0},
    "bad_words": ["foo"],
    "allowed_token_ids": [1, 2, 3],
    "structured_outputs": {"json": {"type": "object"}},
    "n": 2,
    "best_of": 4,
    "use_beam_search": True,
    "prompt_logprobs": 1,
    "logprob_token_ids": [7],
    "skip_special_tokens": False,
}

# The value vLLM itself would have used, so honouring and ignoring it agree.
NEUTRAL_VALUES = {
    "stop_token_ids": [],
    "min_tokens": 0,
    "frequency_penalty": 0.0,
    "min_p": 0.0,
    "seed": None,
    "logit_bias": {},
    "bad_words": [],
    "allowed_token_ids": None,
    "structured_outputs": None,
    "n": 1,
    "best_of": 1,
    "use_beam_search": False,
    "prompt_logprobs": None,
    "logprob_token_ids": [],
    "skip_special_tokens": True,
}

FULL_CAPS = NodeCapabilities(penalties=True, ignore_eos=True)
NO_CAPS = NodeCapabilities()


def _body(**extra):
    return {"messages": [{"role": "user", "content": "hi"}], **extra}


# --------------------------------------------------------------------------- #
# The field tables are complete and self-consistent
# --------------------------------------------------------------------------- #
def test_every_static_field_has_a_live_and_a_neutral_fixture():
    """A field added to the gate without a fixture here would be untested."""
    assert set(STATIC_FIELD_NAMES) == set(LIVE_VALUES) == set(NEUTRAL_VALUES)


def test_no_forwarded_field_is_left_ungated():
    """``_sampling_of``'s whitelist and the gate must partition the fields.

    A field that is forwarded AND statically refused is a contradiction; a field
    that is neither forwarded nor gated is the original bug. The only fields
    allowed to be forwarded ungated are the three the decode sampler implements
    unconditionally.
    """
    forwarded = {
        "temperature",
        "top_p",
        "top_k",
        "repetition_penalty",
        "presence_penalty",
        "ignore_eos",
    }
    unconditional = {"temperature", "top_p", "top_k"}
    assert forwarded.isdisjoint(STATIC_FIELD_NAMES)
    assert forwarded - unconditional == set(PROFILE_FIELD_NAMES)


# --------------------------------------------------------------------------- #
# validate_generation_request: the unit contract
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("field", sorted(STATIC_FIELD_NAMES))
def test_a_live_static_field_is_refused(field):
    with pytest.raises(CapabilityUnavailable) as e:
        validate_generation_request(_body(**{field: LIVE_VALUES[field]}), FULL_CAPS)
    # The message must name the field, so an operator reading a client's error
    # knows which one to drop.
    assert field in str(e.value)


@pytest.mark.parametrize("field", sorted(STATIC_FIELD_NAMES))
def test_a_neutral_static_field_is_accepted(field):
    """Mentioning a field is not asking for it.

    Refusing the neutral value would reject clients and SDKs that send the whole schema with
    defaults filled in -- which is what made a blanket "unknown field" rejection unusable.
    """
    validate_generation_request(_body(**{field: NEUTRAL_VALUES[field]}), FULL_CAPS)


@pytest.mark.parametrize("field", sorted(STATIC_FIELD_NAMES))
def test_an_absent_static_field_is_accepted(field):
    validate_generation_request(_body(), FULL_CAPS)


def test_n_greater_than_one_is_rejected():
    with pytest.raises(CapabilityUnavailable):
        validate_generation_request(_body(n=2), FULL_CAPS)


def test_n_of_one_is_accepted():
    validate_generation_request(_body(n=1), FULL_CAPS)


def test_n_below_one_is_a_client_error_not_a_capability_gap():
    """``n=0`` is not "a feature we lack"; no backend would serve it."""
    with pytest.raises(InvalidParameter):
        validate_generation_request(_body(n=0), FULL_CAPS)


def test_a_string_where_a_number_belongs_is_a_client_error():
    with pytest.raises(InvalidParameter):
        validate_generation_request(_body(frequency_penalty="high"), FULL_CAPS)


def test_a_bool_is_not_a_number():
    """bool is an int subclass in Python; ``min_p: true`` is a type error."""
    with pytest.raises(InvalidParameter):
        validate_generation_request(_body(min_p=True), FULL_CAPS)


def test_a_number_where_a_flag_belongs_is_a_client_error():
    with pytest.raises(InvalidParameter):
        validate_generation_request(_body(use_beam_search=1), FULL_CAPS)


def test_explicit_null_is_treated_as_absent():
    """An explicit null is how SDKs spell "unset", and vLLM resolves it to the
    same default as an absent key.
    """
    for field in STATIC_FIELD_NAMES:
        validate_generation_request(_body(**{field: None}), FULL_CAPS)


def test_error_payload_shape_matches_the_other_gates():
    for exc in (CapabilityUnavailable("x"), InvalidParameter("y")):
        payload = exc.to_payload()
        assert set(payload) == {"error", "error_type"}
    assert CapabilityUnavailable.http_status == 501
    assert InvalidParameter.http_status == 400


# --------------------------------------------------------------------------- #
# Profile-dependent fields follow the node's declaration
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "field,value",
    [
        ("repetition_penalty", 1.2),
        ("presence_penalty", 0.4),
    ],
)
def test_penalties_need_a_node_that_declares_them(field, value):
    validate_generation_request(_body(**{field: value}), FULL_CAPS)
    with pytest.raises(CapabilityUnavailable):
        validate_generation_request(_body(**{field: value}), NO_CAPS)


@pytest.mark.parametrize(
    "field,neutral",
    [
        ("repetition_penalty", 1.0),
        ("presence_penalty", 0.0),
    ],
)
def test_neutral_penalties_pass_on_a_node_without_them(field, neutral):
    validate_generation_request(_body(**{field: neutral}), NO_CAPS)


def test_ignore_eos_is_allowed_only_where_it_is_honoured():
    validate_generation_request(_body(ignore_eos=True), FULL_CAPS)
    with pytest.raises(CapabilityUnavailable):
        validate_generation_request(_body(ignore_eos=True), NO_CAPS)
    # False asks for nothing, so it needs no capability.
    validate_generation_request(_body(ignore_eos=False), NO_CAPS)


def test_unknown_capabilities_fail_closed():
    """``None`` means the router could not establish what the node can do.

    Guessing "supported" would restore the silent-wrong-answer; guessing
    "unsupported" costs an honest 501.
    """
    with pytest.raises(CapabilityUnavailable):
        validate_generation_request(_body(ignore_eos=True), None)


# --------------------------------------------------------------------------- #
# NodeCapabilities: parsing and pool intersection
# --------------------------------------------------------------------------- #
def test_intersection_keeps_only_what_every_node_can_do():
    mixed = FULL_CAPS.intersect(NodeCapabilities(penalties=True))
    assert mixed == NodeCapabilities(penalties=True, ignore_eos=False)


@pytest.mark.parametrize(
    "payload",
    [
        None,
        "nope",
        42,
        [],
        {},
        {"capabilities": "nope"},
        {"capabilities": {}},
        {"capabilities": {"penalties": "yes"}},
    ],
)
def test_an_unusable_payload_declares_nothing(payload):
    assert NodeCapabilities.from_payload(payload) == NO_CAPS


def test_a_well_formed_payload_is_parsed():
    caps = NodeCapabilities.from_payload({"capabilities": {"penalties": True, "ignore_eos": False}})
    assert caps == NodeCapabilities(penalties=True, ignore_eos=False)


def test_a_bare_capability_dict_is_also_accepted():
    assert NodeCapabilities.from_payload({"penalties": True}) == NodeCapabilities(penalties=True)


# --------------------------------------------------------------------------- #
# engine_capabilities: read from the live engine, absent predicate = no support
# --------------------------------------------------------------------------- #
def test_the_stub_engine_declares_everything():
    assert engine_capabilities(StubEngine()) == FULL_CAPS


def test_an_engine_without_the_predicates_declares_nothing():
    assert engine_capabilities(object()) == NO_CAPS


def test_a_predicate_that_raises_is_treated_as_unsupported():
    class _Broken:
        def supports_penalties(self):
            raise RuntimeError("engine went away")

        def supports_ignore_eos(self):
            return True

    assert engine_capabilities(_Broken()) == NodeCapabilities(ignore_eos=True)


def test_a_demoted_adapter_reports_the_demotion_not_the_claim():
    """An adapter that probes for the penalty pre-pass demotes its claim when
    the installed engine lacks it. ``/capabilities`` must show the demotion, or
    the router would pre-approve a request the engine then refuses.
    """

    class _Demoted:
        _supports_penalties = False

        def supports_penalties(self):
            return self._supports_penalties

        def supports_ignore_eos(self):
            return True

    assert engine_capabilities(_Demoted()).penalties is False


# --------------------------------------------------------------------------- #
# decode_server: /capabilities, and the pre-wire-wait refusal
# --------------------------------------------------------------------------- #
class _FakeReq:
    rid = "rid-1"
    seq_len = 8
    last_prompt_token = 5


class _FakeServer:
    def __init__(self):
        self.completed: queue.Queue = queue.Queue()
        self.completed.put(_FakeReq())
        self.profile = types.SimpleNamespace(
            convert=lambda *a, **k: "converted", num_ranks=8, name="stub"
        )
        self.buffer = None
        self.base_ptr = 0
        self.max_seq_len = 4096
        self.released = 0

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


class _NoPenaltyEngine(StubEngine):
    def supports_penalties(self) -> bool:
        return False


def _decode_client(engine=None):
    return TestClient(build_decode_app(_FakeServer(), engine or StubEngine()))


def test_capabilities_endpoint_reports_the_engine():
    r = _decode_client().get("/capabilities")
    assert r.status_code == 200
    body = r.json()
    assert body["engine"] == "StubEngine"
    assert body["capabilities"]["penalties"] is True
    assert body["capabilities"]["ignore_eos"] is True


def test_capabilities_endpoint_follows_the_engine_that_cannot():
    r = _decode_client(_NoPenaltyEngine()).get("/capabilities")
    assert r.json()["capabilities"]["penalties"] is False


def test_decode_refuses_a_penalty_the_engine_cannot_apply():
    """501 rather than a 200 whose tokens were never penalised."""
    r = _decode_client(_NoPenaltyEngine()).post(
        "/pd/decode",
        json={
            "rid": "rid-1",
            "first_token_id": 7,
            "max_tokens": 8,
            "sampling": {"repetition_penalty": 1.3},
        },
    )
    assert r.status_code == 501
    assert r.json()["error_type"] == "capability_unavailable"


def test_decode_serves_a_neutral_penalty_on_the_same_engine():
    r = _decode_client(_NoPenaltyEngine()).post(
        "/pd/decode",
        json={
            "rid": "rid-1",
            "first_token_id": 7,
            "max_tokens": 8,
            "sampling": {"repetition_penalty": 1.0, "temperature": 0.6},
        },
    )
    assert r.status_code == 200


def test_a_refused_request_hands_back_the_receive_slot():
    """Otherwise one refused request stalls the NEXT one for the full
    kv_transfer_timeout -- the failure ``_abandon_pending_kv`` exists for.
    """
    server = _FakeServer()
    client = TestClient(build_decode_app(server, _NoPenaltyEngine()))
    r = client.post(
        "/pd/decode",
        json={
            "rid": "rid-1",
            "first_token_id": 7,
            "max_tokens": 8,
            "sampling": {"presence_penalty": 0.5},
        },
    )
    assert r.status_code == 501
    assert server.released >= 1
    # ... and the node is not left busy.
    assert client.get("/decode_status").json()["status"] == "idle"


# --------------------------------------------------------------------------- #
# pd_router: refused before prefill, before a node is acquired
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


def _router(monkeypatch, caps=FULL_CAPS, on_post=None):
    """Router whose single decode node declares ``caps``."""

    def fake_get(url, timeout=None, **kw):
        assert url.endswith("/capabilities"), url
        return _Resp({"capabilities": caps.to_payload()})

    def default_post(url, json=None, timeout=None, **kw):
        raise AssertionError(f"network must not be touched: {url}")

    monkeypatch.setattr(pd_router.requests, "get", fake_get)
    monkeypatch.setattr(pd_router.requests, "post", on_post or default_post)
    pool = pd_router.Pool([pd_router.DecodeNode("127.0.0.1", 5556, 5557)])
    ctx = pd_router.RouterCtx("http://vllm.invalid", pool, tokenizer=None, parser_name="none")
    return TestClient(pd_router.build_app(ctx)), pool


@pytest.mark.parametrize("field", sorted(STATIC_FIELD_NAMES))
def test_a_live_static_field_is_refused_before_prefill(monkeypatch, field):
    """The refusal must cost nothing: no vLLM call, so no prompt is prefilled
    and no KV is pushed to a node that will never consume it.
    """
    client, pool = _router(monkeypatch)
    r = client.post("/v1/chat/completions", json=_body(**{field: LIVE_VALUES[field]}))
    assert r.status_code == 501
    assert r.json()["error_type"] == "capability_unavailable"
    # ... and no decode node was reserved.
    assert all(not n.busy for n in pool.nodes)


def test_a_refused_stream_request_also_reserves_nothing(monkeypatch):
    client, pool = _router(monkeypatch)
    r = client.post("/v1/chat/completions", json=_body(seed=7, stream=True))
    assert r.status_code == 501
    assert all(not n.busy for n in pool.nodes)


def test_a_client_error_is_400_not_501(monkeypatch):
    client, _ = _router(monkeypatch)
    r = client.post("/v1/chat/completions", json=_body(n=0))
    assert r.status_code == 400
    assert r.json()["error_type"] == "invalid_parameter"


def test_completions_endpoint_is_gated_too(monkeypatch):
    client, _ = _router(monkeypatch)
    r = client.post("/v1/completions", json={"prompt": "hi", "seed": 1})
    assert r.status_code == 501


def test_a_node_that_declares_penalties_gets_the_request(monkeypatch):
    captured = {}

    def on_post(url, json=None, timeout=None, **kw):
        if url.endswith("/pd/decode"):
            captured["decode"] = json
            return _Resp(
                {"rid": "x", "token_ids": [7], "seq_len": 8, "timing_ms": {"finish_reason": "stop"}}
            )
        captured["prefill"] = json
        return _Resp(
            {
                "id": "cmpl-abc",
                "choices": [{"logprobs": {"content": [{"token": "token_id:7"}]}}],
                "usage": {"prompt_tokens": 3},
                "model": "m",
            }
        )

    client, _ = _router(monkeypatch, on_post=on_post)
    r = client.post("/v1/chat/completions", json=_body(repetition_penalty=1.2, ignore_eos=True))
    assert r.status_code == 200
    assert captured["decode"]["sampling"]["repetition_penalty"] == 1.2
    assert captured["decode"]["sampling"]["ignore_eos"] is True


def test_a_node_that_declares_nothing_refuses_the_same_request(monkeypatch):
    client, _ = _router(monkeypatch, caps=NO_CAPS)
    r = client.post("/v1/chat/completions", json=_body(repetition_penalty=1.2))
    assert r.status_code == 501


def test_a_failed_capability_probe_refuses_rather_than_assumes(monkeypatch):
    def boom(url, timeout=None, **kw):
        raise OSError("connection refused")

    monkeypatch.setattr(pd_router.requests, "get", boom)
    monkeypatch.setattr(pd_router.requests, "post", lambda *a, **k: pytest.fail("must not prefill"))
    pool = pd_router.Pool([pd_router.DecodeNode("127.0.0.1", 5556, 5557)])
    ctx = pd_router.RouterCtx("http://vllm.invalid", pool, tokenizer=None, parser_name="none")
    r = TestClient(pd_router.build_app(ctx)).post(
        "/v1/chat/completions", json=_body(ignore_eos=True)
    )
    assert r.status_code == 501


def test_a_plain_request_survives_an_unreachable_probe(monkeypatch):
    """Fail-closed must cost only the optional fields: a request that asks for
    nothing profile-dependent still goes through when the probe fails.
    """

    def boom(url, timeout=None, **kw):
        raise OSError("connection refused")

    def on_post(url, json=None, timeout=None, **kw):
        if url.endswith("/pd/decode"):
            return _Resp(
                {"rid": "x", "token_ids": [7], "seq_len": 8, "timing_ms": {"finish_reason": "stop"}}
            )
        return _Resp(
            {
                "id": "cmpl-abc",
                "choices": [{"logprobs": {"content": [{"token": "token_id:7"}]}}],
                "usage": {"prompt_tokens": 3},
                "model": "m",
            }
        )

    monkeypatch.setattr(pd_router.requests, "get", boom)
    monkeypatch.setattr(pd_router.requests, "post", on_post)
    pool = pd_router.Pool([pd_router.DecodeNode("127.0.0.1", 5556, 5557)])
    ctx = pd_router.RouterCtx("http://vllm.invalid", pool, tokenizer=None, parser_name="none")
    r = TestClient(pd_router.build_app(ctx)).post(
        "/v1/chat/completions", json=_body(temperature=0.6)
    )
    assert r.status_code == 200


def test_the_pool_intersects_every_node(monkeypatch):
    """Validation runs before a node is chosen, so a field is only safe if EVERY
    node could have executed it.
    """

    def fake_get(url, timeout=None, **kw):
        # :5557 supports penalties, :5559 does not.
        supports = ":5557/" in url
        return _Resp({"capabilities": {"penalties": supports, "ignore_eos": True}})

    monkeypatch.setattr(pd_router.requests, "get", fake_get)
    pool = pd_router.Pool(
        [
            pd_router.DecodeNode("127.0.0.1", 5556, 5557),
            pd_router.DecodeNode("127.0.0.1", 5558, 5559),
        ]
    )
    assert pool.capabilities() == NodeCapabilities(penalties=False, ignore_eos=True)


def test_a_probe_result_is_cached(monkeypatch):
    calls = {"n": 0}

    def fake_get(url, timeout=None, **kw):
        calls["n"] += 1
        return _Resp({"capabilities": FULL_CAPS.to_payload()})

    monkeypatch.setattr(pd_router.requests, "get", fake_get)
    pool = pd_router.Pool([pd_router.DecodeNode("127.0.0.1", 5556, 5557)])
    for _ in range(5):
        pool.capabilities()
    assert calls["n"] == 1


def test_a_failed_probe_is_not_cached(monkeypatch):
    """A node that comes back must be picked up on the next request, not after
    a whole TTL -- the router is meant to survive an independent restart.
    """
    calls = {"n": 0}

    def fake_get(url, timeout=None, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("down")
        return _Resp({"capabilities": FULL_CAPS.to_payload()})

    monkeypatch.setattr(pd_router.requests, "get", fake_get)
    pool = pd_router.Pool([pd_router.DecodeNode("127.0.0.1", 5556, 5557)])
    assert pool.capabilities() == NO_CAPS
    assert pool.capabilities() == FULL_CAPS


def test_an_empty_pool_declares_nothing():
    assert pd_router.Pool([]).capabilities() == NO_CAPS


# --------------------------------------------------------------------------- #
# Review findings on PR #40 (codex)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("field", ["seed", "prompt_logprobs"])
def test_zero_is_a_request_not_an_absence(field):
    """ "Falsy" and "asks for nothing" are different questions.

    ``seed: 0`` is a valid seed and ``prompt_logprobs: 0`` asks for the prompt
    tokens' own log probabilities. Treating either as neutral lets the request
    through, and the prefill instance then applies it to the first token while
    the decode node cannot -- the exact split this gate exists to stop.
    """
    with pytest.raises(CapabilityUnavailable):
        validate_generation_request(_body(**{field: 0}), FULL_CAPS)


@pytest.mark.parametrize("field", ["seed", "prompt_logprobs"])
def test_only_an_explicit_null_is_neutral_for_those_fields(field):
    validate_generation_request(_body(**{field: None}), FULL_CAPS)


@pytest.mark.parametrize(
    "field", ["logit_bias", "bad_words", "stop_token_ids", "logprob_token_ids"]
)
def test_an_empty_collection_is_still_neutral(field):
    """Unlike the two above, an empty collection genuinely asks for nothing --

    the distinction is whether the field has a meaningful zero, not whether the
    value is falsy.
    """
    validate_generation_request(_body(**{field: NEUTRAL_VALUES[field]}), FULL_CAPS)


@pytest.mark.parametrize(
    "body",
    [
        {"stop": ["\n\n"]},
        {"stop": "END"},
        {"stop": ["A"], "include_stop_str_in_output": True},
    ],
)
def test_the_gate_does_not_own_stop_strings(body):
    """The router matches them over the reply text, so the gate lets them by.

    Refusing them here would make the router's implementation unreachable.
    Their validation lives in ``request_gate.resolve_stop_request``, which is where
    the tokenizer that executes them is; see test_stop_semantics.py.
    """
    validate_generation_request(_body(**body), FULL_CAPS)


# ── the three always-supported fields are resolved, so they are type-checked ──
@pytest.mark.parametrize("value", [20, "20", 20.0])
def test_an_integral_top_k_is_accepted_whatever_its_spelling(value):
    """As permissive as vLLM: rejecting ``"20"`` would turn a request vLLM
    serves into a 400.
    """
    validate_generation_request(_body(top_k=value), FULL_CAPS)


@pytest.mark.parametrize("value", [1.9, "1.9", True])
def test_a_non_integral_top_k_is_refused_not_truncated(value):
    """The router resolves top_k and writes the result into the prefill request,
    which overwrites what the client sent -- so vLLM no longer gets to reject a
    bad value. Truncating 1.9 to 1 would have both legs serve a materially
    different request and report success.
    """
    with pytest.raises(InvalidParameter):
        validate_generation_request(_body(top_k=value), FULL_CAPS)


@pytest.mark.parametrize(
    "field,value",
    [
        ("temperature", "0.7"),
        ("top_p", "0.95"),
        ("temperature", 0),
        ("top_p", 1),
    ],
)
def test_a_usable_temperature_or_top_p_is_accepted(field, value):
    validate_generation_request(_body(**{field: value}), FULL_CAPS)


@pytest.mark.parametrize(
    "field,value",
    [
        ("temperature", "hot"),
        ("top_p", "wide"),
        ("temperature", True),
        ("top_p", True),
        ("temperature", -1),
        ("top_p", -0.5),
    ],
)
def test_an_unusable_temperature_or_top_p_is_a_client_error(field, value):
    with pytest.raises(InvalidParameter):
        validate_generation_request(_body(**{field: value}), FULL_CAPS)


@pytest.mark.parametrize("field", ["temperature", "top_p", "top_k"])
def test_the_always_supported_fields_are_never_refused_for_capability(field):
    """They are type-checked, not gated: every profile's decode path applies all
    three on every request.
    """
    assert field not in STATIC_FIELD_NAMES
    assert field not in PROFILE_FIELD_NAMES
    validate_generation_request(_body(**{field: 1}), NO_CAPS)
