"""Stage-1 plumbing tests: grammar_spec end-to-end through the HTTP seams with
a StubEngine — no GPU / no tilert / no real vLLM.

Run:
  CUDA_VISIBLE_DEVICES= python -m pytest \
      tests/pd_vllm/test_grammar_plumbing.py -v

Covers:
  * decode_server: compile-before-inject classification (400/500) returns
    BEFORE the wire-wait; grammar_session threads into decode; runtime
    violation -> 400.
  * pd_router: bad spec -> 400 before any network; grammar_spec forwarded to
    /pd/decode; a decode grammar error propagates its status (not masked 502).
"""

import queue
import types

from fastapi.testclient import TestClient

from tilert.pd_vllm import pd_router
from tilert.pd_vllm.decode_server import build_app
from tilert.pd_vllm.engine_iface import StubEngine


# --------------------------------------------------------------------------- #
# decode_server via TestClient + a fake ReceiveServer
# --------------------------------------------------------------------------- #
class _FakeReq:
    def __init__(self, rid):
        self.rid = rid
        self.seq_len = 8
        self.last_prompt_token = 5


class _FakeServer:
    """Minimal ReceiveServer stand-in: hands back one matching req and
    converts it to a sentinel the StubEngine happily injects.
    """

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


def _client(rid="rid-1"):
    return TestClient(build_app(_FakeServer(rid), StubEngine()))


def test_decode_invalid_grammar_400_before_inject():
    r = _client().post(
        "/pd/decode",
        json={
            "rid": "rid-1",
            "first_token_id": 7,
            "max_tokens": 8,
            "grammar_spec": {"type": "not_a_real_type"},
        },
    )
    assert r.status_code == 400
    assert r.json()["error_type"] == "invalid_grammar"


def test_decode_backend_missing_500():
    r = _client().post(
        "/pd/decode",
        json={
            "rid": "rid-1",
            "first_token_id": 7,
            "max_tokens": 8,
            "grammar_spec": {"type": "__backend_missing__"},
        },
    )
    assert r.status_code == 500
    assert r.json()["error_type"] == "grammar_backend_unavailable"


def test_decode_valid_grammar_threads_and_returns_200():
    r = _client().post(
        "/pd/decode",
        json={
            "rid": "rid-1",
            "first_token_id": 7,
            "max_tokens": 8,
            "grammar_spec": {"type": "regex", "value": "[0-9]"},
            "enable_thinking": False,
        },
    )
    assert r.status_code == 200
    assert r.json()["token_ids"][0] == 7


def test_decode_runtime_violation_400():
    r = _client().post(
        "/pd/decode",
        json={
            "rid": "rid-1",
            "first_token_id": 7,
            "max_tokens": 8,
            "grammar_spec": {"type": "regex", "value": "__violate__"},
        },
    )
    assert r.status_code == 400
    assert r.json()["error_type"] == "grammar_violation"


def test_decode_unconstrained_still_works():
    r = _client().post("/pd/decode", json={"rid": "rid-1", "first_token_id": 7, "max_tokens": 8})
    assert r.status_code == 200
    assert r.json()["token_ids"][0] == 7


# --------------------------------------------------------------------------- #
# pd_router: extraction + forwarding + status propagation (network mocked)
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


def _router_client():
    pool = pd_router.Pool([pd_router.DecodeNode("127.0.0.1", 5556, 5557)])
    ctx = pd_router.RouterCtx("http://vllm.invalid", pool, tokenizer=None, parser_name="none")
    return TestClient(pd_router.build_app(ctx))


def test_router_bad_spec_400_no_network(monkeypatch):
    # If extraction fails, no prefill/decode POST should ever be attempted.
    def _boom(*a, **k):
        raise AssertionError("network must not be touched on bad spec")

    monkeypatch.setattr(pd_router.requests, "post", _boom)
    r = _router_client().post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {"type": "json_schema", "json_schema": {}},
        },
    )
    assert r.status_code == 400
    assert r.json()["error_type"] == "invalid_grammar"


def test_router_forwards_grammar_spec_and_propagates_violation(monkeypatch):
    captured = {}

    def fake_post(url, json=None, timeout=None, **kw):
        if url.endswith("/pd/decode"):
            captured["decode_body"] = json
            # simulate decode-side fail-closed grammar violation
            return _Resp(
                {"error": "first token violates the grammar", "error_type": "grammar_violation"},
                status=400,
            )
        # vLLM prefill response
        return _Resp(
            {
                "id": "cmpl-abc",
                "choices": [{"logprobs": {"content": [{"token": "token_id:7"}]}}],
                "usage": {"prompt_tokens": 3},
                "model": "glm5p2-tilert",
            }
        )

    monkeypatch.setattr(pd_router.requests, "post", fake_post)
    r = _router_client().post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "regex": r"[0-9]{3}"},
    )
    # grammar_spec forwarded to the decode node
    assert captured["decode_body"]["grammar_spec"] == {"type": "regex", "value": r"[0-9]{3}"}
    assert "enable_thinking" in captured["decode_body"]
    # decode's 400 grammar_violation propagates (NOT masked as 502)
    assert r.status_code == 400
    assert r.json()["error_type"] == "grammar_violation"


def test_router_omits_grammar_spec_for_plain_request(monkeypatch):
    captured = {}

    def fake_post(url, json=None, timeout=None, **kw):
        if url.endswith("/pd/decode"):
            captured["decode_body"] = json
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

    monkeypatch.setattr(pd_router.requests, "post", fake_post)
    r = _router_client().post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}
    )
    assert r.status_code == 200
    assert "grammar_spec" not in captured["decode_body"]
