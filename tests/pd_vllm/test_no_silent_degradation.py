"""Every request field the router owns is executed, refused, or declared inert.

The capability gate already has this guard for its own 15 fields:
`test_every_static_field_has_a_live_and_a_neutral_fixture` fails if one is added
without a fixture. The fields the ROUTER handles -- `stop`, `logprobs`,
`chat_template_kwargs` and the rest -- had no such enumeration, and that is where
seven consecutive review findings landed. Each was the same defect:

    200 OK, and the field silently did not take effect.

That is worse than a refusal, because the client is told the request succeeded.
The codebase's stated rule is the opposite -- accept only what can be executed
completely, refuse the rest immediately -- and nothing enforced it.

Two guards here:

* :data:`ROUTER_FIELDS` enumerates them. A field added to the router without an
  entry fails :func:`test_every_router_field_is_accounted_for`, so the next one
  cannot be forgotten rather than found.
* Each entry declares what a request carrying it must produce: ``served`` with a
  predicate proving the field took effect, ``refused`` with the status, or
  ``inert`` with the reason it legitimately changes nothing.

No GPU and no weights: the real `build_app` against stub backends.

Run:
  CUDA_VISIBLE_DEVICES= python -m pytest \
      tests/pd_vllm/test_no_silent_degradation.py -v
"""

from __future__ import annotations

import json
import re

import httpx
import pytest

from tests.pd_vllm.test_stop_strings import (  # noqa: E402
    _SEEN,
    _make_decode,
    _make_vllm,
    _serve,
    _Tok,
)
from tilert.pd_vllm.decode_pool import DecodeNode, Pool
from tilert.pd_vllm.pd_router import RouterCtx, build_app

# ── the enumeration ─────────────────────────────────────────────────────────
#
# `served`  -- the request is 200 AND `check(reply, sent)` proves the field took
#              effect. A predicate that only checks the status would pass for a
#              field that was silently dropped, which is the whole point.
#              `reply` is the response JSON; `sent` is the body the decode node
#              received, which is where a FORWARDED field's effect is visible --
#              the stub does not implement `max_tokens`, and asserting on the
#              reply length would pin the stub rather than the router.
# `refused` -- the request must fail with this status and a typed error.
# `inert`   -- 200 with nothing observable, and a reason that says why that is
#              correct rather than a gap.

SERVED, REFUSED, INERT = "served", "refused", "inert"


def _reply(body: dict) -> str:
    msg = body["choices"][0].get("message") or {}
    return (msg.get("reasoning_content") or "") + (msg.get("content") or "")


ROUTER_FIELDS: dict[str, list[tuple]] = {
    "stop": [
        (
            SERVED,
            {"stop": ["STOP"]},
            lambda j, sent: j["choices"][0]["stop_reason"] == "STOP" and "STOP" not in _reply(j),
        ),
        (REFUSED, {"stop": [""]}, 400),
        (REFUSED, {"stop": 5}, 400),
    ],
    "include_stop_str_in_output": [
        (
            SERVED,
            {"stop": ["STOP"], "include_stop_str_in_output": True},
            lambda j, sent: _reply(j).endswith("STOP"),
        ),
        (
            SERVED,
            {"stop": ["STOP"], "include_stop_str_in_output": 1},
            lambda j, sent: _reply(j).endswith("STOP"),
        ),
        (REFUSED, {"stop": ["STOP"], "include_stop_str_in_output": None}, 400),
        (REFUSED, {"stop": ["STOP"], "include_stop_str_in_output": "maybe"}, 400),
    ],
    "logprobs": [
        (
            SERVED,
            {"logprobs": True, "temperature": 0.6},
            lambda j, sent: (j["choices"][0]["logprobs"] or {}).get("content"),
        ),
        (SERVED, {"logprobs": False}, lambda j, sent: j["choices"][0]["logprobs"] is None),
    ],
    "top_logprobs": [
        # Forwarded to the node, which is the only place it can act. Position 0's
        # candidate row comes from the prefill reply and carries one entry, so
        # the response alone cannot show the requested count.
        (
            SERVED,
            {"logprobs": True, "top_logprobs": 2, "temperature": 0.6},
            lambda j, sent: sent.get("top_logprobs") == 2,
        ),
        (REFUSED, {"logprobs": True, "top_logprobs": 99}, 400),
        (REFUSED, {"top_logprobs": 1}, 400),
    ],
    "chat_template_kwargs": [
        (
            SERVED,
            {"chat_template_kwargs": {"enable_thinking": False}},
            lambda j, sent: j["choices"][0]["message"]["content"] is not None,
        ),
        (REFUSED, {"chat_template_kwargs": "bad"}, 400),
        (REFUSED, {"chat_template_kwargs": 5}, 400),
    ],
    "max_tokens": [
        (SERVED, {"max_tokens": 3}, lambda j, sent: sent.get("max_tokens") == 3),
        # Coerced, not rejected: the request model takes these and so must we.
        # Asserting on `sent` is the point: the coerced value has to reach
        # the node, where re-reading the body used to 502 after prefill.
        (SERVED, {"max_tokens": "20.0"}, lambda j, sent: sent.get("max_tokens") == 20),
        (SERVED, {"max_tokens": True}, lambda j, sent: sent.get("max_tokens") == 1),
        # Refused before the prefill, not after it.
        (REFUSED, {"max_tokens": 0}, 400),
        (REFUSED, {"max_tokens": 1.9}, 400),
    ],
    "max_completion_tokens": [
        # vLLM resolves this ahead of `max_tokens`, so the router has to too --
        # otherwise the prefill leg and the decode leg disagree about the length.
        (SERVED, {"max_completion_tokens": 3}, lambda j, sent: sent.get("max_tokens") == 3),
        # Precedence survives the coercion, and the shadowed name is checked
        # for its type but not its range -- as on vLLM.
        (
            SERVED,
            {"max_completion_tokens": "7.0", "max_tokens": 3},
            lambda j, sent: sent.get("max_tokens") == 7,
        ),
        (
            SERVED,
            {"max_completion_tokens": 5, "max_tokens": 0},
            lambda j, sent: sent.get("max_tokens") == 5,
        ),
        (REFUSED, {"max_completion_tokens": 5, "max_tokens": 1.9}, 400),
    ],
    "temperature": [
        (SERVED, {"temperature": 0.0}, lambda j, sent: sent["sampling"].get("temperature") == 0.0),
    ],
    "stream": [
        (INERT, {"stream": False}, "false is the default; the streaming path has its own tests"),
    ],
    "stream_options": [
        (
            INERT,
            {"stream_options": {"include_usage": True}},
            "meaningful only with stream: true, where test_stream_e2e pins it; "
            "stripped from the prefill request so vLLM cannot reject the pair",
        ),
    ],
    "kv_transfer_params": [
        (
            INERT,
            {},
            "set BY the router on the prefill request, never read from the " "client's body",
        ),
    ],
}


@pytest.fixture(scope="module")
def router():
    mp = pytest.MonkeyPatch()
    for var in ("no_proxy", "NO_PROXY"):
        mp.setenv(var, "127.0.0.1,localhost")
    for var in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY"):
        mp.delenv(var, raising=False)
    vllm_port = _serve(_make_vllm())
    decode_port = _serve(_make_decode())
    node = DecodeNode("127.0.0.1", 5556, decode_port)
    ctx = RouterCtx(f"http://127.0.0.1:{vllm_port}", Pool([node]), _Tok(), "none")
    yield f"http://127.0.0.1:{_serve(build_app(ctx))}"
    mp.undo()


def _post(url: str, over: dict) -> httpx.Response:
    body = {"model": "stub", "max_tokens": 64, "messages": [{"role": "user", "content": "hi"}]}
    body.update(over)
    with httpx.Client(timeout=30, trust_env=False) as c:
        return c.post(f"{url}/v1/chat/completions", json=body)


def _cases():
    for field, entries in ROUTER_FIELDS.items():
        for i, entry in enumerate(entries):
            yield pytest.param(field, entry, id=f"{field}-{i}")


@pytest.mark.parametrize("field,entry", list(_cases()))
def test_a_router_field_is_never_silently_dropped(router, field, entry):
    """200 with the field not taking effect is the defect this pins.

    A `served` case is not satisfied by the status alone: the predicate has to
    find the field's effect in the response, because a silently ignored field
    also answers 200.
    """
    kind = entry[0]
    if kind is INERT:
        _, over, why = entry
        r = _post(router, over)
        assert r.status_code == 200, f"{field}: {r.status_code} {r.text[:160]}"
        assert why, "an inert field needs a stated reason"
        return
    if kind is REFUSED:
        _, over, want = entry
        r = _post(router, over)
        assert (
            r.status_code == want
        ), f"{field}: expected {want}, got {r.status_code} {r.text[:160]}"
        assert r.json().get("error_type"), (
            f"{field}: refused without a typed error_type, so a client cannot "
            f"tell what to change"
        )
        return
    _, over, check = entry
    r = _post(router, over)
    assert r.status_code == 200, f"{field}: {r.status_code} {r.text[:200]}"
    assert check(r.json(), _SEEN["decode_body"]), (
        f"{field}: answered 200 but the field did not take effect -- "
        f"{json.dumps(r.json())[:300]}"
    )


def test_every_router_field_is_accounted_for():
    """A field the router reads but does not declare here fails this.

    The point is that the next one is forgotten loudly. Seven review findings
    were the same defect on this surface, each found one at a time.
    """
    src = pytest.importorskip("pathlib").Path(build_app.__globals__["__file__"]).read_text()
    read = set(re.findall(r'body\.get\("([a-z_]+)"', src))
    read |= set(re.findall(r'body\["([a-z_]+)"\]', src))
    read |= set(re.findall(r'"([a-z_]+)" in body', src))
    # Fields other modules read off the same body, resolved on the router's
    # behalf, so they belong to this surface too.
    read |= {"stop", "max_completion_tokens", "temperature"}
    from tilert.pd_vllm.capabilities import STATIC_FIELD_NAMES

    unaccounted = read - set(ROUTER_FIELDS) - set(STATIC_FIELD_NAMES)
    assert not unaccounted, (
        f"the router reads {sorted(unaccounted)} but ROUTER_FIELDS does not say "
        f"whether each is executed, refused, or inert. Add an entry rather than "
        f"leaving the next silent drop to be found in review."
    )


# --------------------------------------------------------------------------- #
# The same rule on a deployment that cannot execute the field at all
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def tokenizerless_router(router):
    """`--parser none` with no `--model-path`: a supported configuration.

    Reuses the module fixture's proxy setup and backends. It has no tokenizer, so
    it cannot match stop strings or name tokens -- and the rule is the same:
    refuse, do not answer 200 with the field quietly inert.
    """
    vllm_port = _serve(_make_vllm())
    decode_port = _serve(_make_decode())
    node = DecodeNode("127.0.0.1", 5556, decode_port)
    ctx = RouterCtx(f"http://127.0.0.1:{vllm_port}", Pool([node]), None, "none")
    return f"http://127.0.0.1:{_serve(build_app(ctx))}"


@pytest.mark.parametrize(
    "over,field",
    [
        ({"stop": ["STOP"]}, "stop"),
        ({"logprobs": True, "top_logprobs": 1, "temperature": 0.6}, "logprobs"),
    ],
)
def test_a_field_this_deployment_cannot_execute_is_refused(tokenizerless_router, over, field):
    """501, not 200 with the field inert.

    Both were silent once: `stop` was ignored and the reply ran past it, and
    `logprobs` came back `null` under a 200 after both backends had computed the
    values. Nothing in the response said the request had not been honoured.
    """
    r = _post(tokenizerless_router, over)
    assert r.status_code == 501, f"{field}: got {r.status_code} {r.text[:200]}"
    assert r.json()["error_type"] == "capability_unavailable"


def test_the_same_deployment_still_serves_what_it_can(tokenizerless_router):
    """The refusals are narrow: a request that asks for neither is served, with
    `logprobs` declared null rather than absent (#43).
    """
    r = _post(tokenizerless_router, {})
    assert r.status_code == 200, r.text[:200]
    assert r.json()["choices"][0]["logprobs"] is None
