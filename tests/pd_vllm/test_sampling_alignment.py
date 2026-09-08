"""The two PD legs must sample under the same rules.

The vLLM prefill instance samples token 1; the decode node samples tokens 2..N.
Anything they resolve independently can disagree, and the response carries no
sign of it. Two such disagreements are pinned here:

* ``top_p`` -- the adapters each defaulted it to 0.95 while vLLM resolved its own
  default through ``client value > generation_config.json > 1.0``. A client that
  sent ``temperature`` but no ``top_p`` had its first token drawn from one
  nucleus and the rest from another.
* ``ignore_eos`` -- forwarded to the decode node and honoured by the adapter; an
  adapter that dropped it made a fixed-length benchmark stop at the first EOS and
  report throughput for a shorter reply than it asked for.

CPU only -- no GPU, no tilert, no real vLLM.

Run:
  CUDA_VISIBLE_DEVICES= python -m pytest \
      tests/pd_vllm/test_sampling_alignment.py -v
"""

import pathlib
import types

import pytest
from fastapi.testclient import TestClient

from tilert.pd_vllm import pd_router
from tilert.pd_vllm.generation_defaults import GenerationDefaults
from tilert.pd_vllm.profiles.mla_nsa import MlaNsaEngineAdapter
from tilert.pd_vllm.sampling import VLLM_DEFAULT_TOP_P, resolve_top_p

ADAPTER_MODULES = (
    "tilert.pd_vllm.profiles.mla_nsa",
    "tilert.pd_vllm.profiles.glm5_rocm_engine",
)


# --------------------------------------------------------------------------- #
# resolve_top_p: the single resolution point
# --------------------------------------------------------------------------- #
def test_the_baseline_is_vllms_framework_default():
    """vLLM's ``_DEFAULT_SAMPLING_PARAMS["top_p"]`` is 1.0, not 0.95."""
    assert VLLM_DEFAULT_TOP_P == 1.0
    assert resolve_top_p({}) == 1.0


def test_an_explicit_value_wins_over_the_default():
    assert resolve_top_p({"top_p": 0.8}, 0.95) == 0.8


def test_an_explicit_null_falls_through_to_the_default():
    """SDKs spell "unset" as an explicit null; vLLM resolves it as absent."""
    assert resolve_top_p({"top_p": None}, 0.7) == 0.7


def test_the_deployment_default_stands_in_for_generation_config():
    """A deployment whose checkpoint recommends 0.95 sets it once, and BOTH legs
    are handed that number -- which is the property that matters, not the value.
    """
    assert resolve_top_p({}, 0.95) == 0.95


def test_a_string_is_coerced_like_the_other_resolvers():
    assert resolve_top_p({"top_p": "0.5"}) == 0.5


def test_a_bool_is_not_a_top_p():
    with pytest.raises(ValueError):
        resolve_top_p({"top_p": True})


@pytest.mark.parametrize("module", ADAPTER_MODULES)
def test_no_adapter_carries_its_own_top_p_default(module):
    """Source-level, in the shape test_top_k_resolution.py uses: a re-introduced
    literal default is the exact regression this file exists for, and it would
    otherwise only show up as a quality drift nobody can attribute.
    """
    src = pathlib.Path(pytest.importorskip(module).__file__).read_text()
    assert (
        'sampling.get("top_p"' not in src
    ), f"{module} must consume the resolved value, not default it again"
    assert "0.95" not in src


# --------------------------------------------------------------------------- #
# Both legs receive the same resolved value
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


def _both_legs(monkeypatch, body, defaults=None):
    """Drive one request through the router; return (prefill_body, decode_body)."""
    seen = {}

    def fake_get(url, timeout=None, **kw):
        return _Resp({"capabilities": {"penalties": True, "ignore_eos": True}})

    def fake_post(url, json=None, timeout=None, **kw):
        if url.endswith("/pd/decode"):
            seen["decode"] = json
            return _Resp(
                {"rid": "x", "token_ids": [7], "seq_len": 8, "timing_ms": {"finish_reason": "stop"}}
            )
        seen["prefill"] = json
        return _Resp(
            {
                "id": "cmpl-abc",
                "choices": [{"logprobs": {"content": [{"token": "token_id:7"}]}}],
                "usage": {"prompt_tokens": 3},
                "model": "m",
            }
        )

    monkeypatch.setattr(pd_router.requests, "get", fake_get)
    monkeypatch.setattr(pd_router.requests, "post", fake_post)
    pool = pd_router.Pool([pd_router.DecodeNode("127.0.0.1", 5556, 5557)])
    ctx = pd_router.RouterCtx(
        "http://vllm.invalid", pool, tokenizer=None, parser_name="none", gen_defaults=defaults
    )
    r = TestClient(pd_router.build_app(ctx)).post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}], **body}
    )
    assert r.status_code == 200, r.text
    return seen["prefill"], seen["decode"]


def test_an_absent_top_p_is_pinned_identically_on_both_legs(monkeypatch):
    prefill, decode = _both_legs(monkeypatch, {"temperature": 0.7})
    assert prefill["top_p"] == decode["sampling"]["top_p"] == 1.0


def test_an_explicit_top_p_reaches_both_legs_unchanged(monkeypatch):
    prefill, decode = _both_legs(monkeypatch, {"top_p": 0.8})
    assert prefill["top_p"] == decode["sampling"]["top_p"] == 0.8


def test_the_deployment_default_reaches_both_legs(monkeypatch):
    """A deployment's resolved defaults must land on both legs identically.

    This is what keeps a checkpoint's recommended sampling (say temperature
    0.6 / top_p 0.95 / top_k 20) from applying to the first token only.
    """
    prefill, decode = _both_legs(
        monkeypatch, {}, defaults=GenerationDefaults(temperature=0.6, top_p=0.95, top_k=20)
    )
    for field, want in (("temperature", 0.6), ("top_p", 0.95), ("top_k", 20)):
        assert prefill[field] == decode["sampling"][field] == want, field


def test_the_prefill_leg_is_pinned_even_for_a_greedy_request(monkeypatch):
    """Greedy ignores top_p, but leaving it unset on one leg only would let a
    generation_config default reappear the moment temperature rises.
    """
    prefill, decode = _both_legs(monkeypatch, {"temperature": 0.0})
    assert prefill["top_p"] == decode["sampling"]["top_p"] == 1.0


@pytest.mark.parametrize(
    "flag",
    [
        "--model",
        "--generation-config",
        "--default-temperature",
        "--default-top-p",
        "--default-top-k",
        "--default-repetition-penalty",
    ],
)
def test_the_cli_exposes_the_default_resolution_knobs(flag):
    """Mirrors vLLM's own surface (--generation-config plus per-key overrides).

    Without them, matching a checkpoint's recommended sampling would mean
    editing code -- and the last time that was true, three adapters ended up
    each carrying their own literal.
    """
    src = pathlib.Path(pd_router.__file__).read_text()
    assert f'"{flag}"' in src


# --------------------------------------------------------------------------- #
# ignore_eos is honoured by the adapter
# --------------------------------------------------------------------------- #
def _mla_adapter():
    ad = object.__new__(MlaNsaEngineAdapter)
    ad.gen = types.SimpleNamespace(update_sampling_params=lambda **kw: None)
    ad.with_mtp = False
    ad.mtp_seq_len = 4
    ad.max_seq_len = 4096
    ad._seq_len = 8
    ad.stop_ids = {7, 8}
    ad._ignore_eos = False
    ad.last_stats = {}
    return ad


def test_the_mla_nsa_adapter_still_honours_the_flag():
    ad = _mla_adapter()
    ad.decode(5, 0, {"ignore_eos": True}, cancel_event=None, grammar_session=None)
    assert ad._ignore_eos is True


@pytest.mark.parametrize("adapter_factory", [_mla_adapter])
def test_every_adapter_declares_whether_it_honours_ignore_eos(adapter_factory):
    """The router refuses the field on any node that does not declare it, so an
    adapter that honours it and stays silent loses the feature.
    """
    ad = adapter_factory()
    assert ad.supports_ignore_eos() is True


# --------------------------------------------------------------------------- #
# Penalties: declared per adapter, and refused rather than silently dropped
# --------------------------------------------------------------------------- #
def test_the_mla_nsa_adapter_declares_no_penalty_support():
    assert _mla_adapter().supports_penalties() is False


@pytest.mark.parametrize(
    "sampling",
    [
        {"repetition_penalty": 1.2},
        {"presence_penalty": 0.4},
    ],
)
def test_the_mla_nsa_adapter_refuses_a_penalty_it_cannot_apply(sampling):
    """It used to accept the parameter and decode unpenalised -- the one thing
    an adapter must never do.
    """
    with pytest.raises(NotImplementedError):
        _mla_adapter().decode(5, 0, sampling, cancel_event=None, grammar_session=None)


@pytest.mark.parametrize(
    "sampling",
    [
        {},
        {"repetition_penalty": 1.0},
        {"presence_penalty": 0.0},
        {"repetition_penalty": None, "presence_penalty": None},
    ],
)
def test_the_mla_nsa_adapter_serves_neutral_penalties(sampling):
    ad = _mla_adapter()
    assert ad.decode(5, 0, sampling, cancel_event=None, grammar_session=None) == [5]
