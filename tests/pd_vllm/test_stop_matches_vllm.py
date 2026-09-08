"""``stop`` validation, compared value by value against vLLM's own.

This PR's whole justification for matching vLLM is that the same request should
behave the same on both stacks -- so the comparison is made by running vLLM's
validation next to ours, not by reading its source and reimplementing what it
seems to say. Reading it is how three of these cases were got wrong:

* `include_stop_str_in_output` -- I refused `1` and `"true"`, which vLLM accepts
  (pydantic coerces them), and silently accepted an explicit `null`, which it
  refuses. Exactly backwards.
* `include_stop_str_in_output` without `stop` -- I refused it as meaningless.
  vLLM serves it; with no stop strings the flag is a no-op.

Both directions matter, and they are not symmetric. Being STRICTER than vLLM
turns a request a native endpoint serves into a 400 -- the mistake the
capability gate's `top_k: "20"` handling already warns about. Being MORE LENIENT
serves something vLLM would have refused, which for `stop` means returning an
unrestricted completion to a client who asked for a restricted one.

vLLM validates in two layers and both count:

* ``ChatCompletionRequest`` (pydantic) checks types;
* ``to_sampling_params()`` builds a ``SamplingParams``, whose ``_verify_args``
  checks values -- and this is where an empty stop string dies, with
  ``ValueError("stop cannot contain an empty string.")``.

Comparing against only the first layer says vLLM accepts ``stop: [""]``. It does
not; the serving layer turns that ValueError into a 400.

Needs vllm, so it is skipped in the default dev env:

  CUDA_VISIBLE_DEVICES= python -m pytest \
      tests/pd_vllm/test_stop_matches_vllm.py -v

No GPU and no weights -- request validation only.
"""

from __future__ import annotations

import pytest

pytest.importorskip("vllm")

from vllm.entrypoints.openai.chat_completion.protocol import (  # noqa: E402
    ChatCompletionRequest,
)

from tilert.pd_vllm.capabilities import CapabilityError  # noqa: E402
from tilert.pd_vllm.request_gate import resolve_stop_request  # noqa: E402


class _Tok:
    """A tokenizer only has to exist here; nothing decodes anything."""

    def decode(self, ids, skip_special_tokens=False):
        return ""


def _vllm(body: dict):
    """``(accepted, (stop, include))`` from vLLM's own two layers."""
    try:
        req = ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "hi"}], **body)
    except Exception:
        return False, None
    try:
        params = req.to_sampling_params(max_tokens=8, default_sampling_params={})
    except Exception:
        return False, None
    stop = params.stop
    stop = [stop] if isinstance(stop, str) else list(stop or [])
    return True, (stop, params.include_stop_str_in_output)


def _ours(body: dict):
    try:
        return True, resolve_stop_request(dict(body), _Tok())
    except CapabilityError:
        return False, None


# Every case is a request a client can actually send. The three that were wrong
# before this test existed are marked.
CASES = [
    {},
    {"stop": None},
    {"stop": "END"},
    {"stop": ["A", "B"]},
    {"stop": []},
    {"stop": ["A", "A"]},
    {"stop": ["\n\n"]},
    {"stop": 5},
    {"stop": [1, 2]},
    {"stop": {"a": 1}},
    {"stop": [""]},  # refused, layer 2
    {"stop": ["", "END"]},  # refused, layer 2
    {"stop": ""},  # refused, layer 2
    {"stop": ["A"], "include_stop_str_in_output": True},
    {"stop": ["A"], "include_stop_str_in_output": False},
    {"stop": ["A"], "include_stop_str_in_output": None},  # was wrong: accepted
    {"stop": ["A"], "include_stop_str_in_output": 1},  # was wrong: refused
    {"stop": ["A"], "include_stop_str_in_output": 0},  # was wrong: refused
    {"stop": ["A"], "include_stop_str_in_output": 2},
    {"stop": ["A"], "include_stop_str_in_output": "true"},  # was wrong: refused
    {"stop": ["A"], "include_stop_str_in_output": "False"},
    {"stop": ["A"], "include_stop_str_in_output": "yes"},
    {"stop": ["A"], "include_stop_str_in_output": "maybe"},
    {"stop": ["A"], "include_stop_str_in_output": []},
    {"include_stop_str_in_output": True},  # was wrong: refused
    {"include_stop_str_in_output": False},
]


@pytest.mark.parametrize("body", CASES, ids=lambda b: repr(b))
def test_the_router_accepts_exactly_what_vllm_accepts(body):
    """Accept/refuse must agree, and so must the normalised value.

    The error TYPES differ by construction -- pydantic raises ValidationError,
    the router raises InvalidParameter -- and that is not what is being
    compared. What matters is whether the request is served at all, and with
    which stop strings and which flag if it is.
    """
    v_ok, v_val = _vllm(body)
    o_ok, o_val = _ours(body)
    assert o_ok == v_ok, (
        f"vLLM {'accepts' if v_ok else 'refuses'} this and the router "
        f"{'accepts' if o_ok else 'refuses'} it"
    )
    if v_ok:
        assert o_val == v_val, "accepted by both, normalised differently"


def test_an_empty_stop_string_is_refused_by_vllms_second_layer():
    """Pins where it happens, because comparing the wrong layer misleads.

    `ChatCompletionRequest` accepts `stop: [""]`; `SamplingParams` refuses it.
    A test written against the model alone would conclude vLLM serves it and
    would push the router into serving an unrestricted completion.
    """
    req = ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "hi"}], stop=[""])
    assert req.stop == [""], "the type check passes"
    with pytest.raises(ValueError, match="empty string"):
        req.to_sampling_params(max_tokens=8, default_sampling_params={})
