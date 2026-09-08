"""The chat ``logprobs`` contract: what is accepted, and what comes back.

Ranges and interdependencies come from the OpenAI chat-completions reference as
narrowed to the stricter vendor reading:

* ``logprobs``: boolean, default false.
* ``top_logprobs``: integer, default 0, ``[0, 5]``, only valid with
  ``logprobs: true``.
* logprobs cover ``message.content``; a reasoning segment carries none.
* ``-9999.0`` is the documented stand-in when a real log probability is absent.

No GPU, no tilert, no vllm: request parsing and response assembly are pure.

Run:
  CUDA_VISIBLE_DEVICES= python -m pytest tests/pd_vllm/test_logprobs.py -v
"""

from __future__ import annotations

import pytest

from tilert.pd_vllm.logprobs import (
    LOGPROB_UNAVAILABLE,
    TOP_LOGPROBS_MAX,
    LogprobsRequest,
    LogprobsUnsupported,
    build_logprobs,
    resolve_logprobs_request,
)

# A toy vocabulary; 300 decodes to a multi-byte character so `bytes` is
# exercised on something other than ASCII.
VOCAB = {1: "Hello", 2: " world", 3: "!", 300: "世", 400: "</think>"}


def decode_one(tid: int) -> str:
    return VOCAB.get(tid, f"<{tid}>")


# --------------------------------------------------------------------------- #
# request parsing
# --------------------------------------------------------------------------- #


def test_absent_means_no_logprobs() -> None:
    assert resolve_logprobs_request({"model": "m"}) is None


def test_false_means_no_logprobs() -> None:
    assert resolve_logprobs_request({"logprobs": False}) is None


def test_true_alone_defaults_top_n_to_zero() -> None:
    """`logprobs: true` on its own returns the chosen token's logprob only."""
    assert resolve_logprobs_request({"logprobs": True}) == LogprobsRequest(0)


@pytest.mark.parametrize("n", [0, 1, 3, 5])
def test_accepted_top_logprobs_range(n) -> None:
    req = resolve_logprobs_request({"logprobs": True, "top_logprobs": n})
    assert req == LogprobsRequest(n)


@pytest.mark.parametrize("n", [6, 20, 256, -1])
def test_out_of_range_is_rejected_not_clamped(n) -> None:
    """A client asking for more than we serve must be told, not quietly cut."""
    with pytest.raises(LogprobsUnsupported) as e:
        resolve_logprobs_request({"logprobs": True, "top_logprobs": n})
    assert e.value.http_status == 400
    assert str(TOP_LOGPROBS_MAX) in str(e.value)


def test_top_logprobs_without_logprobs_is_rejected() -> None:
    with pytest.raises(LogprobsUnsupported) as e:
        resolve_logprobs_request({"top_logprobs": 3})
    assert e.value.http_status == 400
    assert "logprobs must be set to true" in str(e.value)


def test_zero_top_logprobs_without_logprobs_is_not_an_error() -> None:
    """0 is the default, so its presence alone does not request anything."""
    assert resolve_logprobs_request({"top_logprobs": 0}) is None


def test_top_logprobs_true_is_a_type_error_not_one() -> None:
    """bool is an int subclass; `top_logprobs: true` must not mean 1."""
    with pytest.raises(LogprobsUnsupported):
        resolve_logprobs_request({"logprobs": True, "top_logprobs": True})


@pytest.mark.parametrize("bad", ["3", 3.5, [3]])
def test_non_integer_top_logprobs_is_rejected(bad) -> None:
    with pytest.raises(LogprobsUnsupported):
        resolve_logprobs_request({"logprobs": True, "top_logprobs": bad})


def test_non_boolean_logprobs_is_rejected() -> None:
    with pytest.raises(LogprobsUnsupported):
        resolve_logprobs_request({"logprobs": "yes"})


def test_error_payload_shape_matches_the_grammar_errors() -> None:
    """The router returns these the same way it returns grammar errors."""
    try:
        resolve_logprobs_request({"logprobs": True, "top_logprobs": 9})
    except LogprobsUnsupported as e:
        assert e.to_payload()["error_type"] == "invalid_logprobs"
        assert "error" in e.to_payload()


# --------------------------------------------------------------------------- #
# response assembly
# --------------------------------------------------------------------------- #


def test_shape_is_content_plus_nullable_refusal() -> None:
    out = build_logprobs([1], [-0.5], [[(1, -0.5)]], LogprobsRequest(1), decode_one)
    assert set(out) == {"content", "refusal"}
    assert out["refusal"] is None
    assert set(out["content"][0]) == {"token", "logprob", "bytes", "top_logprobs"}


def test_one_entry_per_content_token() -> None:
    out = build_logprobs([1, 2, 3], [-0.1, -0.2, -0.3], None, LogprobsRequest(0), decode_one)
    assert [c["token"] for c in out["content"]] == ["Hello", " world", "!"]
    assert [c["logprob"] for c in out["content"]] == [-0.1, -0.2, -0.3]


def test_top_n_zero_gives_an_empty_candidate_list() -> None:
    out = build_logprobs([1], [-0.1], [[(1, -0.1), (2, -2.0)]], LogprobsRequest(0), decode_one)
    assert out["content"][0]["top_logprobs"] == []


def test_candidates_are_truncated_to_top_n() -> None:
    """A decode node may return more than asked; the response must not."""
    alts = [(1, -0.1), (2, -2.0), (3, -3.0), (300, -4.0)]
    out = build_logprobs([1], [-0.1], [alts], LogprobsRequest(2), decode_one)
    assert len(out["content"][0]["top_logprobs"]) == 2
    assert [a["token"] for a in out["content"][0]["top_logprobs"]] == ["Hello", " world"]


def test_bytes_is_utf8_of_the_token() -> None:
    out = build_logprobs([300], [-1.0], None, LogprobsRequest(0), decode_one)
    assert out["content"][0]["bytes"] == list("世".encode())
    assert len(out["content"][0]["bytes"]) == 3


def test_missing_logprob_uses_the_documented_sentinel() -> None:
    out = build_logprobs([1], [None], None, LogprobsRequest(0), decode_one)
    assert out["content"][0]["logprob"] == LOGPROB_UNAVAILABLE


def test_negative_infinity_becomes_the_sentinel() -> None:
    """JSON cannot carry -inf; it must not reach the client as Infinity."""
    out = build_logprobs([1], [float("-inf")], None, LogprobsRequest(0), decode_one)
    assert out["content"][0]["logprob"] == LOGPROB_UNAVAILABLE


def test_sentinel_also_applies_inside_candidates() -> None:
    out = build_logprobs([1], [-0.1], [[(2, float("-inf"))]], LogprobsRequest(1), decode_one)
    assert out["content"][0]["top_logprobs"][0]["logprob"] == LOGPROB_UNAVAILABLE


def test_response_is_json_serialisable() -> None:
    """The whole point of the sentinel: no inf/nan reaches json.dumps."""
    import json

    out = build_logprobs(
        [1, 300],
        [float("-inf"), -0.2],
        [[(1, float("-inf"))], [(300, -0.2)]],
        LogprobsRequest(1),
        decode_one,
    )
    assert "Infinity" not in json.dumps(out, allow_nan=False)


@pytest.mark.parametrize("n_lp,n_top", [(2, 1), (1, 2)])
def test_length_mismatch_is_caught(n_lp, n_top) -> None:
    """A decode node returning the wrong count is a bug, not a client error."""
    with pytest.raises(ValueError):
        build_logprobs([1], [-0.1] * n_lp, [[(1, -0.1)]] * n_top, LogprobsRequest(1), decode_one)


# --------------------------------------------------------------------------- #
# reasoning segment carries no logprobs
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# token 1: sourced from the prefill response
#
# The decode node echoes first_token_id without sampling it, so it sends null
# for that position. The distribution that produced the token exists only at the
# prompt's last position, which the vLLM prefill instance evaluated -- and the
# router already reads that same entry to recover the token id.
# --------------------------------------------------------------------------- #


def _prefill_resp(logprob=-0.25, cands=((7, -0.25), (8, -1.5), (9, -2.5))):
    """A vLLM prefill reply shaped as --return-tokens-as-token-ids produces."""
    return {
        "choices": [
            {
                "logprobs": {
                    "content": [
                        {
                            "token": "token_id:7",
                            "logprob": logprob,
                            "top_logprobs": [
                                {"token": f"token_id:{i}", "logprob": lp} for i, lp in cands
                            ],
                        }
                    ]
                }
            }
        ]
    }


def test_prefill_entry_is_parsed_with_its_candidates() -> None:
    from tilert.pd_vllm.pd_router import first_token_logprob_from_prefill

    lp, cands = first_token_logprob_from_prefill(_prefill_resp(), top_n=3)
    assert lp == -0.25
    assert cands == [(7, -0.25), (8, -1.5), (9, -2.5)]


def test_prefill_candidates_are_capped_at_the_requested_count() -> None:
    from tilert.pd_vllm.pd_router import first_token_logprob_from_prefill

    _, cands = first_token_logprob_from_prefill(_prefill_resp(), top_n=1)
    assert cands == [(7, -0.25)]


def test_prefill_entry_without_logprobs_is_absent_not_fatal() -> None:
    """A prefill reply carrying no usable entry must not fail the request.

    Every other position is still correct, so the documented sentinel for that
    one entry beats a 500 for the whole completion.
    """
    from tilert.pd_vllm.pd_router import first_token_logprob_from_prefill

    assert first_token_logprob_from_prefill({}, 3) == (None, [])
    assert first_token_logprob_from_prefill({"choices": [{"logprobs": {"content": []}}]}, 3) == (
        None,
        [],
    )


def test_prefill_candidates_that_are_not_token_ids_are_dropped() -> None:
    """Without --return-tokens-as-token-ids the alternatives are plain text.

    There is no id to report then, so the entry is dropped rather than guessed
    at -- the token's own logprob still comes through.
    """
    from tilert.pd_vllm.pd_router import first_token_logprob_from_prefill

    resp = _prefill_resp()
    resp["choices"][0]["logprobs"]["content"][0]["top_logprobs"][1]["token"] = "he"
    lp, cands = first_token_logprob_from_prefill(resp, top_n=3)
    assert lp == -0.25
    assert cands == [(7, -0.25), (9, -2.5)]


def test_prefill_body_raises_top_logprobs_to_the_requested_count() -> None:
    """Token 1's candidate row can only come from the prefill instance."""
    from tilert.pd_vllm.pd_router import DecodeNode, build_prefill_body

    node = DecodeNode(host="h", ctrl_port=1, http_port=2)
    body = {"messages": [], "logprobs": True, "top_logprobs": 4}
    out = build_prefill_body("/v1/chat/completions", body, node, LogprobsRequest(4))
    assert out["logprobs"] is True
    assert out["top_logprobs"] == 4


def test_prefill_body_still_asks_for_one_entry_without_logprobs() -> None:
    """The id extraction needs an entry even when the client asked for none."""
    from tilert.pd_vllm.pd_router import DecodeNode, build_prefill_body

    node = DecodeNode(host="h", ctrl_port=1, http_port=2)
    out = build_prefill_body("/v1/chat/completions", {"messages": []}, node)
    assert out["top_logprobs"] == 1


# --------------------------------------------------------------------------- #
# temperature bound
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("temp", [1e-5, 0.05, 0.1, 0.19])
def test_middle_band_temperature_is_refused(temp) -> None:
    """Between greedy and the bound there is no path that can serve it.

    The temperature selects the distribution being sampled, so it cannot be
    substituted the way greedy's can, and the top-p path's raw denominator is
    imprecise there.
    """
    with pytest.raises(LogprobsUnsupported) as e:
        resolve_logprobs_request({"logprobs": True, "temperature": temp})
    assert "temperature" in str(e.value)


@pytest.mark.parametrize("temp", [0.0, 1e-9, 9e-6])
def test_greedy_is_served_without_candidates(temp) -> None:
    """The top-1 kernel exports the chosen token's value, so this is servable."""
    req = resolve_logprobs_request({"logprobs": True, "temperature": temp})
    assert req == LogprobsRequest(0)


@pytest.mark.parametrize("temp", [0.0, 1e-9])
@pytest.mark.parametrize("n", [1, 3, 5])
def test_greedy_with_candidates_is_served(temp, n) -> None:
    """The greedy kernel exports a candidate row too, so the full range serves.

    TOP_LOGPROBS_MAX is that row's width, so nothing this accepts can ask for
    more entries than the kernel writes.
    """
    assert resolve_logprobs_request(
        {"logprobs": True, "temperature": temp, "top_logprobs": n}
    ) == LogprobsRequest(n)


@pytest.mark.parametrize("temp", [0.0, 1e-9])
def test_greedy_beyond_the_row_width_is_refused(temp) -> None:
    with pytest.raises(LogprobsUnsupported) as e:
        resolve_logprobs_request(
            {"logprobs": True, "temperature": temp, "top_logprobs": TOP_LOGPROBS_MAX + 1}
        )
    assert "top_logprobs" in str(e.value)


@pytest.mark.parametrize("temp", [0.2, 0.6, 1.0, 2.0])
def test_supported_temperature_is_accepted(temp) -> None:
    req = resolve_logprobs_request({"logprobs": True, "temperature": temp})
    assert req is not None


def test_absent_temperature_is_accepted() -> None:
    """vLLM's default is 1.0, which is inside the bound."""
    assert resolve_logprobs_request({"logprobs": True}) is not None


def test_low_temperature_without_logprobs_is_not_an_error() -> None:
    """The bound constrains the logprobs export, not sampling."""
    assert resolve_logprobs_request({"temperature": 0.0}) is None
