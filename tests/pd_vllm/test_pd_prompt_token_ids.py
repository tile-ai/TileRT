"""The decode node must receive the prompt ids so repetition_penalty keeps its scope.

`repetition_penalty` is scoped over prompt UNION output in both HF and vLLM. On
a single machine TileRT matches that -- `seed_prompt_tokens()` fills a second
bitmap. Under PD the wire carried only `last_prompt_token` and the decode server
has no tokeniser, so the prompt half stayed empty and the knob silently degraded
to output-only scope, inconsistent with the same request served non-PD.

vLLM does not need a wire field for this: its PD proxy sends the SAME request to
both instances, so the decode instance runs `add_request` with the full
`prompt_token_ids` and the connector only carries KV block locations
(`nixl/pull_scheduler.py::get_num_new_matched_tokens` reads
`request.prompt_token_ids` locally). This change is the explicit equivalent of
what vLLM gets for free.

No GPU, no tilert, no vllm: `prefill_connector` cannot be imported here (it
pulls in vllm), so the send-side gate is tested through `wire`, where it lives.
The engine-side seeding (``seed_prompt_tokens`` after ``reset``) belongs to a
decode runtime with a penalty pre-pass; no public profile ships one today.

Run:
  CUDA_VISIBLE_DEVICES= python -m pytest tests/pd_vllm/test_pd_prompt_token_ids.py -v
"""

from __future__ import annotations

import pytest

from tilert.pd_vllm import wire
from tilert.pd_vllm.receive_server import ReceivedRequest

# --------------------------------------------------------------------------- #
# send-side gate: who pays for the payload
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "sampling",
    [
        None,
        {},
        {"temperature": 0.7},
        {"repetition_penalty": 1.0},  # the kernel's no-op value
        {"presence_penalty": 1.5},  # output-scoped, does not need the prompt
        {"repetition_penalty": "not-a-number"},
    ],
)
def test_no_prompt_ids_when_they_would_be_unused(sampling) -> None:
    """A request that cannot use the prompt half must not ship it.

    Mirrors vLLM's `needs_prompt_token_ids` gate. Shipping unconditionally would
    put a per-request `4 * prompt_len` payload on every plain request.
    """
    assert wire.wants_prompt_token_ids(sampling) is False


@pytest.mark.parametrize("rep", [1.05, 1.5, 2.0, 0.5])
def test_prompt_ids_shipped_when_repetition_penalty_is_live(rep) -> None:
    assert wire.wants_prompt_token_ids({"repetition_penalty": rep}) is True


def test_gate_ignores_presence_but_honours_a_paired_repetition() -> None:
    assert wire.wants_prompt_token_ids({"presence_penalty": 2.0, "repetition_penalty": 1.3}) is True


# --------------------------------------------------------------------------- #
# receive-side parse
# --------------------------------------------------------------------------- #


def test_received_request_defaults_to_no_prompt_ids() -> None:
    """Ranks 1-7 and older prefill builds send no such field.

    The field is optional on the wire (JSON, `req.get(...)`), so a prefill connector
    without this change must still work -- it just gets output-only scope.
    """
    r = ReceivedRequest(rid="x", seq_len=8, last_prompt_token=5, first_token_id=None, sampling=None)
    assert r.prompt_token_ids == []


def test_received_request_carries_prompt_ids() -> None:
    r = ReceivedRequest(
        rid="x",
        seq_len=8,
        last_prompt_token=5,
        first_token_id=None,
        sampling={"repetition_penalty": 1.3},
        prompt_token_ids=[1, 2, 3],
    )
    assert r.prompt_token_ids == [1, 2, 3]


# --------------------------------------------------------------------------- #
# Multi-rank arrival order.
#
# Found by a live PD run, not by the tests above: `ReceivedRequest` is built by
# whichever rank connects FIRST for a given rid, and only rank 0 carries the
# prompt ids. Ranks connect in arbitrary order (observed: 5, 1, 0, 7, 5, 1 across
# consecutive requests), so keying the ids off the creation path dropped them
# ~7/8 of the time -- a NON-DETERMINISTIC feature that unit tests over the
# dataclass and the gate could not see.
# --------------------------------------------------------------------------- #


def _absorb(messages: list[dict]) -> list[int]:
    """Replay `receive_server`'s per-connection absorb step for one rid.

    Mirrors the body of the `with self._lock` block: the first message builds the
    request, every message may contribute the prompt ids.
    """
    cur = None
    for req in messages:
        if cur is None or cur.rid != req["rid"]:
            cur = ReceivedRequest(
                rid=req["rid"],
                seq_len=req["seq_len"],
                last_prompt_token=req.get("last_prompt_token", 0),
                first_token_id=req.get("first_token_id"),
                sampling=req.get("sampling"),
                prompt_token_ids=list(req.get("prompt_token_ids") or []),
            )
        if not cur.prompt_token_ids and req.get("prompt_token_ids"):
            cur.prompt_token_ids = list(req["prompt_token_ids"])
    return cur.prompt_token_ids if cur else []


def _msgs(rank_order: list[int], ids: list[int]) -> list[dict]:
    """One request message per rank, in the given connection order.

    Only rank 0 carries the ids -- what `prefill_connector._send` does.
    """
    return [
        {
            "rid": "r1",
            "rank": r,
            "seq_len": 8,
            "last_prompt_token": 5,
            "sampling": {"repetition_penalty": 1.5},
            **({"prompt_token_ids": ids} if r == 0 else {}),
        }
        for r in rank_order
    ]


IDS = [11, 22, 33]


@pytest.mark.parametrize(
    "first",
    [0, 1, 2, 3, 4, 5, 6, 7],
    ids=[f"rank{r}_first" for r in range(8)],
)
def test_prompt_ids_survive_whatever_rank_connects_first(first: int) -> None:
    """The regression this file exists for, second edition.

    Before the fix only `rank0_first` passed; the other seven silently produced
    an empty prompt bitmap and repetition degraded to output-only scope.
    """
    order = [first] + [r for r in range(8) if r != first]
    assert _absorb(_msgs(order, IDS)) == IDS, f"lost the ids when rank {first} led"


def test_absorb_is_idempotent_and_does_not_grow() -> None:
    """Rank 0 appearing once must not be double-counted, and no rank may clear it."""
    order = [3, 0, 1, 2, 4, 5, 6, 7]
    assert _absorb(_msgs(order, IDS)) == IDS
    # ranks after rank 0 send no ids -- they must not reset the field
    assert _absorb(_msgs(order + [4, 5], IDS)) == IDS


def test_no_ids_anywhere_stays_empty() -> None:
    """A penalty-free request (or an old prefill connector) leaves the bitmap clear."""
    msgs = [{"rid": "r1", "rank": r, "seq_len": 8} for r in range(8)]
    assert _absorb(msgs) == []
