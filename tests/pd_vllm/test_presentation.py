"""The reply's shape, decided once and rendered twice, without HTTP.

`presentation` is what the two response paths used to spell out for themselves.
Everything here was previously only reachable by driving a real router over a
socket, which is why the ordering rules below were each found by a client rather
than by a test.

  CUDA_VISIBLE_DEVICES= python -m pytest tests/pd_vllm/test_presentation.py -v
"""

from __future__ import annotations

import json

import pytest

from tilert.pd_vllm.presentation import (
    SseWriter,
    collect,
    finish_reason,
    sse_chunk,
    sse_delta,
)
from tilert.pd_vllm.reply import CONTENT, REASONING, TOOL_CALL, Emission

CHUNK = {"chunk_id": "cmpl-1", "model": "m", "created": 7}


def _chunk(delta, **kw):
    return sse_chunk(delta, **CHUNK, **kw)


def _data(frame):
    assert frame.startswith("data: ") and frame.endswith("\n\n")
    body = frame[6:-2]
    return body if body == "[DONE]" else json.loads(body)


class _Stream:
    """The three facts presentation reads off a ReplyStream."""

    def __init__(self, held=(), stop_reason=None, tokens=3):
        self._held = list(held)
        self.stop_reason = stop_reason
        self.completion_tokens = tokens

    def finish(self):
        out, self._held = self._held, []  # idempotent, as the real one is
        return out

    def finish_reason(self, from_node):
        return "stop" if self.stop_reason is not None else from_node


# --------------------------------------------------------------------------- #
# collect: channels, and where logprob entries go
# --------------------------------------------------------------------------- #
def test_entries_follow_content_and_nothing_else():
    """#22: `logprobs` covers `message.content`.

    Reasoning text is text, not a channel entries can be attributed to.
    """
    got = collect(
        [
            Emission(REASONING, "thinking"),
            Emission(CONTENT, "hi", [{"token": "hi"}]),
            Emission(CONTENT, " there", [{"token": " there"}]),
        ]
    )
    assert got.content == "hi there"
    assert got.reasoning == "thinking"
    assert [e["token"] for e in got.entries] == ["hi", " there"]
    assert got.tool_calls == []


# --------------------------------------------------------------------------- #
# finish_reason: one precedence for both paths
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "saw_tool,stop,from_node,want,why",
    [
        (
            True,
            "END",
            "length",
            "tool_calls",
            "tool_calls outranks a stop, as it does in vLLM's serving path",
        ),
        (False, "END", "length", "stop", "a stop outranks the node: it cannot see text"),
        (False, None, "length", "length", "otherwise the node's own reason"),
        (True, None, "length", "tool_calls", "tool_calls outranks that too"),
    ],
)
def test_the_finish_reason_precedence(saw_tool, stop, from_node, want, why):
    assert (
        finish_reason(saw_tool=saw_tool, from_node=from_node, stream=_Stream(stop_reason=stop))
        == want
    ), why


# --------------------------------------------------------------------------- #
# SseWriter: the order the frames have to come out in
# --------------------------------------------------------------------------- #
def test_the_role_chunk_goes_out_once_and_only_before_content():
    """#34: a request that emits nothing sends no role either."""
    out = SseWriter(_Stream(), _chunk)
    frames = list(out.frames([Emission(CONTENT, "a"), Emission(CONTENT, "b")]))
    assert [_data(f)["choices"][0]["delta"] for f in frames] == [
        {"role": "assistant"},
        {"content": "a"},
        {"content": "b"},
    ]


def test_a_reply_that_emits_nothing_sends_no_role():
    out = SseWriter(_Stream(), _chunk)
    assert list(out.frames([])) == []
    assert not out.role_sent


def test_an_emission_with_neither_text_nor_entries_makes_no_frame():
    """The parser consuming a tag produces one; a chunk for it is noise."""
    out = SseWriter(_Stream(), _chunk)
    assert list(out.frames([Emission(CONTENT, "")])) == []


def test_an_empty_emission_carrying_entries_is_still_a_frame():
    """A token whose text a stop removed still has a logprob to report."""
    out = SseWriter(_Stream(), _chunk)
    frames = list(out.frames([Emission(CONTENT, "", [{"token": "x"}])]))
    assert len(frames) == 2, "the role chunk, then the entries"
    assert _data(frames[1])["choices"][0]["logprobs"]["content"]


def test_a_tool_call_is_remembered_for_the_finish_reason():
    out = SseWriter(_Stream(), _chunk)
    list(
        out.frames(
            [
                Emission(
                    TOOL_CALL,
                    "",
                    tool_call={"index": 0, "id": "c1", "name": "f", "arguments": "{}"},
                )
            ]
        )
    )
    assert out.saw_tool, "the finish reason needs it after the loop"


def test_held_text_is_flushed_before_anything_else():
    """With a stop the matcher may hold len(stop)-1 characters the reply earned;
    a marker emitted first lands as `prefix[decode error]suffix`.
    """
    out = SseWriter(_Stream(held=[Emission(CONTENT, "tail")]), _chunk)
    frames = [_data(f) for f in out.flush_held()]
    assert [f["choices"][0]["delta"] for f in frames] == [
        {"role": "assistant"},
        {"content": "tail"},
    ]


def test_flushing_twice_is_safe():
    """Three exits need the flush and two of them return early."""
    out = SseWriter(_Stream(held=[Emission(CONTENT, "tail")]), _chunk)
    first = list(out.flush_held())
    assert list(out.flush_held()) == [], "nothing is repeated"
    assert len(first) == 2


def test_a_stream_that_cannot_be_completed_still_ends_properly():
    """The 200 is spent, so the only honest ending is: the reply's own text, a
    finish_reason, a typed error event, [DONE].
    """
    out = SseWriter(_Stream(held=[Emission(CONTENT, "earned")]), _chunk)
    frames = [
        _data(f) for f in out.fail_closed({"error": "boom", "error_type": "decode_truncated"})
    ]
    assert frames[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert frames[1]["choices"][0]["delta"] == {"content": "earned"}
    assert frames[2]["choices"][0]["finish_reason"] == "stop"
    assert frames[3]["error"]["error_type"] == "decode_truncated"
    assert frames[4] == "[DONE]"


# --------------------------------------------------------------------------- #
# frames
# --------------------------------------------------------------------------- #
def test_stop_reason_rides_only_the_closing_chunk():
    """Where a client reads it, and where vLLM puts it."""
    assert "stop_reason" not in _data(_chunk({"content": "x"}))["choices"][0]
    closing = _data(_chunk({}, finish="stop", stop_reason="END"))
    assert closing["choices"][0]["stop_reason"] == "END"


def test_a_channel_becomes_its_own_delta_field():
    assert sse_delta(Emission(CONTENT, "a")) == {"content": "a"}
    assert sse_delta(Emission(REASONING, "b")) == {"reasoning_content": "b"}
    tool = sse_delta(
        Emission(TOOL_CALL, "", tool_call={"index": 0, "id": "c1", "name": "f", "arguments": "{}"})
    )
    assert tool["tool_calls"][0]["function"]["name"] == "f"
    assert tool["tool_calls"][0]["type"] == "function"
