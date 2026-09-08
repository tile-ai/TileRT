"""One reply, in two presentations.

Both response paths consume the same :class:`reply.Emission` values, so they
agree about text, channels, logprob entries and the token count by construction.
What they did NOT share was the policy applied to those emissions -- the
``tool_calls`` over ``stop`` precedence, ``stop_reason``, where ``usage`` comes
from, and how a channel becomes a field. Each path spelled that out for itself
and they agreed because a test drove one request through both and compared the
whole reply.

Here the policy is decided once (:func:`finish_reason`, :func:`collect`) and
rendered twice: :func:`blocking_envelope` builds the single JSON body,
:func:`sse_chunk` the frames. A difference between the two presentations is now
a difference in this file.

Free of HTTP and asyncio: dicts and strings out; the caller wraps them in
whatever response class it uses.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from tilert.pd_vllm.reply import (
    CONTENT,
    REASONING,
    TOOL_CALL,
    as_logprobs,
)

__all__ = [
    "Collected",
    "SseWriter",
    "blocking_envelope",
    "collect",
    "finish_reason",
    "sse_chunk",
    "sse_delta",
    "usage_chunk",
]


@dataclass
class Collected:
    """Emissions, gathered per channel, as the blocking body needs them."""

    content: str = ""
    reasoning: str = ""
    entries: list = field(default_factory=list)
    tool_calls: list = field(default_factory=list)


def collect(emissions) -> Collected:
    """Split emissions by channel.

    Logprob entries follow content, which is what #22 means by ``logprobs`` covering
    ``message.content``.
    """
    out = Collected()
    for e in emissions:
        if e.channel == CONTENT:
            out.content += e.text
            out.entries += e.logprobs
        elif e.channel == REASONING:
            out.reasoning += e.text
        else:
            out.tool_calls.append(e.tool_call)
    return out


def finish_reason(*, saw_tool: bool, from_node: str, stream) -> str:
    """What ended the reply, in vLLM's precedence at this layer.

    ``tool_calls`` outranks a matched stop, as it does in vLLM's own serving
    path: it is set whenever the tool parser extracted calls, ahead of the
    engine's reason. A client that needs to know the reply was cut short reads
    ``stop_reason`` instead. A stop then outranks what the NODE said, since the
    node cannot see text and cannot reach that conclusion.
    """
    if saw_tool:
        return "tool_calls"
    return stream.finish_reason(from_node)


def _tool_calls_field(tool_calls: list) -> list:
    return [
        {
            "index": c["index"],
            "id": c["id"],
            "type": "function",
            "function": {"name": c["name"], "arguments": c["arguments"]},
        }
        for c in tool_calls
    ]


def blocking_choice(
    got: Collected,
    *,
    is_chat: bool,
    stream,
    from_node: str,
    logprobs_asked: bool,
    token_ids: list[int],
) -> tuple[dict, int]:
    """The one ``choices[0]`` of a non-streamed reply, and its token count."""
    choice: dict = {
        "index": 0,
        "finish_reason": finish_reason(
            saw_tool=bool(got.tool_calls), from_node=from_node, stream=stream
        ),
        "logprobs": as_logprobs(got.entries) if logprobs_asked else None,
        # Non-null when a stop string ended the reply. Names which one, which a
        # client cannot recover from the text when the string was cut out of it.
        "stop_reason": stream.stop_reason,
    }
    if is_chat:
        msg: dict[str, Any] = {"role": "assistant", "content": got.content}
        if got.reasoning:
            msg["reasoning_content"] = got.reasoning
        if got.tool_calls:
            msg["tool_calls"] = _tool_calls_field(got.tool_calls)
        choice["message"] = msg
    else:
        choice["text"] = got.content
        # Every id that ran, including those whose text a stop removed:
        # `token_ids` reports what ran and `text` what came back. vLLM's
        # `CompletionOutput.token_ids` is untruncated too.
        choice["token_ids"] = token_ids
    return choice, stream.completion_tokens


def textless_choice(*, is_chat: bool, from_node: str, token_ids: list[int]) -> tuple[dict, int]:
    """``--parser none`` with no ``--model-path``: no tokenizer, so no text.

    No stop matching, no channels and no logprobs are possible; token ids are
    all the reply can carry, and only ``/v1/completions`` exposes them.
    """
    choice: dict = {"index": 0, "finish_reason": from_node, "logprobs": None, "stop_reason": None}
    if is_chat:
        choice["message"] = {"role": "assistant", "content": None}
    else:
        choice["text"] = None
        choice["token_ids"] = token_ids
    return choice, len(token_ids)


def blocking_envelope(
    choice: dict, *, is_chat: bool, prefill: dict, created: int, usage: dict, timing: dict
) -> dict:
    """The whole non-streamed body."""
    return {
        "id": prefill["id"],
        "object": "chat.completion" if is_chat else "text_completion",
        "created": created,
        "model": prefill.get("model"),
        "choices": [choice],
        "usage": usage,
        # Empty apart from `prefill` when a stop string ended the reply: the node
        # reports its timings on the `done` line, which we stopped reading
        # before.
        "pd_timing_ms": timing,
    }


def sse_delta(e) -> dict:
    """One emission as an OpenAI streaming delta."""
    if e.channel == REASONING:
        return {"reasoning_content": e.text}
    if e.channel == CONTENT:
        return {"content": e.text}
    return {"tool_calls": _tool_calls_field([e.tool_call])}


def sse_chunk(
    delta: dict,
    *,
    chunk_id: str,
    model: Any,
    created: int,
    finish: str | None = None,
    usage: dict | None = None,
    logprobs: dict | None = None,
    stop_reason: str | None = None,
) -> str:
    """One ``chat.completion.chunk`` frame."""
    choice: dict = {"index": 0, "delta": delta, "finish_reason": finish}
    if logprobs is not None:
        choice["logprobs"] = logprobs
    if finish is not None:
        # Only on the chunk that closes the choice, which is where a client
        # reads it, and where vLLM puts it.
        choice["stop_reason"] = stop_reason
    payload = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [choice],
    }
    if usage is not None:
        payload["usage"] = usage
    return _frame(payload)


def usage_chunk(usage: dict, *, chunk_id: str, model: Any, created: int) -> str:
    """The trailing usage-only chunk, the shape vLLM and OpenAI emit.

    ``choices: []`` on a chunk of its own, not on the one that closes the
    choice: clients read usage off the final chunk and stop at the first one
    bearing a finish_reason.
    """
    return _frame(
        {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [],
            "usage": usage,
        }
    )


def _frame(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


class SseWriter:
    """The order an SSE reply has to come out in.

    Every rule here is one a client noticed when it was missing:

    * the opening ``delta.role`` chunk goes out ONCE, and only if something
      follows it (#34: a request that emits nothing sends no role either);
    * an emission carrying neither text nor entries produces no chunk -- the
      parser consuming a tag, say -- because a chunk for it is noise;
    * whatever the reply stream still holds is flushed before any marker, or the
      marker lands before text the reply earned: ``prefix[decode error]suffix``;
    * a stream that cannot be completed correctly still ends properly. The 200 is
      spent, so the only honest ending is the reply's own text, a finish_reason, a
      typed error event, ``[DONE]`` -- not a truncated stream, and not more
      content.

    ``saw_tool`` is a fact about the reply the caller needs afterwards: it
    outranks a stop in the finish reason.

    The methods are plain generators, not async ones: ``yield from`` is not
    allowed inside an async generator, so the caller iterates them.
    """

    def __init__(self, stream, chunk):
        self._stream = stream  # the ReplyStream
        self._chunk = chunk  # a bound sse_chunk
        self.role_sent = False
        self.saw_tool = False

    def role_once(self) -> str:
        self.role_sent = True
        return self._chunk({"role": "assistant"})

    def emit(self, e) -> str | None:
        """One emission as a frame, or None if it carries nothing."""
        if e.channel == TOOL_CALL:
            self.saw_tool = True
        elif not e.text and not e.logprobs:
            return None
        return self._chunk(sse_delta(e), logprobs=(as_logprobs(e.logprobs) if e.logprobs else None))

    def frames(self, emissions):
        """Emissions as frames, with the role chunk first if it is still owed."""
        for e in emissions:
            frame = self.emit(e)
            if frame is None:
                continue
            if not self.role_sent:
                yield self.role_once()
            yield frame

    def flush_held(self):
        """What the reply stream still holds, then the role if still owed.

        Safe to call twice: ``finish()`` is idempotent and ``role_once`` sets its
        own flag. With a stop the matcher may be holding up to ``len(stop) - 1``
        characters the reply earned.
        """
        yield from self.frames(self._stream.finish())
        if not self.role_sent:
            yield self.role_once()

    def fail_closed(self, payload: dict):
        """End a stream that cannot be completed correctly."""
        yield from self.flush_held()
        yield self._chunk({}, finish="stop")
        yield _frame({"error": payload})
        yield "data: [DONE]\n\n"
