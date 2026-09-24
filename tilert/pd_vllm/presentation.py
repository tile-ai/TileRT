from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from tilert.pd_vllm.reply import CONTENT, REASONING, TOOL_CALL, as_logprobs

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
    content: str = ""
    reasoning: str = ""
    entries: list = field(default_factory=list)
    tool_calls: list = field(default_factory=list)


def collect(emissions) -> Collected:
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
    choice: dict = {
        "index": 0,
        "finish_reason": finish_reason(
            saw_tool=bool(got.tool_calls), from_node=from_node, stream=stream
        ),
        "logprobs": as_logprobs(got.entries) if logprobs_asked else None,
        "stop_reason": stream.stop_reason,
    }
    if is_chat:
        msg = {"role": "assistant", "content": got.content}
        if got.reasoning:
            msg["reasoning_content"] = got.reasoning
        if got.tool_calls:
            msg["tool_calls"] = _tool_calls_field(got.tool_calls)
        choice["message"] = msg
    else:
        choice["text"] = got.content
        choice["token_ids"] = token_ids
    return (choice, stream.completion_tokens)


def textless_choice(*, is_chat: bool, from_node: str, token_ids: list[int]) -> tuple[dict, int]:
    choice: dict = {"index": 0, "finish_reason": from_node, "logprobs": None, "stop_reason": None}
    if is_chat:
        choice["message"] = {"role": "assistant", "content": None}
    else:
        choice["text"] = None
        choice["token_ids"] = token_ids
    return (choice, len(token_ids))


def blocking_envelope(
    choice: dict, *, is_chat: bool, prefill: dict, created: int, usage: dict, timing: dict
) -> dict:
    return {
        "id": prefill["id"],
        "object": "chat.completion" if is_chat else "text_completion",
        "created": created,
        "model": prefill.get("model"),
        "choices": [choice],
        "usage": usage,
        "pd_timing_ms": timing,
    }


def sse_delta(e) -> dict:
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
    choice: dict = {"index": 0, "delta": delta, "finish_reason": finish}
    if logprobs is not None:
        choice["logprobs"] = logprobs
    if finish is not None:
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

    def __init__(self, stream, chunk):
        self._stream = stream
        self._chunk = chunk
        self.role_sent = False
        self.saw_tool = False

    def role_once(self) -> str:
        self.role_sent = True
        return self._chunk({"role": "assistant"})

    def emit(self, e) -> str | None:
        if e.channel == TOOL_CALL:
            self.saw_tool = True
        elif not e.text and (not e.logprobs):
            return None
        return self._chunk(sse_delta(e), logprobs=as_logprobs(e.logprobs) if e.logprobs else None)

    def frames(self, emissions):
        for e in emissions:
            frame = self.emit(e)
            if frame is None:
                continue
            if not self.role_sent:
                yield self.role_once()
            yield frame

    def flush_held(self):
        yield from self.frames(self._stream.finish())
        if not self.role_sent:
            yield self.role_once()

    def fail_closed(self, payload: dict):
        yield from self.flush_held()
        yield self._chunk({}, finish="stop")
        yield _frame({"error": payload})
        yield "data: [DONE]\n\n"
