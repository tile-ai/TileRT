from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from tilert.pd_vllm.logprobs import LogprobsRequest, build_logprobs
from tilert.pd_vllm.oai_parser import IncrementalDetok
from tilert.pd_vllm.stop_strings import StopWindow

__all__ = ["CONTENT", "REASONING", "TOOL_CALL", "Emission", "ReplyStream", "as_logprobs"]
CONTENT = "content"
REASONING = "reasoning"
TOOL_CALL = "tool_call"


def as_logprobs(entries: list[dict]) -> dict:
    return {"content": entries, "refusal": None}


@dataclass(frozen=True)
class Emission:
    channel: str
    text: str = ""
    logprobs: list[dict] = field(default_factory=list)
    tool_call: dict | None = None


@dataclass
class _Entry:
    token_id: int
    logprob: float | None
    candidates: list[tuple[int, float]]
    ends_at: int


class ReplyStream:

    def __init__(
        self,
        tokenizer,
        *,
        stop: Iterable[str] = (),
        include_stop_in_output: bool = False,
        parser_session: Any | None = None,
        logprobs_req: LogprobsRequest | None = None,
        first_token_logprob: tuple[float | None, list] | None = None,
    ) -> None:
        stop = list(stop)
        if stop and parser_session is not None and (logprobs_req is not None):
            raise ValueError(
                "stop with both an output parser and logprobs is not attributable; the request gate must refuse it"
            )
        self._session = parser_session
        self._logprobs_req = logprobs_req
        self._first_lp = first_token_logprob
        self._detok = IncrementalDetok(tokenizer, skip_special_tokens=parser_session is None)
        self._decode_one: Callable[[int], str] = lambda t: tokenizer.decode(
            [t], skip_special_tokens=False
        )
        self._window = StopWindow(stop, include_stop_in_output)
        self._ids: list[int] = []
        self._chars = 0
        self._due: list[_Entry] = []
        self._held: list[dict] = []
        self._finished = False

    @property
    def token_ids(self) -> list[int]:
        return list(self._ids)

    @property
    def completion_tokens(self) -> int:
        return len(self._ids)

    @property
    def stop_reason(self) -> str | None:
        return self._window.stopped

    def finish_reason(self, from_decode: str) -> str:
        return "stop" if self._window.stopped is not None else from_decode

    def push(
        self,
        token_ids: Iterable[int],
        logprobs: list[float | None] | None = None,
        candidates: list[list[tuple[int, float]]] | None = None,
    ) -> list[Emission]:
        out: list[Emission] = []
        for i, tid in enumerate(token_ids):
            if self._window.stopped is not None:
                break
            self._ids.append(tid)
            delta = self._detok.push([tid])
            self._chars += len(delta)
            self._queue(tid, logprobs, candidates, i, bool(delta))
            self._window.push(delta)
            out += self._route(self._window.take(limit=self._cap()))
        return out

    def finish(self) -> list[Emission]:
        if self._finished:
            return []
        self._finished = True
        tail = self._detok.finish()
        if tail:
            self._chars += len(tail)
            for pending in self._due:
                if pending.ends_at < 0:
                    pending.ends_at = self._chars
            self._window.push(tail)
        out = self._route(self._window.take(final=True))
        if self._session is not None:
            out += self._events(self._session.finish(), self._collect())
            return out
        leftover = self._collect()
        if leftover:
            out.append(Emission(CONTENT, "", leftover))
        return out

    def _queue(self, tid, logprobs, candidates, i, produced_text) -> None:
        if self._logprobs_req is None:
            return
        lp = logprobs[i] if logprobs and i < len(logprobs) else None
        cands = list(candidates[i]) if candidates and i < len(candidates) else []
        if len(self._ids) == 1 and lp is None and (self._first_lp is not None):
            lp, cands = (self._first_lp[0], list(self._first_lp[1] or []))
        if produced_text:
            for pending in self._due:
                if pending.ends_at < 0:
                    pending.ends_at = self._chars
        self._due.append(_Entry(tid, lp, cands, ends_at=self._chars if produced_text else -1))

    def _cap(self) -> int | None:
        if self._logprobs_req is None:
            return None
        end = self._chars - self._window.hold
        cap = self._window.visible
        for pending in self._due:
            if 0 <= pending.ends_at <= end:
                cap = pending.ends_at
        return cap

    def _route(self, text: str) -> list[Emission]:
        if self._session is None:
            entries = self._ready()
            if not text and (not entries):
                return []
            return [Emission(CONTENT, text, entries)]
        entries = self._held + self._ready()
        if not text:
            self._held = entries
            return []
        self._held = []
        return self._events(self._session.feed(text), entries)

    def _events(self, events: list[dict], entries: list[dict]) -> list[Emission]:
        out: list[Emission] = []
        for ev in events:
            kind = ev.get("kind")
            if kind == "tool":
                out.append(
                    Emission(
                        TOOL_CALL,
                        tool_call={
                            "index": ev["index"],
                            "id": ev["id"],
                            "name": ev["name"],
                            "arguments": ev["arguments"],
                        },
                    )
                )
            elif kind == "reasoning":
                out.append(Emission(REASONING, ev.get("text", "")))
            else:
                out.append(Emission(CONTENT, ev.get("text", ""), entries))
                entries = []
        return out

    def _ready(self) -> list[dict]:
        visible = self._window.visible
        stopped = self._window.stopped is not None
        n = 0
        while (
            n < len(self._due)
            and self._due[n].ends_at >= 0
            and (stopped or self._due[n].ends_at <= visible)
        ):
            n += 1
        return self._build(n)

    def _collect(self) -> list[dict]:
        return self._build(len(self._due))

    def _build(self, n: int) -> list[dict]:
        if not n:
            return []
        taken, self._due = (self._due[:n], self._due[n:])
        return build_logprobs(
            [e.token_id for e in taken],
            [e.logprob for e in taken],
            [[tuple(c) for c in e.candidates] for e in taken],
            self._logprobs_req,
            self._decode_one,
        )["content"]
