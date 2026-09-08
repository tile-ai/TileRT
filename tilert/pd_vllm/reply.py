"""Token ids -> emissions, for both response channels.

Three transformations stand between the node's ids and either reply: detokenise
incrementally (``IncrementalDetok``), match stop strings and hold back what could
still become one (``StopWindow``), route text to channels (the output parser).
Both channels consume the same emissions, which is what makes them agree.

Logprob entries ride along, and are the one place the transformations are not
independent: an entry is keyed by TOKEN POSITION, the stop cut by CHARACTER
OFFSET. What this module does about that is one-sided: it keeps an emission from
ENDING inside a token whose entry is still due, so text and the entry describing
it go out together. It does not attribute an entry to a CHANNEL --
``refuse_unattributable_logprobs``
refuses the only shape where the join has no answer, and the rest needs no offset
arithmetic:

    no parser              multi-token chunk    the reply IS content, so every
                                                token is described
    parser, no stop        one token's text     exact by construction
    parser, no logprobs    multi-token chunk    nothing to attribute
    parser + stop + lp     --                   REFUSED at the gate

The refused row has no answer, not an expensive one: a parser reports no
difference between buffering an ambiguous marker prefix and consuming a complete
marker, so dropping the entry under-reports and keeping it corrupts
``logprobs.content``.

Free of HTTP, asyncio and vLLM: driven from a list of token ids.
"""

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
    """``choices[].logprobs`` around a list of entries, as vLLM shapes it.

    One envelope for both channels. ``refusal`` is always null: this path has no
    refusal channel, and the field is nullable.
    """
    return {"content": entries, "refusal": None}


@dataclass(frozen=True)
class Emission:
    """One piece of reply, on one channel, with the logprobs of its own tokens.

    ``text`` for :data:`CONTENT` / :data:`REASONING`; ``tool_call`` instead for
    :data:`TOOL_CALL`, already shaped for the OpenAI delta and always WHOLE --
    the parser emits each index once, arguments complete.

    ``logprobs`` is non-empty only on :data:`CONTENT`: the contract covers
    ``message.content`` alone. Empty ``text`` with non-empty ``logprobs`` is
    legitimate -- a token can run and produce no visible text, and vLLM reports
    it.
    """

    channel: str
    text: str = ""
    logprobs: list[dict] = field(default_factory=list)
    tool_call: dict | None = None


@dataclass
class _Entry:
    """One token's logprob record, waiting for the emission carrying its text."""

    token_id: int
    logprob: float | None
    candidates: list[tuple[int, float]]
    # Character offset where this token's text ends, or -1 while unknown: a
    # token can decode to nothing of its own -- the first half of a split
    # multi-byte character -- and its text arrives with a later token.
    ends_at: int


class ReplyStream:
    """Drive one request's tokens through the three transformations.

    ``push`` may be called once with the whole sequence or per streamed message;
    the emissions are the same either way, which is what the two channels'
    agreement rests on. Then ``finish()``, then ``token_ids`` / ``stop_reason``.
    """

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
        if stop and parser_session is not None and logprobs_req is not None:
            # Reaching here past the gate is a routing bug; 502 is the honest
            # answer for one.
            raise ValueError(
                "stop with both an output parser and logprobs is not "
                "attributable; the request gate must refuse it"
            )

        self._session = parser_session
        self._logprobs_req = logprobs_req
        self._first_lp = first_token_logprob

        # Specials are kept only when a parser will consume them. One policy
        # drives both the matcher and the output, or a stop spelled like a
        # special ends one channel's reply and not the other's.
        self._detok = IncrementalDetok(tokenizer, skip_special_tokens=parser_session is None)
        self._decode_one: Callable[[int], str] = lambda t: tokenizer.decode(
            [t], skip_special_tokens=False
        )
        self._window = StopWindow(stop, include_stop_in_output)

        self._ids: list[int] = []
        self._chars = 0  # characters the detokeniser produced
        self._due: list[_Entry] = []  # entries not yet on an emission
        self._held: list[dict] = []  # built entries with no channel yet
        self._finished = False

    # ── what the caller reports rather than computes ─────────────────────────

    @property
    def token_ids(self) -> list[int]:
        """The tokens this reply is made of, in order.

        The node delivers in batches, so a stop lands part-way into one; the
        tokens behind it ran only because the node cannot see text. A caller
        keeping its own list from the wire reported 30 ids against a
        ``completion_tokens`` of 6 on a live pair.
        """
        return list(self._ids)

    @property
    def completion_tokens(self) -> int:
        """Tokens generated for this reply, which is what the client is billed.

        A stop truncates the text, not the count. Matches vLLM, whose
        ``completion_tokens`` is its detokeniser's untruncated id list.
        """
        return len(self._ids)

    @property
    def stop_reason(self) -> str | None:
        """The stop string that ended the sequence, or None."""
        return self._window.stopped

    def finish_reason(self, from_decode: str) -> str:
        """What to report, given what the node said.

        A stop overrides it: the node cannot see text, so it cannot reach that conclusion.
        """
        return "stop" if self._window.stopped is not None else from_decode

    # ── the stream ──────────────────────────────────────────────────────────

    def push(
        self,
        token_ids: Iterable[int],
        logprobs: list[float | None] | None = None,
        candidates: list[list[tuple[int, float]]] | None = None,
    ) -> list[Emission]:
        """Absorb tokens; return whatever became emittable.

        One token at a time internally: batching the detokeniser is cheaper but
        loses the boundaries the stop cut and the entries both need.
        """
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
        """Release the held tail, then flush the parser. Idempotent.

        Everything owed after the last ``push`` comes from here, including
        entries for tokens that produced no visible text.
        """
        if self._finished:
            return []
        self._finished = True

        # The detokeniser may still hold a partial character. Its text belongs to
        # the reply -- the tokens are already counted -- so it goes through the
        # matcher like any other, before the window is drained.
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
            # These tokens ran; the stop removed their text, or the caller
            # strips them. vLLM reports them, so so do we.
            out.append(Emission(CONTENT, "", leftover))
        return out

    # ── internals ───────────────────────────────────────────────────────────

    def _queue(self, tid, logprobs, candidates, i, produced_text) -> None:
        """Record this token's logprob, to be attached when its text goes out.

        A token that produced nothing gets ``ends_at = -1`` and is backfilled by
        whichever token completes its character. Marking it complete at the
        current offset would make it due before the character it belongs to is
        out, so its entry would ride an empty chunk while the chunk carrying the
        character described one token too few.
        """
        if self._logprobs_req is None:
            return
        lp = logprobs[i] if logprobs and i < len(logprobs) else None
        cands = list(candidates[i]) if candidates and i < len(candidates) else []
        if len(self._ids) == 1 and lp is None and self._first_lp is not None:
            # Token 1 was echoed, not sampled, by the node: its distribution
            # exists only in the prefill reply.
            lp, cands = self._first_lp[0], list(self._first_lp[1] or [])
        if produced_text:
            for pending in self._due:
                if pending.ends_at < 0:
                    pending.ends_at = self._chars
        self._due.append(_Entry(tid, lp, cands, ends_at=self._chars if produced_text else -1))

    def _cap(self) -> int | None:
        """Ceiling so an emission never ends inside a token whose entry is still due.

        Text and the entry describing it then go out together.

        The window holds text back by a CHARACTER count, which lands mid-token:
        the visible part of a token would go out while its entry waited for the
        offset to clear, and the entry then rode a later chunk. Only stop strings
        create a holdback, so this binds for stop with logprobs and nothing else.
        """
        if self._logprobs_req is None:
            return None
        end = self._chars - self._window.hold
        cap = self._window.visible
        for pending in self._due:
            if 0 <= pending.ends_at <= end:
                cap = pending.ends_at
        return cap

    def _route(self, text: str) -> list[Emission]:
        """Send text the client may see to its channel(s)."""
        if self._session is None:
            entries = self._ready()
            if not text and not entries:
                return []
            return [Emission(CONTENT, text, entries)]
        entries = self._held + self._ready()
        if not text:
            # No characters contributed, so no channel decided: the first half
            # of a split multi-byte character waits for the second.
            self._held = entries
            return []
        self._held = []
        return self._events(self._session.feed(text), entries)

    def _events(self, events: list[dict], entries: list[dict]) -> list[Emission]:
        """Parser events as emissions, with the pending entries if content."""
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
                entries = []  # the first content emission carries them
        return out

    def _ready(self) -> list[dict]:
        """Entries whose token's text the client can now see.

        Once the window has stopped no more text can arrive, so a token the cut
        ran through is as visible as it will ever be: its surviving prefix is in
        the emission being built, and its entry belongs there rather than on a
        later empty chunk. `ends_at` still points past the untruncated token, so
        the offset comparison alone would defer it.

        A token that produced nothing yet (`ends_at < 0`) is never ready: its
        text arrives with a later token, or at the detokeniser's final flush.
        """
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
        """Every remaining entry, at end of stream."""
        return self._build(len(self._due))

    def _build(self, n: int) -> list[dict]:
        if not n:
            return []
        taken, self._due = self._due[:n], self._due[n:]
        assert self._logprobs_req is not None  # entries are queued only when asked for
        return build_logprobs(
            [e.token_id for e in taken],
            [e.logprob for e in taken],
            [[(int(c[0]), float(c[1])) for c in e.candidates] for e in taken],
            self._logprobs_req,
            self._decode_one,
        )["content"]
