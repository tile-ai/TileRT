"""Reading a ``/pd/decode`` HTTP response, independent of how the bytes arrive.

Not the PD wire protocol -- that is prefill <-> decode over TCP and RDMA and
lives in ``wire.py``. This is the HTTP answer the node gives the ROUTER.

``/pd/decode`` returns one JSON object, or NDJSON lines (``{"t": [...]}``
repeatedly, then ``{"done": ...}`` or ``{"error": ...}``). Both router paths read
it and both used to carry their own copy: four message kinds, a logprobs line to
validate, a terminal message to notice, a node to cancel when one never arrived.

Only the TRANSPORT and the PRESENTATION genuinely differ -- ``requests`` on a
worker thread vs ``httpx`` on the event loop, and an HTTP status vs an SSE event
once the streaming response has sent its 200. So the caller keeps the loop and
the presentation, and :class:`DecodeReader` takes the rest::

    for line in <transport>:
        for emission in reader.feed(line):
            <present emission>
        if reader.finished:
            break
    # then, once: reader.refusal / reader.node_error / reader.timing
"""

from __future__ import annotations

import json
from collections.abc import Container
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "BUSY",
    "DecodeReader",
    "OK",
    "PROPAGATE",
    "REFUSED",
    "RETRY",
    "SERVER_ERROR",
    "TRUNCATED",
    "TYPED_ERROR",
    "UNTYPED_ERROR",
    "PROPAGATED_ERROR_STATUS",
    "PROPAGATED_ERROR_TYPES",
    "terminal_verdict",
    "decode_refusal",
    "classify_decode_status",
]

# What to do about a `/pd/decode` response status.
RETRY = "retry"  # 429 and attempts remain
BUSY = "busy"  # 429 and they do not: an honest, retryable 429
PROPAGATE = "propagate"  # a typed error the client should see verbatim
SERVER_ERROR = "server_error"  # anything else: the node is broken


def classify_decode_status(
    status: int, payload: Any, *, attempts_left: bool, propagated_types: Container[str]
) -> str:
    """What a ``/pd/decode`` status means, without deciding how to say it.

    A 429 is the node's admission control answering, so it is retryable rather
    than a fault. A typed error is the node's considered answer about THIS
    request and reaches the client verbatim; flattening it into 502 would claim
    a component is broken when none is.
    """
    if status == 429:
        return RETRY if attempts_left else BUSY
    if status == 200:
        raise ValueError("200 is not an error status")
    if isinstance(payload, dict) and payload.get("error_type") in propagated_types:
        return PROPAGATE
    return SERVER_ERROR


# What a node's own error types mean to the client. Taken from the classes that
# raise them, because the blocking protocol carries an HTTP status while the
# streaming one carries the error inside a spent 200.
PROPAGATED_ERROR_STATUS = {
    # grammar_spec.py
    "invalid_grammar": 400,
    "grammar_violation": 400,
    "grammar_backend_unavailable": 500,
    # decode_server.py: the engine cannot produce what was asked for.
    "logprobs_unavailable": 501,
    # capabilities.py. Reachable even though the router pre-validates: its
    # capability cache can be up to Pool.CAPS_TTL_S stale, and the node is the
    # authority.
    "capability_unavailable": 501,
    "invalid_parameter": 400,
    # The caller asked the node to stop mid-request; not a component fault.
    "request_cancelled": 499,
}
PROPAGATED_ERROR_TYPES = frozenset(PROPAGATED_ERROR_STATUS)


# What the reader's FINAL state means.
OK = "ok"  # nothing to refuse
REFUSED = "refused"  # the router cannot use what arrived
TRUNCATED = "truncated"  # a clean EOF with no terminal message
TYPED_ERROR = "typed_error"  # the node classified it; the client can act
UNTYPED_ERROR = "untyped_error"  # the node broke


def terminal_verdict(reader, *, client_gone: bool = False) -> tuple[str, dict, int]:
    """What the reader's final state means, and what says it.

    The order is the point. Both response paths had this chain, in DIFFERENT
    orders -- one checked the node's error before the truncation test and the
    other after. They happened to agree, because an error line also marks the
    node terminated, so truncation cannot fire alongside one; nothing said so.

    Rendering stays with the caller: only the blocking path can answer a status,
    and only the streaming path has to say it inside a spent 200.
    """
    if reader.refusal is not None:
        return REFUSED, reader.refusal, reader.refusal_status
    if reader.node_error is not None:
        error_type = reader.node_error.get("error_type")
        if error_type in PROPAGATED_ERROR_TYPES:
            return (TYPED_ERROR, reader.node_error, PROPAGATED_ERROR_STATUS[error_type])
        return UNTYPED_ERROR, reader.node_error, 502
    if not reader.node_terminated and not reader.stop_hit and not client_gone:
        # Assembling it would report finish_reason "stop" for a reply whose body
        # was cut short, or whose node died.
        return (
            TRUNCATED,
            {
                "error": "decode stream ended without a terminal message",
                "error_type": "decode_truncated",
                "rid": reader.rid,
            },
            502,
        )
    return OK, {}, 200


def decode_refusal(verdict: str, status: int, payload: Any, rid: str) -> tuple[dict, int]:
    """How to SAY what :func:`classify_decode_status` decided.

    One body and one status per verdict, for both response paths. They answered
    a non-200 separately before, and the 502 differed between them: the blocking
    path let `raise_for_status` reach a generic handler and reported its message,
    the streaming one reported `decode call failed` with the status. Same node
    behaviour, two shapes, depending on whether the client asked for a stream.

    ``RETRY`` has no answer -- the caller retries -- so asking for one is a bug.
    """
    if verdict == BUSY:
        return ({"error": "decode node busy", "error_type": "decode_busy", "rid": rid}, 429)
    if verdict == PROPAGATE:
        return (payload, status)
    if verdict == SERVER_ERROR:
        return ({"error": "decode call failed", "status": status, "rid": rid}, 502)
    raise ValueError(f"{verdict} is not a refusal")


@dataclass
class DecodeReader:
    """Read one ``/pd/decode`` response, line by line, into emissions.

    ``feed`` returns whatever became emittable; the caller reads the rest after
    the loop.

    ``finished``        stop reading: the node terminated, or a stop ended the
                        reply while the node kept generating.
    ``node_terminated`` whether the NODE said so. It owns its slot until then, so
                        any other exit has to cancel.
    ``refusal`` /       a payload that cannot be served, and the status it
    ``refusal_status``  deserves: 501 when the router cannot use what arrived,
                        502 when what arrived is incomplete.
    ``node_error``      a typed error the node reported.
    ``timing`` /        from the terminal message; ``timing`` is empty when a stop
    ``finish_reason``   ended the reply, since it rides that line.
    """

    stream: Any = None  # a ReplyStream, or None with no tokenizer
    logprobs_req: Any = None
    rid: str = ""

    token_ids: list[int] = field(default_factory=list)
    timing: dict = field(default_factory=dict)
    finish_reason: str = "stop"
    node_terminated: bool = False
    refusal: dict | None = None
    # The status that refusal deserves. 501 when the router cannot serve what the
    # node sent, 502 when what the node sent is incomplete -- different faults,
    # and the caller should not have to infer which from the payload.
    refusal_status: int = 501
    node_error: dict | None = None
    _stopped: bool = False

    @property
    def finished(self) -> bool:
        """Whether the caller should stop reading."""
        return self.node_terminated or self._stopped or self.refusal is not None

    @property
    def stop_hit(self) -> bool:
        """Whether a stop string ended the reply rather than the node."""
        return self._stopped

    def feed(self, line: str) -> list:
        """One NDJSON line -> whatever became emittable."""
        if not line:
            return []
        return self._message(json.loads(line))

    def feed_blocking(self, body: dict) -> list:
        """The one-object form of the same protocol.

        Ids and logprobs arrive together, so it is one token message plus the
        terminal one.
        """
        self.timing = body.get("timing_ms", {})
        self._set_finish(self.timing.get("finish_reason", "stop"))
        self.node_terminated = True
        lp = body.get("logprobs") or {}
        return self._tokens({"t": body["token_ids"], **lp})

    # ── internals ───────────────────────────────────────────────────────────

    def _message(self, msg: dict) -> list:
        if "t" in msg:
            return self._tokens(msg)
        if "done" in msg:
            self.node_terminated = True
            self.timing = msg.get("timing_ms", {})
            self._set_finish(msg.get("finish_reason", "stop"))
            declared = msg.get("n")
            got = len(self.stream.token_ids) if self.stream is not None else len(self.token_ids)
            if declared is not None and declared != got:
                # The node states its own count on the terminal line, so a
                # mismatch means token lines were lost on the way -- a proxy, or
                # a node whose stream did not survive. Accepting it would report
                # a shortened generation with a successful finish reason and an
                # understated usage. Only reachable when nothing stopped the read
                # early: a matched stop or an earlier refusal never gets here.
                self.refusal = {
                    "error": f"decode node declared {declared} tokens and sent " f"{got}",
                    "error_type": "decode_truncated",
                    "rid": self.rid,
                }
                self.refusal_status = 502
            return []
        if "error" in msg:
            self.node_terminated = True
            self.node_error = msg
            return []
        return []

    def _set_finish(self, reason: str) -> None:
        """Both forms of the protocol normalise the same way.

        `cancelled` is the router's own doing, not a client-visible outcome: the
        reply it already has is complete. Setting it directly in one form and
        not the other is how it leaked once.
        """
        self.finish_reason = "stop" if reason == "cancelled" else reason

    def _tokens(self, msg: dict) -> list:
        ids = msg["t"]
        seen = len(self.stream.token_ids) if self.stream is not None else len(self.token_ids)
        if self.logprobs_req is not None and not _logprobs_line_ok(
            msg, len(ids), self.logprobs_req, seen
        ):
            # Refused, not padded: the documented sentinel would report a model
            # that had nothing to say about its own tokens.
            self.refusal = {
                "error": "decode node returned no logprobs",
                "error_type": "logprobs_unavailable",
                "rid": self.rid,
            }
            return []
        if self.stream is None:
            # No tokenizer: ids are all the reply can carry.
            self.token_ids += ids
            return []
        out = self.stream.push(ids, msg.get("lp"), msg.get("tp"))
        if self.stream.stop_reason is not None:
            self._stopped = True
        return out  # noqa: R504 (stop_reason is read after push)


def _logprobs_line_ok(payload: dict, n_tokens: int, req, seen: int = 0) -> bool:
    """Whether a node's logprobs cover every token in the same message.

    ``lp`` always. ``tp`` only when candidates were asked for: ``logprobs: true``
    alone resolves to ``top_n == 0``, where empty rows are the right answer. A
    short or absent ``tp`` is not an empty row -- it loses the alignment too.

    Null is a value, not a length: only the reply's FIRST token may report it,
    because prefill sampled it and no decode-side value exists. `seen` is how
    many tokens already arrived, so the exemption cannot follow a batch. Past
    there, a null is a node that cannot produce what was asked for -- padding it
    reaches the client as -9999.0, which OpenAI documents for "very unlikely"
    and is indistinguishable from a measurement. The decode server states the
    same invariant by raising ``LogprobsUnavailable``; this is where an older or
    faulty node is caught.
    """
    lp = payload.get("lp")
    if lp is None or len(lp) != n_tokens:
        return False
    if any(value is None for i, value in enumerate(lp) if not (seen == 0 and i == 0)):
        return False
    if req.top_n > 0:
        tp = payload.get("tp")
        if tp is None or len(tp) != n_tokens:
            return False
    return True
