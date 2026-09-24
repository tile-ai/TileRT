from __future__ import annotations

import json
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
RETRY = "retry"
BUSY = "busy"
PROPAGATE = "propagate"
SERVER_ERROR = "server_error"


def classify_decode_status(
    status: int, payload: Any, *, attempts_left: bool, propagated_types: tuple[str, ...]
) -> str:
    if status == 429:
        return RETRY if attempts_left else BUSY
    if status == 200:
        raise ValueError("200 is not an error status")
    if isinstance(payload, dict) and payload.get("error_type") in propagated_types:
        return PROPAGATE
    return SERVER_ERROR


PROPAGATED_ERROR_STATUS = {
    "grammar_unsupported": 501,
    "invalid_grammar": 400,
    "grammar_violation": 400,
    "grammar_backend_unavailable": 500,
    "logprobs_unavailable": 501,
    "capability_unavailable": 501,
    "invalid_parameter": 400,
    "request_cancelled": 499,
}
PROPAGATED_ERROR_TYPES = frozenset(PROPAGATED_ERROR_STATUS)
OK = "ok"
REFUSED = "refused"
TRUNCATED = "truncated"
TYPED_ERROR = "typed_error"
UNTYPED_ERROR = "untyped_error"


def terminal_verdict(reader, *, client_gone: bool = False) -> tuple[str, dict, int]:
    if reader.refusal is not None:
        return (REFUSED, reader.refusal, reader.refusal_status)
    if reader.node_error is not None:
        error_type = reader.node_error.get("error_type")
        if error_type in PROPAGATED_ERROR_TYPES:
            return (TYPED_ERROR, reader.node_error, PROPAGATED_ERROR_STATUS[error_type])
        return (UNTYPED_ERROR, reader.node_error, 502)
    if not reader.node_terminated and (not reader.stop_hit) and (not client_gone):
        return (
            TRUNCATED,
            {
                "error": "decode stream ended without a terminal message",
                "error_type": "decode_truncated",
                "rid": reader.rid,
            },
            502,
        )
    return (OK, {}, 200)


def decode_refusal(verdict: str, status: int, payload: Any, rid: str) -> tuple[dict, int]:
    if verdict == BUSY:
        return ({"error": "decode node busy", "error_type": "decode_busy", "rid": rid}, 429)
    if verdict == PROPAGATE:
        return (payload, status)
    if verdict == SERVER_ERROR:
        return ({"error": "decode call failed", "status": status, "rid": rid}, 502)
    raise ValueError(f"{verdict} is not a refusal")


@dataclass
class DecodeReader:
    stream: Any = None
    logprobs_req: Any = None
    rid: str = ""
    token_ids: list[int] = field(default_factory=list)
    timing: dict = field(default_factory=dict)
    finish_reason: str = "stop"
    node_terminated: bool = False
    refusal: dict | None = None
    refusal_status: int = 501
    node_error: dict | None = None
    _stopped: bool = False

    @property
    def finished(self) -> bool:
        return self.node_terminated or self._stopped or self.refusal is not None

    @property
    def stop_hit(self) -> bool:
        return self._stopped

    def feed(self, line: str) -> list:
        if not line:
            return []
        return self._message(json.loads(line))

    def feed_blocking(self, body: dict) -> list:
        self.timing = body.get("timing_ms", {})
        self._set_finish(self.timing.get("finish_reason", "stop"))
        self.node_terminated = True
        lp = body.get("logprobs") or {}
        return self._tokens({"t": body["token_ids"], **lp})

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
                self.refusal = {
                    "error": f"decode node declared {declared} tokens and sent {got}",
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
        self.finish_reason = "stop" if reason == "cancelled" else reason

    def _tokens(self, msg: dict) -> list:
        ids = msg["t"]
        seen = len(self.stream.token_ids) if self.stream is not None else len(self.token_ids)
        if self.logprobs_req is not None and (
            not _logprobs_line_ok(msg, len(ids), self.logprobs_req, seen)
        ):
            self.refusal = {
                "error": "decode node returned no logprobs",
                "error_type": "logprobs_unavailable",
                "rid": self.rid,
            }
            return []
        if self.stream is None:
            self.token_ids += ids
            return []
        out = self.stream.push(ids, msg.get("lp"), msg.get("tp"))
        if self.stream.stop_reason is not None:
            self._stopped = True
        return out


def _logprobs_line_ok(payload: dict, n_tokens: int, req, seen: int = 0) -> bool:
    lp = payload.get("lp")
    if lp is None or len(lp) != n_tokens:
        return False
    if any((value is None for i, value in enumerate(lp) if not (seen == 0 and i == 0))):
        return False
    if req.top_n > 0:
        tp = payload.get("tp")
        if tp is None or len(tp) != n_tokens:
            return False
    return True
