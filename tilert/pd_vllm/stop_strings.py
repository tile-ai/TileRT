from __future__ import annotations

import sys

__all__ = ["StopWindow", "check_stop_strings", "resolve_stop"]


def resolve_stop(body: dict) -> list[str]:
    raw = body.get("stop")
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"stop must be a string or list of strings, got {type(raw).__name__}")
    out = []
    for s in raw:
        if not isinstance(s, str):
            raise ValueError(f"stop entries must be strings, got {type(s).__name__}")
        if not s:
            raise ValueError("stop cannot contain an empty string")
        out.append(s)
    return out


def check_stop_strings(
    output_text: str, new_char_count: int, stop: list[str], include_in_output: bool
) -> tuple[str, int] | None:
    if not new_char_count or not stop:
        return None
    best_stop_str: str | None = None
    best_stop_index = 0
    best_end = sys.maxsize
    for stop_str in stop:
        stop_len = len(stop_str)
        stop_index = output_text.find(stop_str, 1 - new_char_count - stop_len)
        if stop_index == -1:
            continue
        end = stop_index + stop_len
        if end < best_end:
            best_stop_str = stop_str
            best_stop_index = stop_index
            best_end = end
    if best_stop_str is None:
        return None
    if include_in_output:
        if best_end >= len(output_text):
            return (best_stop_str, -1)
        return (best_stop_str, best_end)
    return (best_stop_str, best_stop_index)


class StopWindow:

    def __init__(self, stop: list[str], include_in_output: bool = False):
        self.stop = list(stop)
        self.include_in_output = include_in_output
        longest = max((len(s) for s in self.stop), default=1) - 1
        self._hold = 0 if include_in_output else longest
        self._keep = longest
        self._slack = 4096 if self.stop else 0
        self._text = ""
        self._base = 0
        self._taken = 0
        self._stopped: str | None = None

    @property
    def stopped(self) -> str | None:
        return self._stopped

    @property
    def hold(self) -> int:
        return self._hold

    @property
    def visible(self) -> int:
        return self._taken

    def push(self, delta: str) -> None:
        if self._stopped is not None or not delta:
            return
        self._text += delta
        hit = check_stop_strings(self._text, len(delta), self.stop, self.include_in_output)
        if hit is None:
            return
        self._stopped, truncate_to = hit
        if truncate_to != -1:
            self._text = self._text[:truncate_to]

    def take(self, *, final: bool = False, limit: int | None = None) -> str:
        end = self._base + len(self._text)
        if not (final or self._stopped is not None):
            end -= self._hold
            if limit is not None and limit < end:
                end = limit
        if end <= self._taken:
            return ""
        out = self._text[self._taken - self._base : end - self._base]
        self._taken = end
        self._trim()
        return out

    def _trim(self) -> None:
        if len(self._text) <= self._keep + self._hold + self._slack:
            return
        cut = self._taken - self._keep - self._base
        if cut > 0:
            self._text = self._text[cut:]
            self._base += cut
