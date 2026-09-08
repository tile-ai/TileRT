"""Text-level ``stop`` matching, on the router's side of the detokeniser.

``stop`` is a property of the decoded TEXT, not of token ids: a stop string
routinely spans two tokens ("Observation:" is often three), and byte-level
BPE can split one character across tokens. The node emits ids; the router is
where text exists.

:func:`check_stop_strings` is a port of vLLM's ``v1/engine/detokenizer.py``, so
the same request stops at the same character on both stacks. Two rules are easy
to get wrong: the search starts at ``1 - new_char_count - len(stop_str)``, so a
stop straddling the delta boundary is still found; and when several match in one
step -- routine under MTP -- the one COMPLETING EARLIEST wins, so the result does
not depend on the batch size. Ties go to stop-list order.

:class:`StopWindow` splits accumulation from release as vLLM does: ``push``
absorbs and matches, ``take`` is a read cursor. Neither then needs a queue of
held pieces. One difference: vLLM keeps the whole reply because its non-streaming
response needs it, so this class drops what has gone out -- keeping it cost
0.75 s and 200 KB over a 200k-token generation.
"""

from __future__ import annotations

import sys

__all__ = ["StopWindow", "check_stop_strings", "resolve_stop"]


def resolve_stop(body: dict) -> list[str]:
    """The request's stop strings, normalised to a list.

    ``str | list[str] | None``, as vLLM accepts. An empty string is REJECTED, not
    dropped: it matches at position 0 of everything, so it is not a neutral value
    the way ``[]`` is, and dropping it would serve an unrestricted completion to
    a client who asked for a restricted one. vLLM raises the same from
    ``SamplingParams._verify_args``, and the router strips ``stop`` from the
    prefill request, so vLLM no longer gets the chance to.
    """
    raw = body.get("stop")
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"stop must be a string or list of strings, " f"got {type(raw).__name__}")
    out = []
    for s in raw:
        if not isinstance(s, str):
            raise ValueError(f"stop entries must be strings, " f"got {type(s).__name__}")
        if not s:
            raise ValueError("stop cannot contain an empty string")
        out.append(s)
    return out


def check_stop_strings(
    output_text: str,
    new_char_count: int,
    stop: list[str],
    include_in_output: bool,
) -> tuple[str, int] | None:
    """``(stop_string, truncate_to)`` if one matched, else ``None``.

    ``truncate_to`` is the length to cut ``output_text`` to, or ``-1`` for none.
    vLLM port; see the module docstring for the two rules.
    """
    if not new_char_count or not stop:
        return None

    best_stop_str: str | None = None
    best_stop_index = 0
    best_end = sys.maxsize
    for stop_str in stop:
        stop_len = len(stop_str)
        # Start before the new text so a stop spanning the boundary is found,
        # without re-scanning text that was already checked.
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
            return best_stop_str, -1
        return best_stop_str, best_end
    return best_stop_str, best_stop_index


class StopWindow:
    """Decoded text, matched against the stop strings, behind a read cursor.

    ``take`` lags ``push`` by ``max(len(s) for s in stop) - 1`` characters: text
    on the wire cannot be recalled, and the last few may begin a stop string.
    Nothing lags without stop strings, or when the stop stays in the output --
    nothing to remove then, the same condition vLLM uses for its
    ``stop_buffer_length``.
    """

    def __init__(self, stop: list[str], include_in_output: bool = False):
        self.stop = list(stop)
        self.include_in_output = include_in_output
        longest = max((len(s) for s in self.stop), default=1) - 1
        # How far `take` stays behind the end of the text.
        self._hold = 0 if include_in_output else longest
        # Look-back kept behind the cursor: the matcher reaches at most
        # len(stop)-1 characters back from the newest delta.
        self._keep = longest
        # Slack above what matching needs, so trimming is amortised. Zero
        # without stop strings, which empties the window on every `take` --
        # one code path, degenerating to a pass-through.
        self._slack = 4096 if self.stop else 0

        self._text = ""  # a WINDOW of the stream, not all of it
        self._base = 0  # absolute index of self._text[0]
        self._taken = 0  # absolute count handed to `take`
        self._stopped: str | None = None

    # ── what the caller reports ─────────────────────────────────────────────

    @property
    def stopped(self) -> str | None:
        """The stop string that ended the sequence, or None."""
        return self._stopped

    @property
    def hold(self) -> int:
        """How far `take` stays behind the newest text, in characters.

        A caller pairing text with per-token metadata rounds this down to a
        token boundary: a character count lands mid-token.
        """
        return self._hold

    @property
    def visible(self) -> int:
        """How many characters `take` has handed out.

        A caller holding per-token metadata pairs it against this: an entry is
        due once the text it describes is past this point.
        """
        return self._taken

    # ── the stream ──────────────────────────────────────────────────────────

    def push(self, delta: str) -> None:
        """Absorb newly decoded text and match. Releasing is ``take``'s job --

        doing both here is what forced a released/unreleased split.
        """
        if self._stopped is not None or not delta:
            return
        self._text += delta
        # Matcher offsets are window-relative; `_base` converts them.
        hit = check_stop_strings(self._text, len(delta), self.stop, self.include_in_output)
        if hit is None:
            return
        self._stopped, truncate_to = hit
        if truncate_to != -1:
            self._text = self._text[:truncate_to]

    def take(self, *, final: bool = False, limit: int | None = None) -> str:
        """The text the client may see now.

        ``final`` releases the held tail: nothing more can arrive to turn it into
        a stop. A matched stop is also final -- the text is already cut.

        ``limit`` is an absolute ceiling a caller sets to keep the release on a
        boundary of its own; it does not apply once the text is final or cut,
        when everything must go out regardless.
        """
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
        return out  # noqa: R504 (_trim mutates the window after the slice)

    def _trim(self) -> None:
        """Drop text the matcher can no longer reach.

        Keeps from ``_taken - _keep`` onward, and only runs once the window
        exceeds that by ``_slack``, so the copy is amortised.
        """
        if len(self._text) <= self._keep + self._hold + self._slack:
            return
        cut = self._taken - self._keep - self._base
        if cut > 0:
            self._text = self._text[cut:]
            self._base += cut
