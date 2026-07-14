"""Engine seam for the PD decode server (model-agnostic).

``PDEngine`` is the interface the decode server drives; concrete adapters are
built by the active model profile (``profile.build_engine(...)``).
``StubEngine`` runs the whole serving path with no GPU / no tilert.
"""

from collections.abc import Callable
from typing import Any, Protocol


class PDEngine(Protocol):
    def inject(self, req: Any) -> None:
        """Restore engine state to 'prefilled seq_len tokens' from req."""

    def decode(
        self,
        first_token_id: int,
        max_tokens: int,
        sampling: dict | None,
        on_token: Callable[[int], None] | None = None,
        cancel_event=None,
    ) -> list[int]:
        """AR/MTP decode from first_token_id; returns completion ids.

        Includes first_token_id, excludes the stop token. on_token never fires
        for stop tokens; cancel_event stops early; last_stats['finish_reason']
        is 'stop' | 'length' | 'cancelled'.
        """

    def reset(self) -> None:
        """Release per-request state."""


class StubEngine:
    """Echo engine for plumbing tests: no GPU, no tilert."""

    def __init__(self, fixed_tokens: tuple[int, ...] = (11, 22, 33)):
        self._fixed = fixed_tokens
        self.injected: Any = None
        self.last_stats: dict = {}

    def inject(self, req: Any) -> None:
        self.injected = req

    def decode(self, first_token_id, max_tokens, sampling, on_token=None, cancel_event=None):
        out = ([int(first_token_id)] + list(self._fixed))[:max_tokens]
        if on_token:
            for t in out:
                on_token(t)
        self.last_stats = {"finish_reason": "stop"}
        return out

    def reset(self) -> None:
        self.injected = None
