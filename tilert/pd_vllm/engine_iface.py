"""Engine seam for the PD decode server (model-agnostic).

``PDEngine`` is the interface the decode server drives; concrete adapters are
built by the active model profile (``profile.build_engine(...)``).
``StubEngine`` runs the whole serving path with no GPU / no tilert.
"""

from collections.abc import Callable
from typing import Any, Protocol

from tilert.pd_vllm.grammar_spec import (
    GrammarBackendUnavailable,
    GrammarViolationError,
    InvalidGrammarError,
)


class PDEngine(Protocol):
    def inject(self, req: Any) -> None:
        """Restore engine state to 'prefilled seq_len tokens' from req."""

    def prepare_grammar(self, grammar_spec: dict | None, enable_thinking: bool = True) -> Any:
        """Compile ``grammar_spec`` into a per-request grammar session.

        Runs BEFORE any KV wire-wait / inject / GPU work, so a bad spec fails
        fast.

        Returns an opaque session object (passed back to :meth:`decode`), or
        None when ``grammar_spec`` is None (unconstrained). Raises
        ``InvalidGrammarError`` (client 400) for a malformed/unsupported spec
        and ``GrammarBackendUnavailable`` (server 500) when xgrammar is
        absent — never silently degrades to unconstrained decoding.
        """

    def decode(
        self,
        first_token_id: int,
        max_tokens: int,
        sampling: dict | None,
        on_token: Callable[..., None] | None = None,
        cancel_event=None,
        grammar_session: Any = None,
        top_logprobs: int | None = None,
    ) -> list[int]:
        """AR/MTP decode from first_token_id; returns completion ids.

        Includes first_token_id, excludes the stop token. on_token never fires
        for stop tokens; cancel_event stops early; last_stats['finish_reason']
        is 'stop' | 'length' | 'cancelled'. When ``grammar_session`` is set,
        every emitted token is grammar-masked/validated; a rejected token
        raises ``GrammarViolationError``.

        ``top_logprobs`` is the number of candidates the caller wants per
        position, or None for no logprobs. An engine that supports it calls
        ``on_token(token_id, logprob, candidates)`` -- ``candidates`` being a
        list of ``(token_id, logprob)`` longest-first -- instead of
        ``on_token(token_id)``. The extra arguments are optional at the call
        site, so an engine that ignores ``top_logprobs`` keeps working; the
        decode server detects the absence and reports it rather than returning a
        response with the field silently missing. Support is declared by
        :meth:`supports_logprobs`.
        """

    def supports_logprobs(self) -> bool:
        """Whether :meth:`decode` honours ``top_logprobs``.

        Optional: an engine that does not define it is treated as unsupported.
        """

    def supports_penalties(self) -> bool:
        """Whether :meth:`decode` honours ``repetition_penalty`` / ``presence_penalty``.

        Optional, and read by ``decode_server``'s ``/capabilities`` so the
        router can refuse a penalty request BEFORE prefilling it -- the
        alternative is a 501 after the KV has already crossed the wire. An
        engine that does not define it is treated as unsupported, which is the
        safe direction: the request is refused rather than decoded unpenalised.
        """

    def supports_ignore_eos(self) -> bool:
        """Whether :meth:`decode` honours ``ignore_eos``.

        Optional, same contract as :meth:`supports_penalties`. An engine that
        claims this must clear its stop set for the request, not merely accept
        the key -- accepting and ignoring it is the failure this exists to stop.
        """

    def reset(self) -> None:
        """Release per-request state."""


class StubEngine:
    """Echo engine for plumbing tests: no GPU, no tilert.

    ``prepare_grammar`` simulates the real classification deterministically via
    spec sentinels so the fail-closed HTTP mapping can be tested without a GPU:
      - ``{"type": "__backend_missing__"}`` -> GrammarBackendUnavailable (500)
      - an unknown/malformed spec           -> InvalidGrammarError (400)
      - ``{"type": "regex", "value": "__violate__"}`` -> decode raises
        GrammarViolationError (400) on the first token
    """

    _KNOWN_SPEC_TYPES = ("json_schema", "json_object", "ebnf", "regex", "structural_tag")

    def __init__(self, fixed_tokens: tuple[int, ...] = (11, 22, 33)):
        self._fixed = fixed_tokens
        self.injected: Any = None
        self.last_stats: dict = {}

    def inject(self, req: Any) -> None:
        self.injected = req

    def prepare_grammar(self, grammar_spec, enable_thinking=True):
        if grammar_spec is None:
            return None
        if not isinstance(grammar_spec, dict) or "type" not in grammar_spec:
            raise InvalidGrammarError("grammar spec must be a dict with a 'type'")
        kind = grammar_spec["type"]
        if kind == "__backend_missing__":
            raise GrammarBackendUnavailable("xgrammar backend not installed")
        if kind not in self._KNOWN_SPEC_TYPES:
            raise InvalidGrammarError(f"unsupported grammar spec type: {kind!r}")
        return {"spec": grammar_spec, "enable_thinking": enable_thinking}

    def supports_logprobs(self) -> bool:
        return True

    def supports_penalties(self) -> bool:
        # The echo engine applies no sampling at all, but it must not be the
        # reason a plumbing test cannot reach the penalty path.
        return True

    def supports_ignore_eos(self) -> bool:
        return True

    @staticmethod
    def fake_logprob(token_id: int) -> float:
        """Deterministic stand-in so tests can assert exact values."""
        return -0.5 - 0.25 * (token_id % 4)

    def decode(
        self,
        first_token_id,
        max_tokens,
        sampling,
        on_token=None,
        cancel_event=None,
        grammar_session=None,
        top_logprobs=None,
    ):
        spec = (grammar_session or {}).get("spec", {})
        if spec.get("value") == "__violate__":
            raise GrammarViolationError(f"first token {first_token_id} violates the grammar")
        out = ([int(first_token_id)] + list(self._fixed))[:max_tokens]
        if on_token:
            for t in out:
                if top_logprobs is None:
                    on_token(t)
                else:
                    # Candidates are the token itself plus neighbours, so the
                    # ordering (chosen first, then descending) is checkable.
                    cands = [(t + k, self.fake_logprob(t) - 0.5 * k) for k in range(top_logprobs)]
                    on_token(t, self.fake_logprob(t), cands)
        self.last_stats = {"finish_reason": "stop"}
        return out

    def reset(self) -> None:
        self.injected = None
