from collections.abc import Callable
from typing import Any, Protocol

from tilert.pd_vllm.grammar_spec import (
    GrammarUnsupported,
    GrammarViolationError,
    InvalidGrammarError,
)


class PDEngine(Protocol):

    def inject(self, req: Any) -> None:
        pass

    def prepare_grammar(self, grammar_spec: dict | None, enable_thinking: bool = True) -> Any:
        pass

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
        pass

    def supports_logprobs(self) -> bool:
        pass

    def supports_penalties(self) -> bool:
        pass

    def supports_ignore_eos(self) -> bool:
        pass

    def reset(self) -> None:
        pass


class StubEngine:
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
            raise GrammarUnsupported("constrained decoding is not supported")
        if kind not in self._KNOWN_SPEC_TYPES:
            raise InvalidGrammarError(f"unsupported grammar spec type: {kind!r}")
        return {"spec": grammar_spec, "enable_thinking": enable_thinking}

    def supports_logprobs(self) -> bool:
        return True

    def supports_penalties(self) -> bool:
        return True

    def supports_ignore_eos(self) -> bool:
        return True

    @staticmethod
    def fake_logprob(token_id: int) -> float:
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
                    cands = [(t + k, self.fake_logprob(t) - 0.5 * k) for k in range(top_logprobs)]
                    on_token(t, self.fake_logprob(t), cands)
        self.last_stats = {"finish_reason": "stop"}
        return out

    def reset(self) -> None:
        self.injected = None
