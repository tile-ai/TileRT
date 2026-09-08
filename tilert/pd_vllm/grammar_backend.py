"""Where the engine keeps its xgrammar host wrapper.

Engine builds have shipped the wrapper at two locations: ``tilert.grammar``
(current) and, in older wheels, under a model package
(``tilert.models.glm_5_2.grammar``). The model-package location decided which
products could constrain anything -- a wheel that excluded that model package
excluded the mask producer with it, so a ``response_format`` request failed
naming a model the client was not serving.

One serve version has to work against engine wheels from either side of that
move, so the new path is tried first and the old one is the fallback.
"""

from __future__ import annotations

from typing import Any


def load_grammar_backend() -> tuple[Any, Any]:
    """Return ``(GrammarEngine, GrammarSession)`` from the installed engine.

    Raises ``ModuleNotFoundError`` when the engine ships neither path -- the
    message names no model, because which model the caller is serving has
    nothing to do with why the backend is missing.
    """
    try:
        from tilert.grammar import GrammarEngine, GrammarSession

        return GrammarEngine, GrammarSession
    except ImportError:
        pass
    try:
        # Engine wheels that predate ``tilert.grammar``.
        from tilert.models.glm_5_2.grammar import GrammarEngine, GrammarSession

        return GrammarEngine, GrammarSession
    except ImportError as e:
        raise ModuleNotFoundError(
            "this engine build ships no xgrammar host backend (looked for tilert.grammar)"
        ) from e
