"""Grammar/constrained-decoding serve-layer helpers for the pd_vllm path.

Two concerns, both framework-agnostic (no vLLM / no tilert imports), so this
module is unit-testable on a plain CPU box:

1. ``extract_request_grammar_spec`` — translate an OpenAI-style request dict
   into the engine's ``grammar_spec`` (``{"type", "value"}``). Selection
   priority mirrors the reference sglang implementation's grammar_manager:
   ``json_schema > regex > ebnf > structural_tag``.

2. The serve-layer error hierarchy used for fail-closed classification:
   - client sent a bad/unsupported spec  -> ``InvalidGrammarError`` (HTTP 400)
   - decode node lacks the xgrammar backend -> ``GrammarBackendUnavailable``
   - an emitted token violated the grammar  -> ``GrammarViolationError`` (400)

   Each carries ``error_type`` + ``http_status`` and a ``to_payload()`` that
   the decode server / router return verbatim, so the classification survives
   the /pd/decode HTTP hop.
"""

from __future__ import annotations

from typing import Any


# --------------------------------------------------------------------------- #
# Error hierarchy (fail-closed classification)
# --------------------------------------------------------------------------- #
class GrammarError(Exception):
    """Base for all grammar serve-layer errors. Never silently degrade."""

    error_type = "grammar_error"
    http_status = 500

    def to_payload(self) -> dict[str, str]:
        return {"error": str(self), "error_type": self.error_type}


class InvalidGrammarError(GrammarError):
    """Client sent a malformed/unsupported constraint -> HTTP 400.

    Raised both by request parsing (:func:`extract_request_grammar_spec`) and
    by grammar compilation (a schema/regex/EBNF xgrammar cannot compile).
    """

    error_type = "invalid_grammar"
    http_status = 400


class GrammarBackendUnavailable(GrammarError):
    """xgrammar is not installed on the decode node -> HTTP 500.

    A missing backend is a server-side deployment fault, NOT a client error: a
    legitimate ``response_format`` request must never be told its grammar is
    invalid just because the backend is absent.
    """

    error_type = "grammar_backend_unavailable"
    http_status = 500


class GrammarViolationError(GrammarError):
    """An emitted token violated the grammar -> HTTP 400 / SSE error event.

    Typically an unconstrained prefill first token that the matcher rejects.
    We fail closed rather than serve the rest of the request unconstrained.
    """

    error_type = "grammar_violation"
    http_status = 400


class GrammarUnsupported(GrammarError):
    """A constrained request hit a code path without grammar masking -> HTTP 500.

    Fail loud instead of silently serving the request unconstrained (staged
    rollout: e.g. the MTP path before it is wired).
    """

    error_type = "grammar_unsupported"
    http_status = 500


# --------------------------------------------------------------------------- #
# Spec validation
# --------------------------------------------------------------------------- #
# Compilation runs inside the decode node's single-slot lock, so an unbounded
# schema is downtime for everyone. Measured compile cost (150k-vocab tokenizer),
# limits chosen for ~1s worst case:
#   depth      100 -> 0.2s   200 -> 1.7s    400 -> 13.3s
#   properties 1000 -> 1.2s  5000 -> 36.3s  10000 -> 170.9s
#   enum      10000 -> 0.1s 20000 -> 1.1s   50000 -> 6.2s
# regex/EBNF need no cap: 200 000 chars of regex compiles in 0.7s.
_MAX_SCHEMA_DEPTH = 64
_MAX_SCHEMA_NODES = 1000
_MAX_ENUM_VALUES = 10000


def _walk_schema(node, depth, counts, path):
    """Check one schema node, recursing into subschemas."""
    where = path or "<root>"
    if depth > _MAX_SCHEMA_DEPTH:
        raise InvalidGrammarError(f"json_schema nests deeper than {_MAX_SCHEMA_DEPTH} at {where}")

    if isinstance(node, list):
        for i, item in enumerate(node):
            _walk_schema(item, depth, counts, f"{path}[{i}]")
        return
    if not isinstance(node, dict):
        return

    counts["nodes"] += 1
    if counts["nodes"] > _MAX_SCHEMA_NODES:
        raise InvalidGrammarError(f"json_schema has over {_MAX_SCHEMA_NODES} subschemas")

    if isinstance(node.get("enum"), list):
        counts["enums"] += len(node["enum"])
        if counts["enums"] > _MAX_ENUM_VALUES:
            raise InvalidGrammarError(f"json_schema has over {_MAX_ENUM_VALUES} enum values")

    for key, child in node.items():
        # Literal data, not subschemas: a 500-value enum is not 500 nodes.
        if key in ("enum", "const", "examples", "default"):
            continue
        if isinstance(child, (dict, list)):
            _walk_schema(child, depth + 1, counts, f"{path}.{key}" if path else key)


def validate_grammar_spec(spec: dict[str, Any] | None) -> None:
    """Reject a spec whose compilation would stall the decode node.

    Walks whatever the spec carries: json_schema and structural_tag hold
    subschemas, regex/ebnf hold a string and json_object nothing, both of which
    the walk skips. Called from :func:`extract_request_grammar_spec`, i.e. in
    the router before the prefill request runs, so an oversized schema never reaches
    a decode node. A direct ``/pd/decode`` call is not covered.

    Raises:
        InvalidGrammarError: over a compile-cost limit (400).
    """
    if spec:
        _walk_schema(spec.get("value"), 0, {"nodes": 0, "enums": 0}, "")


# --------------------------------------------------------------------------- #
# Request -> grammar_spec translation
# --------------------------------------------------------------------------- #
_SPEC_TYPES = ("json_schema", "json_object", "ebnf", "regex", "structural_tag")


def extract_request_grammar_spec(
    request: dict[str, Any],
) -> dict[str, Any] | None:
    """Return the engine grammar spec for a request, or None if unconstrained.

    Priority: json_schema > regex > ebnf > structural_tag (``json_object``
    returns early, before regex/ebnf, matching the reference).

    The returned spec has already passed :func:`validate_grammar_spec`.

    Raises:
        InvalidGrammarError: malformed, or over a compile-cost limit (400).
    """
    response_format = request.get("response_format") or {}
    if not isinstance(response_format, dict):
        raise InvalidGrammarError("response_format must be an object")
    rf_type = response_format.get("type")

    json_schema = None
    structural_tag = None
    if rf_type == "json_schema":
        json_schema = (response_format.get("json_schema") or {}).get("schema")
        if json_schema is None:
            raise InvalidGrammarError("response_format json_schema requires json_schema.schema")
    elif rf_type == "json_object":
        return {"type": "json_object", "value": None}
    if rf_type == "structural_tag":
        structural_tag = response_format

    if json_schema is not None:
        spec = {"type": "json_schema", "value": json_schema}
    elif request.get("regex") is not None:
        spec = {"type": "regex", "value": request["regex"]}
    elif request.get("ebnf") is not None:
        spec = {"type": "ebnf", "value": request["ebnf"]}
    elif structural_tag is not None:
        spec = {"type": "structural_tag", "value": structural_tag}
    else:
        return None
    validate_grammar_spec(spec)
    return spec
