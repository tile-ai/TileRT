from __future__ import annotations

from typing import Any


class GrammarError(Exception):
    error_type = "grammar_error"
    http_status = 500

    def to_payload(self) -> dict[str, str]:
        return {"error": str(self), "error_type": self.error_type}


class InvalidGrammarError(GrammarError):
    error_type = "invalid_grammar"
    http_status = 400


class GrammarBackendUnavailable(GrammarError):
    error_type = "grammar_backend_unavailable"
    http_status = 500


class GrammarViolationError(GrammarError):
    error_type = "grammar_violation"
    http_status = 400


class GrammarUnsupported(GrammarError):
    error_type = "grammar_unsupported"
    http_status = 501


_MAX_SCHEMA_DEPTH = 64
_MAX_SCHEMA_NODES = 1000
_MAX_ENUM_VALUES = 10000


def _walk_schema(node, depth, counts, path):
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
        if key in ("enum", "const", "examples", "default"):
            continue
        if isinstance(child, (dict, list)):
            _walk_schema(child, depth + 1, counts, f"{path}.{key}" if path else key)


def validate_grammar_spec(spec: dict[str, Any] | None) -> None:
    if spec:
        _walk_schema(spec.get("value"), 0, {"nodes": 0, "enums": 0}, "")


_SPEC_TYPES = ("json_schema", "json_object", "ebnf", "regex", "structural_tag")


def extract_request_grammar_spec(request: dict[str, Any]) -> dict[str, Any] | None:
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
    elif rf_type == "structural_tag":
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
