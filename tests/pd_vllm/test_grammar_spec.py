"""Stage-1 unit tests: request -> grammar_spec translation + error hierarchy +
StubEngine classification. Pure CPU, no GPU / no vLLM / no tilert.

  CUDA_VISIBLE_DEVICES= python -m pytest tests/pd_vllm/test_grammar_spec.py -v
"""

import pytest

from tilert.pd_vllm.engine_iface import StubEngine
from tilert.pd_vllm.grammar_spec import (
    GrammarBackendUnavailable,
    GrammarViolationError,
    InvalidGrammarError,
    extract_request_grammar_spec,
)


# --------------------------- spec extraction ------------------------------- #
def test_regex_extension_field():
    assert extract_request_grammar_spec({"regex": r"[0-9]{3}"}) == {
        "type": "regex",
        "value": r"[0-9]{3}",
    }


def test_ebnf_extension_field():
    spec = extract_request_grammar_spec({"ebnf": 'root ::= "hi"'})
    assert spec == {"type": "ebnf", "value": 'root ::= "hi"'}


def test_json_object():
    spec = extract_request_grammar_spec({"response_format": {"type": "json_object"}})
    assert spec == {"type": "json_object", "value": None}


def test_json_schema():
    schema = {"type": "object", "properties": {"n": {"type": "integer"}}}
    spec = extract_request_grammar_spec(
        {"response_format": {"type": "json_schema", "json_schema": {"schema": schema}}}
    )
    assert spec == {"type": "json_schema", "value": schema}


def test_structural_tag():
    rf = {"type": "structural_tag", "structures": [], "triggers": []}
    spec = extract_request_grammar_spec({"response_format": rf})
    assert spec == {"type": "structural_tag", "value": rf}


def test_priority_json_schema_over_regex_and_ebnf():
    req = {
        "regex": r"[0-9]+",
        "ebnf": 'root ::= "x"',
        "response_format": {"type": "json_schema", "json_schema": {"schema": {"type": "object"}}},
    }
    assert extract_request_grammar_spec(req)["type"] == "json_schema"


def test_priority_regex_over_ebnf():
    assert (
        extract_request_grammar_spec({"regex": r"[0-9]+", "ebnf": 'root ::= "x"'})["type"]
        == "regex"
    )


def test_json_object_returns_before_regex():
    # json_object short-circuits even when regex is also present.
    req = {"regex": r"[0-9]+", "response_format": {"type": "json_object"}}
    assert extract_request_grammar_spec(req)["type"] == "json_object"


def test_none_when_unconstrained():
    assert extract_request_grammar_spec({"messages": [], "temperature": 0.7}) is None


def test_json_schema_missing_schema_raises_invalid():
    with pytest.raises(InvalidGrammarError):
        extract_request_grammar_spec(
            {"response_format": {"type": "json_schema", "json_schema": {}}}
        )


def test_response_format_not_object_raises_invalid():
    with pytest.raises(InvalidGrammarError):
        extract_request_grammar_spec({"response_format": "json"})


# --------------------------- error hierarchy ------------------------------- #
def test_error_status_and_payload():
    assert InvalidGrammarError("x").http_status == 400
    assert GrammarBackendUnavailable("x").http_status == 500
    assert GrammarViolationError("x").http_status == 400
    p = InvalidGrammarError("bad schema").to_payload()
    assert p == {"error": "bad schema", "error_type": "invalid_grammar"}


# --------------------------- StubEngine classification --------------------- #
def test_stub_prepare_none():
    assert StubEngine().prepare_grammar(None) is None


def test_stub_prepare_valid_returns_session():
    sess = StubEngine().prepare_grammar({"type": "regex", "value": "[0-9]"}, enable_thinking=False)
    assert sess["spec"]["type"] == "regex" and sess["enable_thinking"] is False


def test_stub_prepare_unknown_type_invalid():
    with pytest.raises(InvalidGrammarError):
        StubEngine().prepare_grammar({"type": "nope"})


def test_stub_prepare_not_a_dict_invalid():
    with pytest.raises(InvalidGrammarError):
        StubEngine().prepare_grammar(["not", "a", "dict"])


def test_stub_prepare_backend_missing_500():
    with pytest.raises(GrammarBackendUnavailable):
        StubEngine().prepare_grammar({"type": "__backend_missing__"})


def test_stub_decode_violation_raises():
    eng = StubEngine()
    sess = eng.prepare_grammar({"type": "regex", "value": "__violate__"})
    with pytest.raises(GrammarViolationError):
        eng.decode(first_token_id=7, max_tokens=8, sampling=None, grammar_session=sess)


def test_stub_decode_threads_session_ok():
    eng = StubEngine()
    sess = eng.prepare_grammar({"type": "regex", "value": "[0-9]"})
    out = eng.decode(first_token_id=7, max_tokens=8, sampling=None, grammar_session=sess)
    assert out[0] == 7 and eng.last_stats["finish_reason"] == "stop"


# --------------------- spec validation: compile-cost caps ------------------ #
# Compilation runs inside the decode node's single-slot lock, so an unbounded
# schema is downtime for everyone, not just a slow request.
def _rf(schema):
    return {
        "response_format": {"type": "json_schema", "json_schema": {"name": "t", "schema": schema}}
    }


def test_deep_nesting_rejected():
    schema = {"type": "string"}
    for _ in range(200):
        schema = {"type": "object", "properties": {"x": schema}}
    with pytest.raises(InvalidGrammarError, match="deeper than"):
        extract_request_grammar_spec(_rf(schema))


def test_many_properties_rejected():
    schema = {"type": "object", "properties": {f"p{i}": {"type": "string"} for i in range(5000)}}
    with pytest.raises(InvalidGrammarError, match="subschemas"):
        extract_request_grammar_spec(_rf(schema))


def test_huge_enum_rejected():
    schema = {"type": "object", "properties": {"v": {"enum": [f"o{i}" for i in range(50000)]}}}
    with pytest.raises(InvalidGrammarError, match="enum"):
        extract_request_grammar_spec(_rf(schema))


def test_large_enum_does_not_count_as_nodes():
    """Enum entries are literal data, not subschemas."""
    schema = {"type": "object", "properties": {"v": {"enum": [f"o{i}" for i in range(900)]}}}
    assert extract_request_grammar_spec(_rf(schema))["type"] == "json_schema"


def test_long_regex_allowed():
    """regex/EBNF are not capped: 200 000 chars compiles in 0.7s, so a length
    limit would only over-block.
    """
    spec = extract_request_grammar_spec({"regex": "a" * 50000})
    assert spec == {"type": "regex", "value": "a" * 50000}


def test_structural_tag_schemas_are_walked():
    """The walk follows spec['value'] whatever its shape, so a tag's schemas are
    covered by the same caps without a per-type branch.
    """
    deep = {"type": "string"}
    for _ in range(200):
        deep = {"type": "object", "properties": {"x": deep}}
    with pytest.raises(InvalidGrammarError, match="deeper than"):
        extract_request_grammar_spec(
            {
                "response_format": {
                    "type": "structural_tag",
                    "structures": [{"begin": "<t>", "schema": deep, "end": "</t>"}],
                }
            }
        )


def test_ordinary_schema_unaffected():
    schema = {
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "temp_c": {"type": "integer", "minimum": -90, "maximum": 60},
            "forecast": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"day": {"type": "string"}, "high": {"type": "integer"}},
                    "required": ["day", "high"],
                },
            },
        },
        "required": ["city", "temp_c"],
        "additionalProperties": False,
    }
    assert extract_request_grammar_spec(_rf(schema))["value"] == schema
