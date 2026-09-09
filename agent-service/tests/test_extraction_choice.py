"""Guards on the extraction schema handed to a tool-calling provider.

`AnthropicLLM.extract()` once passed a bare
`Union[ExtractionDirect, ExtractionMultiSpan, ExtractionInsufficient]` to
`with_structured_output()`. A Union is not a schema, so the call raised
`TypeError: ... is not a module, class, method, or function` before any
request left the process -- and only on text/table questions, since the
numerical branch passes a real model. That made every non-numerical
question fail in a live evaluation run while the mock provider stayed
green. These tests pin both halves of the fix.
"""
from __future__ import annotations

from typing import Union

import pytest

from app.llm import (
    ExtractionChoice,
    ExtractionDirect,
    ExtractionInsufficient,
    ExtractionMultiSpan,
)

convert_to_openai_tool = pytest.importorskip(
    "langchain_core.utils.function_calling"
).convert_to_openai_tool


def test_bare_union_is_still_rejected_by_tool_conversion():
    """The reason ExtractionChoice exists. If this ever stops raising, the
    flat model is no longer load-bearing -- but until then, reverting to a
    Union reintroduces the exact production failure."""
    union = Union[ExtractionDirect, ExtractionMultiSpan, ExtractionInsufficient]
    with pytest.raises(TypeError):
        convert_to_openai_tool(union)


def test_extraction_choice_converts_to_a_tool_schema():
    tool = convert_to_openai_tool(ExtractionChoice)
    assert tool["function"]["name"] == "ExtractionChoice"


def test_schema_is_flat_so_providers_need_not_support_nested_oneof():
    schema = ExtractionChoice.model_json_schema()
    assert not schema.get("$defs"), "nested model refs defeat the point of flattening"
    assert set(schema["properties"]) == {"shape", "value", "values", "reason"}


@pytest.mark.parametrize(
    "payload,expected",
    [
        ({"shape": "direct", "value": "3,875"}, ExtractionDirect),
        ({"shape": "multi_span", "values": ["a", "b"]}, ExtractionMultiSpan),
        ({"shape": "insufficient", "reason": "not in evidence"}, ExtractionInsufficient),
    ],
)
def test_narrowing_returns_the_types_app_graph_branches_on(payload, expected):
    assert isinstance(ExtractionChoice(**payload).as_extraction(), expected)


@pytest.mark.parametrize(
    "payload",
    [
        {"shape": "direct", "value": None},
        {"shape": "multi_span", "values": []},
        {"shape": "multi_span", "values": None},
    ],
)
def test_a_declared_shape_with_no_payload_degrades_to_insufficient(payload):
    """Better an honest 'insufficient' than a Direct carrying value=None,
    which would reach validate_answer_dict as a malformed answer."""
    assert isinstance(ExtractionChoice(**payload).as_extraction(), ExtractionInsufficient)


def test_values_are_coerced_to_str():
    got = ExtractionChoice(shape="multi_span", values=["1", "2"]).as_extraction()
    assert got.values == ["1", "2"]


def test_unknown_shape_is_rejected_at_validation_time():
    with pytest.raises(Exception):
        ExtractionChoice(shape="calculated", value="x")
