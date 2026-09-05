import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.validator import validate_answer  # noqa: E402

DIRECT_OK = {
    "answer_type": "direct",
    "evidence": [{"document_id": "doc_017", "page": 1, "section": "Income Statement"}],
    "params": {"value": "$142.5M"},
}

CALCULATED_OK = {
    "answer_type": "calculated",
    "evidence": [
        {"document_id": "doc_041", "page": 2, "section": "Operating Expenses"},
        {"document_id": "doc_041", "page": 2, "section": "Operating Expenses"},
    ],
    "params": {"value": 13.4, "formula": "(3875-3410)/3410*100"},
}

MULTI_SPAN_OK = {
    "answer_type": "multi_span",
    "evidence": [{"document_id": "doc_022", "page": 3, "section": "Operating Expenses"}],
    "params": {"values": ["Marketing", "R&D", "Logistics"]},
}

INSUFFICIENT_OK = {
    "answer_type": "insufficient_evidence",
    "evidence": [],
    "params": {"reason": "No document in the indexed corpus reports restructuring expenses."},
}


def test_direct_example_is_valid():
    out = validate_answer(DIRECT_OK)
    assert out.valid is True
    assert out.reason is None
    assert "SUCCESS" in out.log_line
    assert "'direct'" in out.log_line


def test_calculated_example_is_valid():
    out = validate_answer(CALCULATED_OK)
    assert out.valid is True
    assert "'calculated'" in out.log_line


def test_multi_span_example_is_valid():
    out = validate_answer(MULTI_SPAN_OK)
    assert out.valid is True


def test_insufficient_evidence_example_is_valid():
    out = validate_answer(INSUFFICIENT_OK)
    assert out.valid is True


def test_calculated_missing_formula_is_rejected():
    bad = {**CALCULATED_OK, "params": {"value": 13.4}}
    out = validate_answer(bad)
    assert out.valid is False
    assert out.log_line == (
        "[ANSWER-VALIDATOR-ERROR] Invalid answer for 'calculated': "
        "Missing required key 'formula'."
    )


def test_missing_evidence_is_rejected_generically():
    bad = {**DIRECT_OK, "evidence": []}
    out = validate_answer(bad)
    assert out.valid is False
    assert out.log_line == (
        "[ANSWER-VALIDATOR-ERROR] Invalid answer. Reason: Missing required evidence citation."
    )


def test_unknown_answer_type_is_rejected():
    bad = {**DIRECT_OK, "answer_type": "guess"}
    out = validate_answer(bad)
    assert out.valid is False
    assert "Unknown answer_type" in out.reason


def test_extra_top_level_key_is_rejected():
    bad = {**DIRECT_OK, "confidence": 0.9}
    out = validate_answer(bad)
    assert out.valid is False


def test_extra_params_key_is_rejected():
    bad = {**DIRECT_OK, "params": {"value": "$142.5M", "currency": "USD"}}
    out = validate_answer(bad)
    assert out.valid is False


def test_multi_span_requires_non_empty_values():
    bad = {**MULTI_SPAN_OK, "params": {"values": []}}
    out = validate_answer(bad)
    assert out.valid is False


def test_calculated_with_two_operands_needs_two_citations():
    bad = {**CALCULATED_OK, "evidence": [CALCULATED_OK["evidence"][0]]}
    out = validate_answer(bad)
    assert out.valid is False


def test_malformed_evidence_missing_page():
    bad = {**DIRECT_OK, "evidence": [{"document_id": "doc_017"}]}
    out = validate_answer(bad)
    assert out.valid is False


def test_insufficient_evidence_requires_reason():
    bad = {**INSUFFICIENT_OK, "params": {}}
    out = validate_answer(bad)
    assert out.valid is False
    assert "reason" in out.log_line
