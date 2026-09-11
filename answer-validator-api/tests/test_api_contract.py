"""
Confirms the exact contract orchestrator-api relies on:

    POST /validate_answer  <-  raw answer object {"answer_type", "evidence",
                                "params"}, unchanged, no wrapper
    <-  {"valid": bool, "reason": string | null}

This is an API-level test (via FastAPI's TestClient) rather than a call
straight into `validate_answer`, so it also exercises request parsing and
response serialization, not just the validation logic.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

client = TestClient(app)


def test_valid_answer_returns_exact_contract_shape():
    response = client.post(
        "/validate_answer",
        json={
            "answer_type": "direct",
            "evidence": [{"document_id": "doc_017", "page": 1, "section": "Income Statement"}],
            "params": {"value": "$142.5M"},
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"valid", "reason"}
    assert body == {"valid": True, "reason": None}


def test_invalid_answer_returns_exact_contract_shape_with_reason():
    response = client.post(
        "/validate_answer",
        json={
            "answer_type": "calculated",
            "evidence": [{"document_id": "doc_041", "page": 2}],
            "params": {"value": 13.4},  # missing required 'formula'
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"valid", "reason"}
    assert body["valid"] is False
    assert isinstance(body["reason"], str)


def test_request_body_is_forwarded_unchanged_no_wrapper_expected():
    """orchestrator-api sends the raw answer object with no wrapper key
    (not {"answer": {...}}) — confirm the endpoint accepts that shape
    directly rather than requiring nesting."""
    raw_answer = {
        "answer_type": "insufficient_evidence",
        "evidence": [],
        "params": {"reason": "No document in the indexed corpus covers this metric."},
    }
    response = client.post("/validate_answer", json=raw_answer)
    assert response.status_code == 200
    assert response.json() == {"valid": True, "reason": None}
