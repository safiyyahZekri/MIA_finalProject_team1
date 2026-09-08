import os

import requests

ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://localhost:8000")

# MOCK_MODE lets the UI run and be demoed before orchestrator-api exists.
# Flip to "false" once orchestrator-api is up (see README).
MOCK_MODE = os.getenv("MOCK_MODE", "true").lower() == "true"

TIMEOUT = 15


def ask(question: str, document_id: str | None = None) -> dict:
    if MOCK_MODE:
        return _mock_ask(question)
    resp = requests.post(
        f"{ORCHESTRATOR_URL}/ask",
        json={"question": question, "document_id": document_id},
        timeout=TIMEOUT,
    )
    if resp.status_code == 422:
        return {
            "answer_type": "insufficient_evidence",
            "evidence": [],
            "params": {"reason": resp.json()["detail"]},
            "valid": False,
        }
    resp.raise_for_status()
    return resp.json()


def list_documents() -> list[dict]:
    if MOCK_MODE:
        return _mock_documents()
    resp = requests.get(f"{ORCHESTRATOR_URL}/documents", timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def recent_queries() -> list[dict]:
    if MOCK_MODE:
        return _mock_recent_queries()
    resp = requests.get(f"{ORCHESTRATOR_URL}/recent_queries", timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def _mock_ask(question: str) -> dict:
    q = question.lower()
    if "restructuring" in q:
        return {
            "answer_type": "insufficient_evidence",
            "evidence": [],
            "params": {"reason": "No document in the indexed corpus reports restructuring expenses."},
            "valid": True,
        }
    if "%" in q or "change" in q:
        return {
            "answer_type": "calculated",
            "evidence": [{"document_id": "doc_041", "page": 2, "section": "Operating Expenses"}],
            "params": {"value": 13.4, "formula": "(3875-3410)/3410*100"},
            "valid": True,
        }
    if "which" in q or "how many" in q:
        return {
            "answer_type": "multi_span",
            "evidence": [{"document_id": "doc_022", "page": 3, "section": "Operating Expenses"}],
            "params": {"values": ["Marketing", "R&D", "Logistics"]},
            "valid": True,
        }
    return {
        "answer_type": "direct",
        "evidence": [{"document_id": "doc_017", "page": 1, "section": "Income Statement"}],
        "params": {"value": "$142.5M"},
        "valid": True,
    }


def _mock_documents() -> list[dict]:
    return [
        {"document_id": "doc_017", "filename": "CTS Corp 10-K excerpt", "pages": 3, "tables": 2},
        {"document_id": "doc_041", "filename": "Jabil Inc Annual Report excerpt", "pages": 2, "tables": 1},
        {"document_id": "doc_022", "filename": "AMCON Distributing 2019", "pages": 4, "tables": 3},
    ]


def _mock_recent_queries() -> list[dict]:
    return [
        {"question": "What was the operating income in 2020?", "latency_ms": 842.3, "valid": True},
        {"question": "What is CTS's restructuring expense?", "latency_ms": 610.1, "valid": True},
    ]
