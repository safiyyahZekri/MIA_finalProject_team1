import os
import uuid
from datetime import datetime, timezone
from urllib.parse import quote

import requests

ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://localhost:8000")

# MOCK_MODE lets the UI run and be demoed before orchestrator-api exists.
# Flip to "false" once orchestrator-api is up (see README).
MOCK_MODE = os.getenv("MOCK_MODE", "true").lower() == "true"

TIMEOUT = 15
_MOCK_REVIEWS: list[dict] = []
_MOCK_CORRECTIONS: list[dict] = []


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


def ingest_pdf(file_path: str) -> dict:
    with open(file_path, "rb") as pdf:
        resp = requests.post(
            f"{ORCHESTRATOR_URL}/documents/ingest",
            files={"file": (os.path.basename(file_path), pdf, "application/pdf")},
            timeout=300,
        )
    resp.raise_for_status()
    return resp.json()


def download_document(document_id: str) -> bytes:
    resp = requests.get(
        f"{ORCHESTRATOR_URL}/documents/{quote(document_id, safe='')}/file",
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.content


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


def submit_review(
    question: str,
    original_answer: dict,
    verdict: str,
    corrected_answer: str | None = None,
    comment: str | None = None,
    document_id: str | None = None,
) -> dict:
    payload = {
        "question": question,
        "original_answer": original_answer,
        "verdict": verdict,
        "corrected_answer": corrected_answer,
        "comment": comment,
        "document_id": document_id,
    }
    if MOCK_MODE:
        record = {
            **payload,
            "review_id": str(uuid.uuid4()),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "recorded",
        }
        _MOCK_REVIEWS.append(record)
        return record
    resp = requests.post(f"{ORCHESTRATOR_URL}/reviews", json=payload, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def list_reviews(limit: int = 20) -> list[dict]:
    if MOCK_MODE:
        return list(reversed(_MOCK_REVIEWS[-limit:]))
    resp = requests.get(
        f"{ORCHESTRATOR_URL}/reviews",
        params={"limit": limit},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def get_extracted_fields(document_id: str) -> list[dict]:
    if MOCK_MODE:
        return [
            {
                "chunk_id": "mock-table-1",
                "document_id": document_id,
                "page": 1,
                "section": "Income Statement",
                "content_type": "table",
                "content": "Metric | 2022\nOperating income | 14Z.5 million",
                "bbox": [80, 120, 920, 430],
                "source_block_ids": ["mock-block-1"],
            }
        ]
    encoded = quote(document_id, safe="")
    resp = requests.get(
        f"{ORCHESTRATOR_URL}/documents/{encoded}/chunks", timeout=TIMEOUT
    )
    resp.raise_for_status()
    return resp.json()


def submit_extraction_correction(
    document_id: str,
    chunk_id: str,
    corrected_text: str,
    corrected_by: str,
    comment: str | None = None,
) -> dict:
    payload = {
        "chunk_id": chunk_id,
        "corrected_text": corrected_text,
        "corrected_by": corrected_by,
        "comment": comment,
    }
    if MOCK_MODE:
        original = next(
            item["content"]
            for item in get_extracted_fields(document_id)
            if item["chunk_id"] == chunk_id
        )
        record = {
            **payload,
            "correction_id": str(uuid.uuid4()),
            "document_id": document_id,
            "original_text": original,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "applied",
        }
        _MOCK_CORRECTIONS.append(record)
        return record
    encoded = quote(document_id, safe="")
    resp = requests.post(
        f"{ORCHESTRATOR_URL}/documents/{encoded}/corrections",
        json=payload,
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()


def list_extraction_corrections(document_id: str) -> list[dict]:
    if MOCK_MODE:
        return list(
            reversed(
                [
                    item
                    for item in _MOCK_CORRECTIONS
                    if item["document_id"] == document_id
                ]
            )
        )
    encoded = quote(document_id, safe="")
    resp = requests.get(
        f"{ORCHESTRATOR_URL}/documents/{encoded}/corrections", timeout=TIMEOUT
    )
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
