"""Highlight boxes travel beside the answer, never inside its citations.

The Strict Answer Schema allows only document_id, page and section per
citation, and answer-validator-api rejects anything else. /ask looks the
cited pages' boxes up from retrieval-api and returns them as evidence_boxes.
"""

import asyncio

from fastapi.testclient import TestClient

from app import clients
from app.main import app


def _chunk(chunk_id, page, section, bbox):
    return {
        "chunk_id": chunk_id,
        "document_id": "report",
        "page": page,
        "section": section,
        "content_type": "table",
        "content": "Revenue | 42",
        "bbox": bbox,
        "source_block_ids": [chunk_id],
    }


CHUNKS = [
    _chunk("revenue", 2, "Revenue", [10, 20, 300, 120]),
    # A long table is split into parts that share the table's box.
    _chunk("revenue-part-2", 2, "Revenue", [10, 20, 300, 120]),
    _chunk("costs", 2, "Costs", [10, 200, 300, 260]),
    _chunk("revenue-next-page", 3, "Revenue", [10, 20, 300, 120]),
]


def _serve_chunks(monkeypatch, calls):
    async def get_extracted_fields(document_id):
        calls.append(document_id)
        return CHUNKS

    monkeypatch.setattr(clients, "MOCK_MODE", False)
    monkeypatch.setattr(clients, "get_extracted_fields", get_extracted_fields)


def test_boxes_come_from_the_cited_page_and_section(monkeypatch):
    calls = []
    _serve_chunks(monkeypatch, calls)
    citation = {"document_id": "report", "page": 2, "section": "Revenue"}

    boxes = asyncio.run(clients.evidence_boxes([citation, citation]))

    assert boxes == [{"document_id": "report", "page": 2, "section": "Revenue", "bbox": [10, 20, 300, 120]}]
    assert calls == ["report"]


def test_a_citation_without_a_section_takes_every_box_on_its_page(monkeypatch):
    _serve_chunks(monkeypatch, [])

    boxes = asyncio.run(clients.evidence_boxes([{"document_id": "report", "page": 2, "section": None}]))

    assert [box["section"] for box in boxes] == ["Revenue", "Costs"]


def test_a_failed_lookup_leaves_the_answer_without_boxes(monkeypatch):
    async def unavailable(document_id):
        raise clients.ServiceIntegrationError("retrieval_correction", "retrieval-api is down", 503)

    monkeypatch.setattr(clients, "MOCK_MODE", False)
    monkeypatch.setattr(clients, "get_extracted_fields", unavailable)

    assert asyncio.run(clients.evidence_boxes([{"document_id": "report", "page": 2}])) == []


def test_ask_returns_strict_citations_and_separate_boxes(monkeypatch):
    _serve_chunks(monkeypatch, [])
    citation = {"document_id": "report", "page": 2, "section": "Revenue"}

    async def ask_agent(question, document_id=None):
        return {"answer_type": "direct", "evidence": [citation], "params": {"value": "$42 million"}}

    async def validate_answer(answer):
        assert all(set(item) <= {"document_id", "page", "section"} for item in answer["evidence"])
        return {"valid": True, "reason": None}

    monkeypatch.setattr(clients, "ask_agent", ask_agent)
    monkeypatch.setattr(clients, "validate_answer", validate_answer)

    response = TestClient(app).post("/ask", json={"question": "What was revenue?"})

    assert response.status_code == 200
    body = response.json()
    assert body["evidence"] == [citation]
    assert body["evidence_boxes"] == [
        {"document_id": "report", "page": 2, "section": "Revenue", "bbox": [10, 20, 300, 120]}
    ]
