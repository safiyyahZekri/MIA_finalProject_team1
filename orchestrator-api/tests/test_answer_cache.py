"""/ask answers a repeated or reworded question from the answer cache.

Every agent call is counted, so a cache hit is shown by the agent not being
called again, not inferred from latency.
"""

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from app import clients, document_store, main
from app.semantic_cache import SemanticCache, content_key

QUESTION = "What was Jabil's total net revenue in 2019?"


@pytest.fixture
def agent_calls(monkeypatch):
    calls = []

    async def ask_agent(question, document_id=None):
        calls.append((question, document_id))
        return {
            "answer_type": "direct",
            "params": {"value": "25,282,320"},
            "evidence": [{"document_id": "jabil", "page": 1, "section": "Revenue"}],
        }

    async def validate_answer(answer):
        return {"valid": True, "reason": None}

    async def evidence_boxes(evidence):
        return []

    monkeypatch.setattr(clients, "MOCK_MODE", False)
    monkeypatch.setattr(clients, "ask_agent", ask_agent)
    monkeypatch.setattr(clients, "validate_answer", validate_answer)
    monkeypatch.setattr(clients, "evidence_boxes", evidence_boxes)
    monkeypatch.setattr(main, "ASK_CACHE_ENABLED", True)
    # A fresh cache per test, so hit counts do not carry over between tests.
    monkeypatch.setattr(main, "answer_cache", SemanticCache())
    return calls


def _ask(question, document_id=None):
    response = TestClient(main.app).post(
        "/ask", json={"question": question, "document_id": document_id}
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_a_repeated_question_does_not_run_the_agent_again(agent_calls):
    first = _ask(QUESTION)
    second = _ask("what was jabil's total net revenue in 2019")

    assert len(agent_calls) == 1
    assert first["cache_hit"] is False and first["cache_match"] is None
    assert second["cache_hit"] is True and second["cache_match"] == "exact"
    assert second["params"] == first["params"]
    assert second["evidence"] == first["evidence"]


def test_a_reworded_question_is_answered_from_the_cache(agent_calls):
    _ask(QUESTION)
    reworded = _ask("Jabil total net revenue in 2019 -- what was it?")

    assert len(agent_calls) == 1
    assert reworded["cache_match"] == "reworded"


@pytest.mark.parametrize(
    "question, document_id",
    [
        ("What was Jabil's total net revenue in 2018?", None),  # another year
        ("What was Flex's total net revenue in 2019?", None),  # another company
        ("What was Jabil's total net sales in 2019?", None),  # a near-synonym
        ("What was Jabil's total net revenue in 2019 and 2018?", None),  # more asked
        (QUESTION, "another-report"),  # another document
    ],
)
def test_a_question_asking_for_something_else_runs_the_agent(agent_calls, question, document_id):
    _ask(QUESTION)
    other = _ask(question, document_id)

    assert len(agent_calls) == 2
    assert other["cache_hit"] is False


def test_a_rejected_answer_is_not_cached(agent_calls, monkeypatch):
    async def reject(answer):
        return {"valid": False, "reason": "evidence is missing"}

    monkeypatch.setattr(clients, "validate_answer", reject)
    client = TestClient(main.app)
    for _ in range(2):
        assert client.post("/ask", json={"question": QUESTION}).status_code == 422

    assert len(agent_calls) == 2


def test_an_extraction_correction_clears_the_cache(agent_calls, monkeypatch):
    async def apply_extraction_correction(document_id, correction):
        return {
            **correction,
            "correction_id": "c1",
            "document_id": document_id,
            "original_text": "Revenue | 25,282,32O",
            "created_at": datetime.now(timezone.utc),
        }

    monkeypatch.setattr(clients, "apply_extraction_correction", apply_extraction_correction)
    _ask(QUESTION)
    response = TestClient(main.app).post(
        "/documents/jabil/corrections",
        json={"chunk_id": "revenue", "corrected_text": "Revenue | 25,282,320", "corrected_by": "reviewer"},
    )
    assert response.status_code == 200, response.text

    again = _ask(QUESTION)

    assert len(agent_calls) == 2
    assert again["cache_hit"] is False


def test_ingesting_a_document_clears_the_cache(agent_calls, monkeypatch):
    async def process_pdf(content, filename, document_id, source_doc_uid):
        return {"pages": [{}]}

    async def index_processed_document(processed, filename, document_id, source_doc_uid, metadata):
        return {"chunks_indexed": 3}

    monkeypatch.setattr(clients, "process_pdf", process_pdf)
    monkeypatch.setattr(clients, "index_processed_document", index_processed_document)
    monkeypatch.setattr(document_store, "save_pdf", lambda document_id, content: None)
    _ask(QUESTION)
    response = TestClient(main.app).post(
        "/documents/ingest", files={"file": ("report.pdf", b"%PDF-1.4 report", "application/pdf")}
    )
    assert response.status_code == 200, response.text

    _ask(QUESTION)

    assert len(agent_calls) == 2


def test_the_cache_can_be_switched_off(agent_calls, monkeypatch):
    monkeypatch.setattr(main, "ASK_CACHE_ENABLED", False)

    _ask(QUESTION)
    second = _ask(QUESTION)

    assert len(agent_calls) == 2
    assert second["cache_hit"] is False


def test_health_reports_the_cache_counts(agent_calls):
    _ask(QUESTION)
    _ask(QUESTION)
    _ask("Jabil total net revenue in 2019 -- what was it?")

    cache = TestClient(main.app).get("/health").json()["answer_cache"]

    assert cache == {"enabled": True, "entries": 1, "exact_hits": 1, "reworded_hits": 1, "misses": 1}


def test_the_oldest_entry_is_evicted_first():
    cache = SemanticCache(max_entries=2)
    for year in ("2017", "2018", "2019"):
        cache.put(f"Jabil revenue in {year}", None, year)

    assert cache.get("Jabil revenue in 2017", None) is None
    assert cache.get("Jabil revenue in 2019", None) == ("2019", "exact")


def test_numbers_keep_their_decimal_point_but_not_their_commas():
    assert content_key("growth of 1.25%") != content_key("growth of 12.5%")
    assert content_key("revenue of 3,875") == content_key("3875 revenue")
