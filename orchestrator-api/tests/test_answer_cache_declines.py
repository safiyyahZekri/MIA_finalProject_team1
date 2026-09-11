"""A validated decline is not cached, so asking again runs the agent again."""

import pytest
from fastapi.testclient import TestClient

from app import clients, main
from app.semantic_cache import SemanticCache

QUESTION = "What was Jabil's total net revenue in 2019?"
ANSWERED = {
    "answer_type": "direct",
    "params": {"value": "25,282,320"},
    "evidence": [{"document_id": "jabil", "page": 1, "section": "Revenue"}],
}
DECLINED = {
    "answer_type": "insufficient_evidence",
    "params": {"reason": "No passage reports Jabil's 2019 net revenue."},
    "evidence": [],
}


@pytest.fixture
def agent(monkeypatch):
    state = {"answers": [], "calls": 0}

    async def ask_agent(question, document_id=None):
        state["calls"] += 1
        return state["answers"].pop(0)

    async def validate_answer(answer):
        return {"valid": True, "reason": None}

    async def evidence_boxes(evidence):
        return []

    monkeypatch.setattr(clients, "MOCK_MODE", False)
    monkeypatch.setattr(clients, "ask_agent", ask_agent)
    monkeypatch.setattr(clients, "validate_answer", validate_answer)
    monkeypatch.setattr(clients, "evidence_boxes", evidence_boxes)
    monkeypatch.setattr(main, "ASK_CACHE_ENABLED", True)
    monkeypatch.setattr(main, "answer_cache", SemanticCache())
    return state


def _ask(question):
    response = TestClient(main.app).post("/ask", json={"question": question, "document_id": None})
    assert response.status_code == 200, response.text
    return response.json()


def test_a_declined_question_runs_the_agent_again_and_its_answer_is_cached(agent):
    agent["answers"] = [dict(DECLINED), dict(ANSWERED)]

    first = _ask(QUESTION)
    second = _ask(QUESTION)
    third = _ask(QUESTION)

    assert first["answer_type"] == "insufficient_evidence" and first["cache_hit"] is False
    assert second["answer_type"] == "direct" and second["cache_hit"] is False
    assert third["answer_type"] == "direct" and third["cache_hit"] is True
    assert agent["calls"] == 2
