"""The chat says whether an answer came from the answer cache or the agent, and
the dashboard marks which recent queries the cache answered (semantic caching
bonus: its effect has to be visible to count)."""

import app as ui_app

ANSWER = {
    "answer_type": "direct",
    "params": {"value": "$42 million"},
    "evidence": [{"document_id": "report", "page": 1}],
    "evidence_boxes": [],
}


def test_a_cached_answer_says_it_came_from_the_cache(monkeypatch) -> None:
    monkeypatch.setattr(
        ui_app.api_client,
        "ask",
        lambda question, document_id: {**ANSWER, "cache_hit": True, "cache_match": "reworded"},
    )
    _, history, _, _, _ = ui_app.answer_with_evidence("What was revenue?", [], "")
    assert "answer cache (a reworded question was asked before)" in history[-1]["content"]
    assert "without running the agent" in history[-1]["content"]


def test_an_agent_answer_shows_its_latency(monkeypatch) -> None:
    monkeypatch.setattr(ui_app.api_client, "ask", lambda question, document_id: dict(ANSWER))
    _, history, _, _, _ = ui_app.answer_with_evidence("What was revenue?", [], "")
    assert "Answered by the agent in" in history[-1]["content"]
    assert "answer cache" not in history[-1]["content"]


def test_the_dashboard_marks_queries_the_cache_answered(monkeypatch) -> None:
    monkeypatch.setattr(ui_app.api_client, "list_documents", lambda: [])
    monkeypatch.setattr(ui_app.api_client, "list_reviews", lambda: [])
    monkeypatch.setattr(
        ui_app.api_client,
        "recent_queries",
        lambda: [
            {"question": "q", "latency_ms": 3.0, "valid": True, "cache": "exact"},
            {"question": "q2", "latency_ms": 900.0, "valid": True, "cache": None},
        ],
    )
    _, query_rows, _ = ui_app.load_dashboard()
    assert query_rows == [["q", 3.0, True, "exact"], ["q2", 900.0, True, ""]]
