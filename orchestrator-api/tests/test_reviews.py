from fastapi.testclient import TestClient

from app.main import app


ANSWER = {
    "answer_type": "direct",
    "params": {"value": "$42 million"},
    "evidence": [{"document_id": "report", "page": 2}],
    "valid": True,
}


def test_human_review_is_persisted_and_listed_newest_first(monkeypatch, tmp_path):
    monkeypatch.setenv("REVIEW_STORE_PATH", str(tmp_path / "reviews.jsonl"))
    client = TestClient(app)

    first = client.post(
        "/reviews",
        json={
            "question": "What was revenue?",
            "verdict": "correct",
            "original_answer": ANSWER,
            "document_id": "report",
        },
    )
    second = client.post(
        "/reviews",
        json={
            "question": "What was operating income?",
            "verdict": "needs_correction",
            "original_answer": ANSWER,
            "corrected_answer": "$21 million",
            "comment": "The cited row reports operating income.",
        },
    )
    listed = client.get("/reviews", params={"limit": 2})

    assert first.status_code == second.status_code == listed.status_code == 200
    assert first.json()["status"] == "recorded"
    assert listed.json()[0]["review_id"] == second.json()["review_id"]
    assert listed.json()[1]["review_id"] == first.json()["review_id"]
    assert (tmp_path / "reviews.jsonl").read_text().count("\n") == 2


def test_incorrect_verdict_requires_a_human_correction(monkeypatch, tmp_path):
    monkeypatch.setenv("REVIEW_STORE_PATH", str(tmp_path / "reviews.jsonl"))

    response = TestClient(app).post(
        "/reviews",
        json={
            "question": "What was revenue?",
            "verdict": "needs_correction",
            "original_answer": ANSWER,
            "corrected_answer": "   ",
        },
    )

    assert response.status_code == 422
    assert "corrected_answer is required" in response.text


def test_blank_question_and_invalid_list_limit_are_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv("REVIEW_STORE_PATH", str(tmp_path / "reviews.jsonl"))
    client = TestClient(app)

    blank = client.post(
        "/reviews",
        json={
            "question": "   ",
            "verdict": "correct",
            "original_answer": ANSWER,
        },
    )

    assert blank.status_code == 422
    assert client.get("/reviews", params={"limit": 0}).status_code == 422


def test_corrupt_review_store_returns_structured_error(monkeypatch, tmp_path):
    path = tmp_path / "reviews.jsonl"
    path.write_text("not-json\n", encoding="utf-8")
    monkeypatch.setenv("REVIEW_STORE_PATH", str(path))

    response = TestClient(app).get("/reviews")

    assert response.status_code == 500
    assert response.json()["detail"]["code"] == "review_storage_invalid"
