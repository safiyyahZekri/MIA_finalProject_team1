"""
Exercises the actual FastAPI endpoints (via TestClient), not just the
internal `run_benchmark` function. This closes a real coverage gap: the
`BenchmarkConfig` field names changed when the record schema was confirmed,
but `main.py` still passed the old field names into it — something only an
end-to-end HTTP-level test catches, since `test_benchmark.py` calls
`run_benchmark` directly and never exercises the endpoint wiring at all.
"""
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

client = TestClient(app)

REAL_QUESTIONS_PATH = Path(__file__).resolve().parents[1] / "questions_setA_practice.json"


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_trace_roundtrip_via_api():
    resp = client.post("/trace/start", json={"name": "api test", "metadata": {"k": "v"}})
    assert resp.status_code == 200
    trace_id = resp.json()["trace_id"]

    resp = client.post(
        "/trace/step",
        json={"trace_id": trace_id, "name": "step1", "latency_ms": 12.3},
    )
    assert resp.status_code == 200

    resp = client.post("/trace/end", json={"trace_id": trace_id, "output": "done"})
    assert resp.status_code == 200

    resp = client.get(f"/trace/{trace_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["output"] == "done"
    assert body["steps"][0]["name"] == "step1"


def test_trace_get_missing_returns_404():
    resp = client.get("/trace/does-not-exist")
    assert resp.status_code == 404


def test_benchmark_run_endpoint_uses_current_config_fields(monkeypatch):
    """Regression test for the BenchmarkConfig field-name mismatch: this
    must not raise a TypeError from stale kwargs like `relevant_docs_field`
    or `gold_field="answer"` defaults."""
    import app.benchmark as bench_module

    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class FakeClient:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, json=None):
            return FakeResponse(
                {
                    "answer_type": "direct",
                    "evidence": [{"document_id": "doc_017", "page": 1}],
                    "params": {"value": "$142.5M"},
                }
            )

    monkeypatch.setattr(bench_module.httpx, "Client", FakeClient)

    resp = client.post(
        "/benchmark/run",
        json={
            "system_url": "http://fake-agent/answer",
            "questions": [
                {
                    "question_id": "q1",
                    "question_text": "What was the operating income reported in 2020?",
                    "ground_truth_answer": "$142.5M",
                }
            ],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["summary"]["num_questions"] == 1
    assert body["summary"]["exact_match"] == 1.0


def test_benchmark_run_from_file_endpoint_with_real_practice_file(monkeypatch):
    """Same regression check but through the multipart upload path, using
    the real 100-question file, with a fake system so no network is used."""
    import app.benchmark as bench_module

    with open(REAL_QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)
    by_text = {}
    for q in questions:
        gold = q.get("ground_truth_answer")
        if q.get("answer_type") == "unanswerable" or gold is None:
            by_text[q["question_text"]] = {
                "answer_type": "insufficient_evidence",
                "evidence": [],
                "params": {"reason": "n/a"},
            }
        elif isinstance(gold, list):
            by_text[q["question_text"]] = {
                "answer_type": "multi_span",
                "evidence": [{"document_id": "doc", "page": 0}],
                "params": {"values": gold},
            }
        else:
            by_text[q["question_text"]] = {
                "answer_type": "direct",
                "evidence": [{"document_id": "doc", "page": 0}],
                "params": {"value": gold},
            }

    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class FakeClient:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, json=None):
            return FakeResponse(by_text[json["question"]])

    monkeypatch.setattr(bench_module.httpx, "Client", FakeClient)

    with open(REAL_QUESTIONS_PATH, "rb") as f:
        resp = client.post(
            "/benchmark/run_from_file",
            data={"system_url": "http://fake-agent/answer"},
            files={"file": ("questions_setA_practice.json", f, "application/json")},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["summary"]["num_questions"] == 100
    assert body["summary"]["num_errors"] == 0

    run_id = body["run_id"]
    resp = client.get(f"/benchmark/results/{run_id}")
    assert resp.status_code == 200
    assert len(resp.json()["results"]) == 100
