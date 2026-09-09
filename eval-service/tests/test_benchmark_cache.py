import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.benchmark import BenchmarkConfig, run_benchmark  # noqa: E402


class CountingFakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class CountingFakeClient:
    """Counts real calls to system_url so cache effectiveness is asserted
    on an actual number, not inferred from timing."""

    call_count = 0

    def __init__(self, *a, **kw):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def post(self, url, json=None):
        if url == "http://fake-agent/answer":
            CountingFakeClient.call_count += 1
            return CountingFakeResponse(
                {
                    "answer": {
                        "answer_type": "direct",
                        "evidence": [{"document_id": "doc_1", "page": 1}],
                        "params": {"value": "$100M"},
                    }
                }
            )
        raise AssertionError(f"unexpected URL {url}")


REPEATED_QUESTIONS = [
    {"question_id": "q1", "question_text": "What was revenue in 2019?", "source_doc_uid": "doc_1", "ground_truth_answer": "$100M"},
    {"question_id": "q2", "question_text": "What was revenue in 2019?", "source_doc_uid": "doc_1", "ground_truth_answer": "$100M"},
    {"question_id": "q3", "question_text": "What was the revenue in 2019?", "source_doc_uid": "doc_1", "ground_truth_answer": "$100M"},
    {"question_id": "q4", "question_text": "What was revenue in 2019?", "source_doc_uid": "doc_2", "ground_truth_answer": "$100M"},
]


def test_cache_disabled_calls_system_every_time(monkeypatch):
    import app.benchmark as bench_module

    CountingFakeClient.call_count = 0
    monkeypatch.setattr(bench_module.httpx, "Client", CountingFakeClient)

    config = BenchmarkConfig(
        system_url="http://fake-agent/answer",
        questions=REPEATED_QUESTIONS,
        use_cache=False,
    )
    report = run_benchmark(config)

    assert CountingFakeClient.call_count == 4  # every question hit the system
    assert report["summary"]["cache_hits"] == 0
    assert report["summary"]["cache_misses"] == 0  # cache wasn't used at all
    assert all(r["cache_hit"] is None for r in report["results"])


def test_cache_enabled_skips_redundant_calls(monkeypatch):
    import app.benchmark as bench_module

    CountingFakeClient.call_count = 0
    monkeypatch.setattr(bench_module.httpx, "Client", CountingFakeClient)

    config = BenchmarkConfig(
        system_url="http://fake-agent/answer",
        questions=REPEATED_QUESTIONS,
        use_cache=True,
        cache_similarity_threshold=0.7,
        scope_to_gold_document=True,  # so document_id is actually populated per-question
    )
    report = run_benchmark(config)

    # q1: miss (first time). q2: exact repeat of q1, same doc -> hit.
    # q3: near-duplicate phrasing of q1, same doc -> fuzzy hit.
    # q4: same text as q1 but a DIFFERENT document -> miss (correctly scoped).
    assert CountingFakeClient.call_count == 2  # only q1 and q4 actually hit the system
    assert report["summary"]["cache_hits"] == 2
    assert report["summary"]["cache_misses"] == 2

    results_by_id = {r["question_id"]: r for r in report["results"]}
    assert results_by_id["q1"]["cache_hit"] is False
    assert results_by_id["q2"]["cache_hit"] is True
    assert results_by_id["q3"]["cache_hit"] is True
    assert results_by_id["q4"]["cache_hit"] is False

    # Every result still has a correct, usable answer regardless of cache hit.
    for r in report["results"]:
        assert r["predicted_answer"]["params"]["value"] == "$100M"
