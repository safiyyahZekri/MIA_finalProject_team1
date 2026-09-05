import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.benchmark import (  # noqa: E402
    BenchmarkConfig,
    _extract_relevant_doc_ids,
    _extract_scoping_doc_id,
    run_benchmark,
)

REAL_QUESTIONS_PATH = Path(__file__).resolve().parents[1] / "questions_setA_practice.json"


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class FakeClient:
    """Stands in for httpx.Client: routes by URL to canned answers so the
    benchmark harness can be tested without real network calls."""

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def post(self, url, json=None):
        if url == "http://fake-system/ask":
            question = json["question"]
            if "operating income" in question:
                return FakeResponse(
                    {
                        "answer_type": "direct",
                        "evidence": [{"document_id": "doc_017", "page": 1}],
                        "params": {"value": "$142.5M"},
                    }
                )
            return FakeResponse(
                {
                    "answer_type": "insufficient_evidence",
                    "evidence": [],
                    "params": {"reason": "not found"},
                }
            )
        if url == "http://fake-validator/validate_answer":
            return FakeResponse({"valid": True, "reason": None})
        raise AssertionError(f"unexpected URL {url}")


def test_run_benchmark_end_to_end(monkeypatch):
    import app.benchmark as bench_module

    monkeypatch.setattr(bench_module.httpx, "Client", FakeClient)

    questions = [
        {
            "question_id": "q1",
            "question_text": "What was the operating income reported in 2020?",
            "ground_truth_answer": "$142.5M",
            "answer_type": "span",
            "source_doc_uid": "doc_017",
            "gold_evidence": [{"source_doc_uid": "doc_017"}],
        },
        {
            "question_id": "q2",
            "question_text": "What were the restructuring expenses?",
            "ground_truth_answer": None,
            "answer_type": "unanswerable",
            "gold_evidence": [],
        },
    ]

    config = BenchmarkConfig(
        system_url="http://fake-system/ask",
        questions=questions,
        validator_url="http://fake-validator/validate_answer",
    )
    report = run_benchmark(config)

    assert report["num_questions"] == 2
    summary = report["summary"]
    assert summary["schema_validity_rate"] == 1.0
    assert summary["exact_match"] == 1.0  # only q1 has a comparable gold answer
    assert summary["retrieval_recall_at_k"] == 1.0
    assert summary["num_errors"] == 0


# ---------------------------------------------------------------------------
# Real-data tests — load the actual practice-set file, not synthetic
# question dicts, to confirm the harness parses the confirmed record schema
# correctly on the first try.
# ---------------------------------------------------------------------------


def _load_real_questions():
    with open(REAL_QUESTIONS_PATH, encoding="utf-8") as f:
        return json.load(f)


def test_real_file_loads_and_has_expected_shape():
    questions = _load_real_questions()
    assert len(questions) == 100
    first = questions[0]
    # These are the actual field names confirmed against
    # record_schema_guide_ar.md — NOT "question"/"answer"/"document_id".
    assert "question_text" in first
    assert "ground_truth_answer" in first
    assert "gold_evidence" in first
    assert "question" not in first
    assert "answer" not in first
    assert "relevant_document_ids" not in first


def test_extract_relevant_doc_ids_reads_gold_evidence():
    questions = _load_real_questions()
    cross_doc = next(q for q in questions if q["task_family"] == "derived_cross_document")
    ids = _extract_relevant_doc_ids(cross_doc)
    assert len(ids) == 2
    assert all(isinstance(i, str) for i in ids)


def test_extract_scoping_doc_id_is_null_for_cross_document():
    questions = _load_real_questions()
    cross_doc = next(q for q in questions if q["task_family"] == "derived_cross_document")
    single_doc = next(q for q in questions if q["task_family"] == "authentic_single_document")
    assert _extract_scoping_doc_id(cross_doc) is None
    assert _extract_scoping_doc_id(single_doc) is not None


def test_benchmark_runs_against_real_practice_file(monkeypatch):
    """The actual end-to-end check for open item #2: feed the real 100
    questions through the harness with a fake system that echoes the gold
    answer back as a Strict-Answer-Schema object, and confirm scoring
    actually engages (non-null metrics) rather than silently no-op'ing on
    field-name mismatches."""
    import app.benchmark as bench_module

    questions = _load_real_questions()

    def fake_answer_for(question: dict) -> dict:
        gold = question.get("ground_truth_answer")
        if question.get("answer_type") == "unanswerable" or gold is None:
            return {"answer_type": "insufficient_evidence", "evidence": [], "params": {"reason": "n/a"}}
        evidence = [
            {"document_id": e.get("source_doc_uid"), "page": e.get("source_page", 0)}
            for e in (question.get("gold_evidence") or [])
        ] or [{"document_id": "unknown", "page": 0}]
        if isinstance(gold, list):
            return {"answer_type": "multi_span", "evidence": evidence, "params": {"values": gold}}
        return {"answer_type": "direct", "evidence": evidence, "params": {"value": gold}}

    by_question_text = {q["question_text"]: fake_answer_for(q) for q in questions}

    class RealDataFakeClient:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, json=None):
            if url == "http://fake-system/ask":
                return FakeResponse(by_question_text[json["question"]])
            raise AssertionError(f"unexpected URL {url}")

    monkeypatch.setattr(bench_module.httpx, "Client", RealDataFakeClient)

    config = BenchmarkConfig(system_url="http://fake-system/ask", questions=questions)
    report = run_benchmark(config)
    summary = report["summary"]

    assert report["num_questions"] == 100
    assert summary["num_errors"] == 0
    # A perfect echo-back system should score perfectly on EM once field
    # names are actually being read correctly.
    assert summary["exact_match"] == 1.0
    assert summary["retrieval_recall_at_k"] == 1.0
