import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import tracing  # noqa: E402
from app.benchmark import (  # noqa: E402
    BenchmarkConfig,
    _extract_relevant_doc_ids,
    _extract_scoping_doc_id,
    _split_core_answer_and_metadata,
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
    benchmark harness can be tested without real network calls. Responses
    use the confirmed agent-service /answer contract (nested under
    "answer", plus question_type/retries_used/trace siblings) — this is the
    real, confirmed shape, not a placeholder."""

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
                        "answer": {
                            "answer_type": "direct",
                            "evidence": [{"document_id": "doc_017", "page": 1}],
                            "params": {"value": "$142.5M"},
                        },
                        "question_type": "financial_lookup",
                        "retries_used": 0,
                        "trace": [],
                    }
                )
            return FakeResponse(
                {
                    "answer": {
                        "answer_type": "insufficient_evidence",
                        "evidence": [],
                        "params": {"reason": "not found"},
                    },
                    "question_type": "unanswerable",
                    "retries_used": 1,
                    "trace": [],
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
    assert summary["avg_retries_used"] == 0.5  # (0 + 1) / 2, from the confirmed contract's field
    assert summary["citation"]["hit_rate"] == 1.0
    # The fake agent has no retrieval trace: its citation must not be used
    # to fabricate retrieval-only quality.
    assert summary["agent_retrieval"]["final_attempt"]["scored_questions"] == 0
    assert report["metric_definitions"]["legacy_retrieval"]


# ---------------------------------------------------------------------------
# Real-data tests — load the actual practice-set file, not synthetic
# question dicts, to confirm the harness parses the confirmed record schema
# correctly on the first try.
# ---------------------------------------------------------------------------


def test_benchmark_scores_ranked_retrieval_when_agent_declines(monkeypatch):
    import app.benchmark as bench_module

    class DecliningClient(FakeClient):
        def post(self, url, json=None):
            assert json["document_id"] is None  # no gold scoping
            return FakeResponse({
                "answer": {"answer_type": "insufficient_evidence", "evidence": [],
                           "params": {"reason": "company not identified"}},
                "trace": [{"step": "retrieve", "hits": ["sha256-old:p1"]}],
            })

    monkeypatch.setattr(bench_module.httpx, "Client", DecliningClient)
    report = run_benchmark(BenchmarkConfig(
        system_url="http://fake-system/ask",
        questions=[{"question_id": "q", "question_text": "Revenue?", "ground_truth_answer": 10,
                    "answer_type": "arithmetic", "gold_evidence": [{"source_doc_uid": "gold"}]}],
        identity_aliases={"sha256-old": "gold"},
    ))
    row = report["results"][0]
    assert row["exact_match"] == 0
    assert row["citation"]["hit_rate"] == 0
    assert row["agent_retrieval"]["final_attempt"]["hit_rate"] == 1
    assert report["summary"]["agent_retrieval"]["final_attempt"]["hit_rate"] == 1


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


def test_end_to_end_benchmark_does_not_scope_to_gold_by_default(monkeypatch):
    import app.benchmark as bench_module

    captured = {}

    class CaptureClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, json=None):
            captured.update(json)
            return FakeResponse(
                {
                    "answer": {
                        "answer_type": "insufficient_evidence",
                        "evidence": [],
                        "params": {"reason": "none"},
                    }
                }
            )

    monkeypatch.setattr(bench_module.httpx, "Client", CaptureClient)
    run_benchmark(
        BenchmarkConfig(
            system_url="http://agent/answer",
            questions=[
                {
                    "question_id": "q",
                    "question_text": "question",
                    "source_doc_uid": "gold-id",
                    "gold_evidence": [{"source_doc_uid": "gold-id"}],
                }
            ],
        )
    )
    assert captured["document_id"] is None


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
            core = {"answer_type": "insufficient_evidence", "evidence": [], "params": {"reason": "n/a"}}
        else:
            evidence = [
                {"document_id": e.get("source_doc_uid"), "page": e.get("source_page", 0)}
                for e in (question.get("gold_evidence") or [])
            ] or [{"document_id": "unknown", "page": 0}]
            if isinstance(gold, list):
                core = {"answer_type": "multi_span", "evidence": evidence, "params": {"values": gold}}
            else:
                core = {"answer_type": "direct", "evidence": evidence, "params": {"value": gold}}
        # Confirmed agent-service /answer shape: the answer object nested
        # under "answer", plus question_type/retries_used/trace siblings.
        return {
            "answer": core,
            "question_type": question.get("answer_type"),
            "retries_used": 0,
            "trace": [],
        }

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


# ---------------------------------------------------------------------------
# Confirmed agent-service /answer contract: the answer object is nested
# under "answer", with question_type/retries_used/trace as siblings. This
# was confirmed end-to-end against the real feature/agent-service branch
# (100/100, 0 errors, 100% schema validity, using answer_key="answer").
# ---------------------------------------------------------------------------


def test_split_core_answer_confirmed_nested_shape():
    body = {
        "answer": {
            "answer_type": "direct",
            "evidence": [{"document_id": "doc_017", "page": 1}],
            "params": {"value": "$142.5M"},
        },
        "question_type": "financial_lookup",
        "retries_used": 2,
        "trace": [{"step": "classify"}, {"step": "retrieve"}, {"step": "generate"}],
    }
    core, system_trace, question_type = _split_core_answer_and_metadata(body, answer_key="answer")
    assert core == {
        "answer_type": "direct",
        "evidence": [{"document_id": "doc_017", "page": 1}],
        "params": {"value": "$142.5M"},
    }
    # trace/question_type are read from the top level, as confirmed
    # siblings of "answer" -- not from inside the nested answer object.
    assert system_trace == [{"step": "classify"}, {"step": "retrieve"}, {"step": "generate"}]
    assert question_type == "financial_lookup"


def test_split_core_answer_flat_shape_still_supported():
    """`answer_key=None` still supports a system that returns the bare
    answer object with trace/question_type as top-level siblings (no
    "answer" wrapper) — kept for systems other than the confirmed
    agent-service contract, or for future/alternate configurations."""
    body = {
        "answer_type": "direct",
        "evidence": [{"document_id": "doc_017", "page": 1}],
        "params": {"value": "$142.5M"},
        "trace": {"anything": "a system could shape this however it wants"},
        "question_type": "financial_lookup",
    }
    core, system_trace, question_type = _split_core_answer_and_metadata(body, answer_key=None)
    assert core == {
        "answer_type": "direct",
        "evidence": [{"document_id": "doc_017", "page": 1}],
        "params": {"value": "$142.5M"},
    }
    assert system_trace == {"anything": "a system could shape this however it wants"}
    assert question_type == "financial_lookup"


def test_split_core_answer_handles_missing_trace_and_question_type():
    body = {
        "answer_type": "insufficient_evidence",
        "evidence": [],
        "params": {"reason": "n/a"},
    }
    core, system_trace, question_type = _split_core_answer_and_metadata(body, answer_key=None)
    assert core == body
    assert system_trace is None
    assert question_type is None


class ContractFakeClient:
    """Mimics the confirmed agent-service /answer contract exactly:
    {"answer": {...}, "question_type": ..., "retries_used": N, "trace": [...]}."""

    def __init__(self, *a, **kw):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def post(self, url, json=None):
        if url == "http://fake-agent/answer":
            return FakeResponse(
                {
                    "answer": {
                        "answer_type": "direct",
                        "evidence": [{"document_id": "doc_017", "page": 1}],
                        "params": {"value": "$142.5M"},
                    },
                    "question_type": "financial_lookup",
                    "retries_used": 2,
                    "trace": [{"step": "classify"}, {"step": "retrieve"}, {"step": "generate"}],
                }
            )
        if url == "http://fake-validator/validate_answer":
            ContractFakeClient.last_validator_payload = json
            return FakeResponse({"valid": True, "reason": None})
        raise AssertionError(f"unexpected URL {url}")


def test_benchmark_stores_system_trace_and_strips_it_before_validation(monkeypatch):
    import app.benchmark as bench_module

    monkeypatch.setattr(bench_module.httpx, "Client", ContractFakeClient)

    questions = [
        {
            "question_id": "q1",
            "question_text": "What was the operating income reported in 2020?",
            "ground_truth_answer": "$142.5M",
            "source_doc_uid": "doc_017",
            "gold_evidence": [{"source_doc_uid": "doc_017"}],
        }
    ]
    config = BenchmarkConfig(
        system_url="http://fake-agent/answer",
        questions=questions,
        validator_url="http://fake-validator/validate_answer",
    )
    report = run_benchmark(config)
    result = report["results"][0]

    assert result["system_trace"] == [
        {"step": "classify"},
        {"step": "retrieve"},
        {"step": "generate"},
    ]
    assert result["question_type"] == "financial_lookup"
    assert result["schema_valid"] is True
    assert result["extra_perf"]["retries_used"] == 2
    assert report["summary"]["avg_retries_used"] == 2.0
    # The validator must only ever see the bare three-key answer object.
    assert set(ContractFakeClient.last_validator_payload.keys()) == {
        "answer_type",
        "evidence",
        "params",
    }


def test_agent_steps_are_logged_to_the_trace(monkeypatch):
    """Failure analysis reads the agent's steps (queries, hits, grader
    reasons) from the trace, so each one must be logged there, not only
    stored in the result row."""
    import app.benchmark as bench_module

    monkeypatch.setattr(bench_module.httpx, "Client", ContractFakeClient)

    questions = [
        {
            "question_id": "q1",
            "question_text": "What was the operating income reported in 2020?",
            "ground_truth_answer": "$142.5M",
            "source_doc_uid": "doc_017",
            "gold_evidence": [{"source_doc_uid": "doc_017"}],
        }
    ]
    config = BenchmarkConfig(
        system_url="http://fake-agent/answer",
        questions=questions,
        validator_url="http://fake-validator/validate_answer",
    )
    result = run_benchmark(config)["results"][0]

    steps = tracing.get_trace(result["trace_id"])["steps"]
    agent_steps = [s for s in steps if s["name"].startswith("agent.")]
    assert [s["name"] for s in agent_steps] == ["agent.01.classify", "agent.02.retrieve", "agent.03.generate"]
    assert agent_steps[1]["output"] == {"step": "retrieve"}


# ---------------------------------------------------------------------------
# eval-service's own trace_id must be independent of upstream tracing and
# must resolve to an actual (local-fallback-acceptable) trace.
# ---------------------------------------------------------------------------


def test_every_result_has_a_resolvable_trace_id(monkeypatch):
    import app.benchmark as bench_module

    monkeypatch.setattr(bench_module.httpx, "Client", FakeClient)

    questions = _load_real_questions()[:10]  # a subset, per the verification requirement
    config = BenchmarkConfig(system_url="http://fake-system/ask", questions=questions)
    report = run_benchmark(config)

    assert len(report["results"]) == 10
    for result in report["results"]:
        trace_id = result["trace_id"]
        assert trace_id, "every benchmarked question must carry a trace_id"
        trace = tracing.get_trace(trace_id)
        assert trace is not None, f"trace_id {trace_id} did not resolve to an actual trace"
        step_names = [s["name"] for s in trace["steps"]]
        assert "call_system" in step_names
        assert "score" in step_names
        assert "output" in trace  # set by end_trace


def test_failed_examples_reference_resolvable_traces(monkeypatch):
    import app.benchmark as bench_module

    monkeypatch.setattr(bench_module.httpx, "Client", FakeClient)

    questions = [
        {
            "question_id": "q_wrong",
            "question_text": "What were the restructuring expenses?",  # -> insufficient_evidence
            "ground_truth_answer": "$5.0M",  # wrong on purpose: forces an EM miss
            "gold_evidence": [],
        }
    ]
    config = BenchmarkConfig(system_url="http://fake-system/ask", questions=questions)
    report = run_benchmark(config)

    failed = report["summary"]["failed_examples"]
    assert report["summary"]["num_failed_examples"] == 1
    assert len(failed) == 1
    assert failed[0]["question_id"] == "q_wrong"
    assert tracing.get_trace(failed[0]["trace_id"]) is not None
