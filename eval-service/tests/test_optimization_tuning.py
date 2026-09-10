import json

import pytest
from app.retrieval_benchmark import RetrievalBenchmarkConfig, run_retrieval_benchmark
from scripts.calibrate_evidence_gate import calibrate
from scripts.replay_answer_normalization import replay
from scripts.tune_retrieval import sweep


def test_calibration_enforces_false_accept_budget_and_preserves_model_rejection():
    rows = [
        {
            "question_id": "a",
            "evidence_sufficient": True,
            "model_sufficient": True,
            "top_score": 0.8,
            "grade_confidence": 0.9,
        },
        {
            "question_id": "b",
            "evidence_sufficient": False,
            "model_sufficient": True,
            "top_score": 0.5,
            "grade_confidence": 0.7,
        },
        {
            "question_id": "c",
            "evidence_sufficient": True,
            "model_sufficient": False,
            "top_score": 0.9,
            "grade_confidence": 0.9,
        },
    ]
    result = calibrate(rows, 0)
    assert result["recommendation"]["false_accepts"] == 0
    assert result["recommendation"]["true_accepts"] == 1
    assert result["model_rejected_sufficient"] == 1
    assert result["recommendation"]["recall"] == 0.5
    with pytest.raises(ValueError):
        calibrate(rows[:1])
    with pytest.raises(ValueError):
        calibrate(rows + [rows[0]])


class Response:
    def __init__(self, body):
        self.body = body

    def json(self):
        return self.body

    def raise_for_status(self):
        pass


class Client:
    def __init__(self, confirm_reranking=True):
        self.calls = []
        self.confirm = confirm_reranking

    def post(self, url, json):
        self.calls.append(json)
        return Response(
            {
                "hits": [{"source_doc_uid": "gold", "document_id": "sha256-content"}],
                "diagnostics": {
                    "search_settings": {
                        key: json[key]
                        for key in ("dense_weight", "rrf_k", "candidate_k", "top_k")
                    }
                },
                "reranked": json["rerank"] and self.confirm,
            }
        )


QUESTIONS = [
    {
        "question_id": "a",
        "question_text": "Revenue?",
        "source_doc_uid": "gold",
        "gold_evidence": [{"source_doc_uid": "gold"}],
    }
]


def test_sweep_uses_same_unscoped_queries_and_saves_real_reports(tmp_path):
    client = Client()
    result = sweep(
        "http://test",
        QUESTIONS,
        tmp_path,
        [0.35, 0.55],
        [20],
        [30],
        [False, True],
        client,
    )
    assert len(client.calls) == 4
    assert all(
        "document_id" not in call and "filters" not in call for call in client.calls
    )
    assert {call["query"] for call in client.calls} == {"Revenue?"}
    assert {call["dense_weight"] for call in client.calls} == {0.35, 0.55}
    assert all(report["summary"]["errors"] == 0 for report in result["reports"])
    assert result["development_winner"] is not None
    assert (
        json.loads((tmp_path / "tuning-comparison.json").read_text())["scope"]
        == "corpus-wide"
    )
    assert "MRR" in (tmp_path / "tuning-comparison.md").read_text()


def test_missing_reranker_is_an_error_not_an_ablation_result():
    result = run_retrieval_benchmark(
        RetrievalBenchmarkConfig(
            retrieval_url="http://test",
            questions=QUESTIONS,
            save_results=False,
        ),
        Client(confirm_reranking=False),
    )
    assert result["summary"]["errors"] == 1
    assert result["summary"]["mrr"] is None


def test_sweep_rejects_old_service_that_does_not_apply_parameters(tmp_path):
    class OldClient:
        def post(self, url, json):
            return Response({"hits": [{"source_doc_uid": "gold"}], "reranked": False})

    result = sweep(
        "http://test", QUESTIONS, tmp_path, [0.35], [60], [30], [False], OldClient()
    )
    assert result["development_winner"] is None
    assert result["reports"][0]["summary"]["errors"] == 1


def test_normalization_replay_leaves_source_and_scorer_unchanged():
    report = {
        "results": [
            {
                "question_id": "a",
                "question": "Revenue?",
                "gold": "100",
                "predicted_answer": {
                    "answer_type": "direct",
                    "params": {"value": "  100 "},
                    "evidence": [],
                },
            }
        ]
    }
    questions = [
        {
            "question_id": "a",
            "question_text": "Revenue?",
            "ground_truth_answer": "100",
            "answer_type": "span",
        }
    ]
    result = replay(report, questions, {})
    assert result["changed_answers"] == ["a"]
    assert result["delta"] == {"exact_match": 0, "f1": 0}
    assert report["results"][0]["predicted_answer"]["params"]["value"] == "  100 "
