import json

import pytest
from scripts.compare_decomposition_runs import compare_runs, load_run


def test_comparison_counts_failed_answers_as_zero_and_checks_same_questions():
    question = {
        "question_id": "q",
        "question_text": "Revenue?",
        "ground_truth_answer": 10,
        "answer_type": "arithmetic",
        "gold_evidence": [{"source_doc_uid": "d"}],
    }
    row = {
        "question_id": "q",
        "question": "Revenue?",
        "gold": 10,
        "predicted_answer": {
            "answer_type": "direct",
            "params": {"value": "10"},
            "evidence": [],
        },
    }
    baseline = {
        "config": {},
        "report": {"results": [{**row, "predicted_answer": None, "error": "timeout"}]},
    }
    candidate = {"config": {}, "report": {"results": [row]}}
    result = compare_runs(baseline, candidate, [question], {})
    assert result["delta"]["exact_match_all"] == 1
    assert result["baseline"]["errors"] == 1
    assert result["baseline"]["retrieval_recall_at_5"] is None
    with pytest.raises(ValueError, match="same non-empty"):
        compare_runs(
            baseline, {"config": {}, "report": {"results": []}}, [question], {}
        )
    with pytest.raises(ValueError, match="settings differ"):
        compare_runs(
            baseline, {**candidate, "config": {"top_k_final": 10}}, [question], {}
        )


def test_snapshot_validation_rejects_wrong_flag_and_mock_provider(tmp_path):
    config = {"query_decomposition": True, "top_k_final": 5}
    info = {"agent": {"llm_provider": "anthropic", "config": config}}
    (tmp_path / "run_info.json").write_text(json.dumps(info))
    (tmp_path / "combined.json").write_text('{"results": []}')
    assert load_run(tmp_path, True)["config"]["top_k_final"] == 5
    with pytest.raises(ValueError, match="expected query_decomposition"):
        load_run(tmp_path, False)
    info["agent"]["llm_provider"] = "mock"
    (tmp_path / "run_info.json").write_text(json.dumps(info))
    with pytest.raises(ValueError, match="Mock-model"):
        load_run(tmp_path, True)
