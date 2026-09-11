import pytest
from app.evidence_scoring import score_evidence, summarize_evidence
from scripts.audit_answer_eval import audit_report


def test_abstention_does_not_erase_successful_retrieval():
    result = score_evidence(
        {"answer_type": "insufficient_evidence", "evidence": []},
        [{"step": "retrieve", "hits": ["sha256-old:p1"]}],
        {"gold"},
        5,
        {"sha256-old": "gold"},
    )
    assert result["citation"]["hit_rate"] == 0
    assert result["agent_retrieval"]["final_attempt"]["hit_rate"] == 1
    assert result["agent_retrieval"]["final_attempt"]["precision_at_k"] == 0.2


def test_first_final_and_retry_union_are_distinct_with_multiple_gold_documents():
    result = score_evidence(
        {},
        [
            {"step": "retrieve", "hits": ["gold-a:p1"]},
            {"step": "retrieve", "hits": ["gold-b:p1", "gold-a:p2"]},
            {"step": "retrieve", "hits": ["wrong:p1"]},
        ],
        {"gold-a", "gold-b"},
        5,
        {},
    )["agent_retrieval"]
    assert result["first_attempt"]["recall_at_k"] == 0.5
    assert result["final_attempt"]["recall_at_k"] == 0
    assert result["any_attempt_recall"] == 1


@pytest.mark.parametrize(
    "trace",
    [None, [], [{"step": "retrieve"}], [{"step": "retrieve", "hits": ["broken"]}]],
)
def test_absent_or_malformed_trace_never_falls_back_to_citations(trace):
    result = score_evidence(
        {"evidence": [{"document_id": "gold"}]}, trace, {"gold"}, 5, {}
    )
    assert result["citation"]["hit_rate"] == 1
    assert result["agent_retrieval"]["final_attempt"] is None
    assert result["agent_retrieval"]["any_attempt_hit_rate"] is None
    assert (
        summarize_evidence([result])["agent_retrieval"]["final_attempt"][
            "scored_questions"
        ]
        == 0
    )


def test_empty_retrieval_is_a_known_miss():
    result = score_evidence({}, [{"step": "retrieve", "hits": []}], {"gold"}, 5, {})
    assert result["agent_retrieval"]["final_attempt"]["hit_rate"] == 0


def test_invalid_null_evidence_does_not_crash_benchmark():
    result = score_evidence({"evidence": None}, [], {"gold"}, 5, {})
    assert result["citation"]["hit_rate"] == 0


def test_structured_hit_metadata_resolves_new_hash_citation_without_static_alias():
    result = score_evidence(
        {"evidence": [{"document_id": "sha256-new", "page": 2}]},
        [
            {
                "step": "retrieve",
                "hits": ["sha256-new:p2"],
                "retrieval_hits": [
                    {"document_id": "sha256-new", "filename": "twin.pdf", "page": 2},
                ],
            }
        ],
        {"canonical"},
        5,
        {"twin": "canonical"},
    )
    assert result["citation"]["document_ids"] == ["canonical"]
    assert result["citation"]["hit_rate"] == 1
    assert result["agent_retrieval"]["unresolved_document_ids"] == []


def test_unresolved_hash_is_visible_and_not_guessed_from_gold():
    result = score_evidence(
        {}, [{"step": "retrieve", "hits": ["sha256-unknown:p1"]}], {"gold"}, 5, {}
    )
    assert result["agent_retrieval"]["unresolved_document_ids"] == ["sha256-unknown"]
    assert result["agent_retrieval"]["final_attempt"]["hit_rate"] == 0


def test_duplicate_hits_do_not_inflate_document_recall_or_move_rank():
    result = score_evidence(
        {},
        [{"step": "retrieve", "hits": ["other:p1", "other:p2", "gold:p1", "gold:p2"]}],
        {"gold"},
        3,
        {},
    )
    final = result["agent_retrieval"]["final_attempt"]
    assert final["recall_at_k"] == 1
    assert final["precision_at_k"] == pytest.approx(1 / 3)
    assert final["reciprocal_rank"] == pytest.approx(1 / 3)


def test_offline_audit_preserves_wrong_answer_and_exposes_retrieval_hit():
    source = {
        "results": [
            {
                "question_id": "q1",
                "question": "Revenue?",
                "gold": 10,
                "predicted_answer": {
                    "answer_type": "insufficient_evidence",
                    "evidence": [],
                },
                "system_trace": [{"step": "retrieve", "hits": ["sha256-old:p1"]}],
                "retrieval": {"hit_rate": 0},
            }
        ]
    }
    questions = [
        {
            "question_id": "q1",
            "question_text": "Revenue?",
            "ground_truth_answer": 10,
            "answer_type": "arithmetic",
            "gold_evidence": [{"source_doc_uid": "gold"}],
        }
    ]
    result = audit_report(source, questions, {"sha256-old": "gold"})
    assert result["summary"]["answer_quality"]["exact_match"] == 0
    assert result["summary"]["gold_document_found_but_abstained"] == ["q1"]
    assert source["results"][0]["retrieval"] == {"hit_rate": 0}
    with pytest.raises(ValueError, match="Duplicate"):
        audit_report({"results": source["results"] * 2}, questions, {})
    with pytest.raises(ValueError, match="Gold differs"):
        audit_report(source, [{**questions[0], "ground_truth_answer": 11}], {})
