import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import metrics as m  # noqa: E402


def test_exact_match_simple():
    assert m.exact_match("$142.5M", "$142.5M") == 1.0
    assert m.exact_match("The $142.5M", "$142.5M") == 1.0  # article stripped
    assert m.exact_match("$100M", "$142.5M") == 0.0


def test_exact_match_multi_span_order_insensitive():
    assert m.exact_match(["R&D", "Marketing"], ["Marketing", "R&D"]) == 1.0


def test_f1_partial_overlap():
    score = m.f1_score("operating income was 142.5 million", "142.5 million")
    assert 0.0 < score < 1.0


def test_f1_no_overlap():
    assert m.f1_score("apples", "oranges") == 0.0


def test_numerical_accuracy_within_tolerance():
    assert m.numerical_accuracy(13.41, 13.4, tolerance=0.01) == 1.0


def test_numerical_accuracy_outside_tolerance():
    assert m.numerical_accuracy(20.0, 13.4, tolerance=0.01) == 0.0


def test_numerical_accuracy_none_when_not_numeric():
    assert m.numerical_accuracy("Marketing", "R&D") is None


# --- Scale handling, matching the real practice-set `scale` field ---


def test_numerical_accuracy_same_units_no_scale_word_needed():
    # Gold is 304811 with scale="thousand"; a prediction already quoted in
    # thousands (no unit word) should match directly, no normalization.
    assert m.numerical_accuracy(304811, 304811, gold_scale="thousand") == 1.0


def test_numerical_accuracy_normalizes_explicit_million_prediction():
    # Gold: 304811 (thousand) == 304.811 million. Prediction quotes "million"
    # explicitly, so it must be normalized before comparing.
    assert m.numerical_accuracy("304.811 million", 304811, gold_scale="thousand") == 1.0


def test_numerical_accuracy_mismatched_scale_is_rejected():
    # Prediction says "304.811 million" but gold is actually in millions
    # already (i.e. true value is 304.811, not 304,811,000) -> should fail.
    assert m.numerical_accuracy("304.811 million", 304.811, gold_scale="million") == 1.0
    assert m.numerical_accuracy("304.811 thousand", 304.811, gold_scale="million") == 0.0


def test_numerical_accuracy_percent_scale_is_not_multiplied():
    assert m.numerical_accuracy(13.4, 13.4, gold_scale="percent") == 1.0


def test_recall_precision_at_k():
    retrieved = ["doc_1", "doc_2", "doc_3", "doc_4", "doc_5"]
    relevant = ["doc_3", "doc_9"]
    assert m.recall_at_k(retrieved, relevant, k=5) == 0.5
    assert m.precision_at_k(retrieved, relevant, k=5) == 1 / 5


def test_hit_rate_and_reciprocal_rank():
    retrieved = ["doc_1", "doc_2", "doc_3"]
    relevant = ["doc_3"]
    assert m.hit_rate(retrieved, relevant, k=3) == 1.0
    assert m.reciprocal_rank(retrieved, relevant) == 1 / 3


def test_mrr_over_multiple_queries():
    retrieved_lists = [["doc_1", "doc_2"], ["doc_5", "doc_6"]]
    relevant_lists = [["doc_2"], ["doc_9"]]
    assert m.mrr(retrieved_lists, relevant_lists) == (0.5 + 0.0) / 2


def test_mean_ignores_none():
    assert m.mean([1.0, None, 3.0]) == 2.0
    assert m.mean([None, None]) is None
