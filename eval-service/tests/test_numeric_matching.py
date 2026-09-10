"""Exact match and F1 must compare figures by value.

Feeding each practice question's gold answer back through the scorer, in the
shape agent-service emits, 20 perfect answers scored EM 0 -- every
integer-valued arithmetic answer (a calculated 304811.0 against gold 304811)
and both count questions -- while 3.5 matched gold -3.5. Character-level
punctuation stripping caused both.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import metrics as m  # noqa: E402

QUESTIONS = Path(__file__).resolve().parents[1] / "questions_setA_practice.json"


@pytest.mark.parametrize(
    "prediction,gold",
    [
        (304811.0, 304811),
        (3.0, "3"),
        ("3,035", "3035"),
        (["2,657", "3,035"], ["3,035", "2,657"]),
        ("(1,234)", "-1234"),
        ("$1,000", "1000"),
        ("-0", "0"),
        ("0 per cent", "0 per cent"),
    ],
)
def test_equal_values_match(prediction, gold):
    assert m.exact_match(prediction, gold) == 1.0


@pytest.mark.parametrize(
    "prediction,gold",
    [
        (3.5, -3.5),
        ("-3.5", "3.5"),
        (304811.5, 304811),
        ("3.1", "31"),
    ],
)
def test_different_values_do_not_match(prediction, gold):
    assert m.exact_match(prediction, gold) == 0.0


def test_f1_counts_a_numeric_token_by_value():
    # "revenue", "of", "304811" against "304811": precision 1/3, recall 1.
    assert m.f1_score("revenue of 304811.0", "304811") == pytest.approx(0.5)


def test_prose_is_normalized_as_before():
    assert m.exact_match("The R&D costs.", "r&d costs") == 1.0


def test_every_numeric_gold_matches_itself_in_calculated_form():
    """agent-service returns arithmetic as round(float(result), 6)."""
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    numeric = [q for q in questions if q["answer_type"] in ("arithmetic", "count")]
    misses = [
        q["question_id"]
        for q in numeric
        if m.exact_match(round(float(q["ground_truth_answer"]), 6), q["ground_truth_answer"]) != 1.0
    ]
    assert numeric and misses == []
