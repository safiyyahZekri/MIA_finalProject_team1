"""
Metrics required by the spec's Evaluation section:

  Answer quality:    Exact Match, F1, numerical accuracy
  Retrieval quality: Recall@K, Precision@K, MRR / Hit Rate
  System perf:       average latency, LLM calls per query, token usage, cost

All functions here are pure and independently testable.
"""
from __future__ import annotations

import re
import string
from typing import Iterable, List, Optional, Sequence, Union

Number = Union[int, float]


# ---------------------------------------------------------------------------
# Answer quality
# ---------------------------------------------------------------------------

_ARTICLES = {"a", "an", "the"}


def _normalize_text(s: str) -> str:
    """SQuAD-style normalization: lowercase, strip punctuation, drop
    articles, collapse whitespace."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    tokens = [t for t in s.split() if t not in _ARTICLES]
    return " ".join(tokens)


def _to_comparable_strings(value) -> List[str]:
    """Flatten a predicted/gold answer (which may be a scalar, a list for
    multi_span, or a dict with 'value'/'values') into a list of strings for
    comparison."""
    if value is None:
        return [""]
    if isinstance(value, (list, tuple)):
        out = []
        for v in value:
            out.extend(_to_comparable_strings(v))
        return out
    if isinstance(value, dict):
        if "values" in value:
            return _to_comparable_strings(value["values"])
        if "value" in value:
            return _to_comparable_strings(value["value"])
        return [str(value)]
    return [str(value)]


def exact_match(prediction, gold) -> float:
    """1.0 if the normalized prediction matches the normalized gold answer
    (order-insensitive for multi-value answers), else 0.0."""
    pred_norm = sorted(_normalize_text(s) for s in _to_comparable_strings(prediction))
    gold_norm = sorted(_normalize_text(s) for s in _to_comparable_strings(gold))
    return 1.0 if pred_norm == gold_norm else 0.0


def f1_score(prediction, gold) -> float:
    """Token-overlap F1 over the bag of words in the flattened
    prediction/gold strings (standard SQuAD-style F1)."""
    pred_tokens: List[str] = []
    for s in _to_comparable_strings(prediction):
        pred_tokens.extend(_normalize_text(s).split())
    gold_tokens: List[str] = []
    for s in _to_comparable_strings(gold):
        gold_tokens.extend(_normalize_text(s).split())

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = {}
    for t in pred_tokens:
        common[t] = min(pred_tokens.count(t), gold_tokens.count(t))
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


_NUMERIC_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def _extract_number(value) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        if "value" in value:
            return _extract_number(value["value"])
    if isinstance(value, str):
        match = _NUMERIC_RE.search(value.replace(",", ""))
        if match:
            try:
                return float(match.group())
            except ValueError:
                return None
    return None


_SCALE_MULTIPLIER = {"": 1, "thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000}


def numerical_accuracy(
    prediction, gold, tolerance: float = 0.01, gold_scale: str = ""
) -> Optional[float]:
    """Returns 1.0 / 0.0 if both prediction and gold contain a parseable
    number and they match within `tolerance` relative error. Returns None
    (not applicable) if either side has no parseable number — callers should
    exclude None results from the numerical-accuracy aggregate rather than
    counting them as failures.

    `gold_scale` matches the practice-dataset `scale` field ("", "thousand",
    "million", "billion", "percent"). The gold number is stored in that
    scale's units (e.g. 304811 with scale="thousand" means 304,811 thousand).
    Since we don't know what units the *predicted* value is quoted in from a
    bare number, we resolve this the same way the spec's own examples do:
    if the predicted string carries its own unit word ("million", "M",
    "thousand", "K", "billion", "B"), normalize it to the gold's scale first.
    Otherwise assume the prediction is already in the gold's scale (the
    common case, since the agent is expected to quote figures in the
    document's native units).
    """
    pred_num = _extract_number(prediction)
    gold_num = _extract_number(gold)
    if pred_num is None or gold_num is None:
        return None

    pred_scale_mult = _detect_prediction_scale(prediction)
    gold_scale_mult = _SCALE_MULTIPLIER.get(gold_scale, 1)
    if pred_scale_mult is not None:
        # Normalize both to raw units before comparing.
        pred_num = pred_num * pred_scale_mult
        gold_num = gold_num * gold_scale_mult

    if gold_num == 0:
        return 1.0 if abs(pred_num) < 1e-9 else 0.0
    rel_error = abs(pred_num - gold_num) / abs(gold_num)
    return 1.0 if rel_error <= tolerance else 0.0


_SCALE_WORD_RE = re.compile(r"(?i)\b(thousand|million|billion|[kmb])\b")
_SCALE_WORD_MULTIPLIER = {
    "thousand": 1_000, "k": 1_000,
    "million": 1_000_000, "m": 1_000_000,
    "billion": 1_000_000_000, "b": 1_000_000_000,
}


def _detect_prediction_scale(value) -> Optional[int]:
    """Looks for an explicit unit word in a string prediction (e.g. "304.8
    million"). Returns None if the prediction isn't a string with a
    detectable scale word, signaling the caller to skip normalization."""
    if not isinstance(value, str):
        return None
    match = _SCALE_WORD_RE.search(value)
    if not match:
        return None
    return _SCALE_WORD_MULTIPLIER.get(match.group(1).lower())


# ---------------------------------------------------------------------------
# Retrieval quality
# ---------------------------------------------------------------------------


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    relevant_set = set(relevant)
    if not relevant_set:
        return 1.0  # nothing was relevant, so nothing to miss
    top_k = set(retrieved[:k])
    hit = len(top_k & relevant_set)
    return hit / len(relevant_set)


def precision_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    relevant_set = set(relevant)
    top_k = list(retrieved[:k])
    if not top_k:
        return 0.0
    hit = len([r for r in top_k if r in relevant_set])
    return hit / len(top_k)


def hit_rate(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    relevant_set = set(relevant)
    return 1.0 if relevant_set & set(retrieved[:k]) else 0.0


def reciprocal_rank(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    relevant_set = set(relevant)
    for i, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant_set:
            return 1.0 / i
    return 0.0


def mrr(list_of_retrieved: Sequence[Sequence[str]], list_of_relevant: Sequence[Iterable[str]]) -> float:
    if not list_of_retrieved:
        return 0.0
    scores = [
        reciprocal_rank(retrieved, relevant)
        for retrieved, relevant in zip(list_of_retrieved, list_of_relevant)
    ]
    return sum(scores) / len(scores)


def mean(values: Sequence[Optional[Number]]) -> Optional[float]:
    filtered = [v for v in values if v is not None]
    if not filtered:
        return None
    return sum(filtered) / len(filtered)
