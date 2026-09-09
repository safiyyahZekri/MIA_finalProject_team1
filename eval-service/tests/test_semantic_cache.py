import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.semantic_cache import SemanticCache  # noqa: E402


def test_exact_repeat_is_a_hit():
    cache = SemanticCache()
    cache.set("What was revenue in 2019?", "doc_1", "cached-answer")
    hit, value = cache.get("What was revenue in 2019?", "doc_1")
    assert hit is True
    assert value == "cached-answer"
    assert cache.stats.exact_hits == 1
    assert cache.stats.misses == 0


def test_case_and_punctuation_insensitive_exact_match():
    cache = SemanticCache()
    cache.set("What was Revenue in 2019?", "doc_1", "answer")
    hit, value = cache.get("what was revenue in 2019", "doc_1")
    assert hit is True
    assert value == "answer"
    assert cache.stats.exact_hits == 1


def test_near_duplicate_phrasing_is_a_fuzzy_hit():
    # Real Jaccard similarity between these two normalized token sets is
    # 0.75 (computed, not assumed) -- threshold set below that.
    cache = SemanticCache(similarity_threshold=0.7)
    cache.set("What was the total revenue reported in 2019?", "doc_1", "answer")
    hit, value = cache.get("What was total revenue reported for 2019?", "doc_1")
    assert hit is True
    assert value == "answer"
    assert cache.stats.fuzzy_hits == 1


def test_unrelated_question_is_a_miss():
    cache = SemanticCache()
    cache.set("What was revenue in 2019?", "doc_1", "answer")
    hit, value = cache.get("What were operating expenses in 2020?", "doc_1")
    assert hit is False
    assert value is None
    assert cache.stats.misses == 1


def test_scoped_by_document_id_same_question_different_doc_is_a_miss():
    cache = SemanticCache()
    cache.set("What was revenue in 2019?", "doc_1", "answer-for-doc-1")
    hit, value = cache.get("What was revenue in 2019?", "doc_2")
    assert hit is False
    assert value is None


def test_below_threshold_similarity_is_a_miss():
    cache = SemanticCache(similarity_threshold=0.95)
    cache.set("What was the total revenue reported in fiscal year 2019?", "doc_1", "answer")
    # Shares some words but not enough to clear a strict 0.95 threshold.
    hit, _ = cache.get("What was the operating margin in fiscal year 2019?", "doc_1")
    assert hit is False


def test_stats_hit_rate():
    cache = SemanticCache()
    cache.set("q1", "doc_1", "a1")
    cache.get("q1", "doc_1")  # hit
    cache.get("q2", "doc_1")  # miss
    assert cache.stats.hits == 1
    assert cache.stats.misses == 1
    assert cache.stats.hit_rate == 0.5


def test_size_reflects_stored_entries():
    cache = SemanticCache()
    assert cache.size() == 0
    cache.set("q1", "doc_1", "a1")
    cache.set("q2", "doc_1", "a2")
    cache.set("q1", "doc_2", "a3")  # different scope, separate entry
    assert cache.size() == 3
