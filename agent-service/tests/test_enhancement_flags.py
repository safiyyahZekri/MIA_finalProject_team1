"""Each enhancement switch changes only what it names, and all of them off
leaves the behaviour run `full-100-v1` measured.

The switches exist so an experiment can turn on one enhancement at a time;
a switch that leaked into another prompt would make that comparison
meaningless.
"""

from types import SimpleNamespace

import pytest

from app import graph
from app import llm as llm_module
from app.decomposition import decomposition_prompt
from app.llm import AnthropicLLM

FLAGS = ("ANSWER_FORMAT_FIXES", "GRADE_COMPANY_CONTEXT", "GRADE_REQUIRE_NAMED_TABLE", "RANK_FUSION_MERGE")
EVIDENCE = [{"document_id": "d", "page": 1, "text": "Revenue was 3,875."}]
GRADE = '{"sufficient": true, "confidence": 0.9, "reason": "r"}'
DIRECT = '{"shape": "direct", "value": "3,875"}'
CALCULATED = '{"formula": "3875-1200", "operand_count": 2}'


class FakeMessages:
    def __init__(self, text):
        self.text = text
        self.requests = []

    def create(self, **request):
        self.requests.append(request)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=self.text)],
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        )


def _prompt_for(call, text):
    llm = AnthropicLLM.__new__(AnthropicLLM)
    messages = FakeMessages(text)
    llm._client = SimpleNamespace(beta=SimpleNamespace(messages=messages))
    llm._transform_schema = lambda schema: {"title": schema.__name__}
    llm.last_usage = {}
    call(llm)
    return messages.requests[-1]["messages"][0]["content"]


def _grade_prompt():
    return _prompt_for(lambda llm: llm.grade("q", EVIDENCE), GRADE)


def _text_prompt():
    return _prompt_for(lambda llm: llm.extract("q", "text", EVIDENCE), DIRECT)


def _calculation_prompt():
    return _prompt_for(lambda llm: llm.extract("q", "numerical", EVIDENCE), CALCULATED)


@pytest.fixture(autouse=True)
def all_switches_off(monkeypatch):
    monkeypatch.setattr(llm_module.settings, "ANTHROPIC_EFFORT", "")
    monkeypatch.setattr(llm_module.settings, "ANTHROPIC_FALLBACKS", "")
    monkeypatch.setattr(llm_module.settings, "GRADE_REQUIRE_ENTITY_MATCH", True)
    for flag in FLAGS:
        monkeypatch.setattr(llm_module.settings, flag, False)


def test_all_switches_off_keep_the_measured_prompts():
    grade, text, calculation = _grade_prompt(), _text_prompt(), _calculation_prompt()

    assert "company are not sufficient.\n\nQuestion: q" in grade
    for added in ("Page numbers come from the citations", "names no company", "names a specific table"):
        assert added not in grade
    assert "Keep a scale word" not in text and "Do not put page numbers" not in text
    assert "take it from the table" not in calculation and "2017 to 2019" not in calculation


def test_answer_format_fixes(monkeypatch):
    """A086 lost "million", A013 put "p1" in its answer and was refused for a
    missing page, A014 used rounded text figures, A007 miscounted a period."""
    monkeypatch.setattr(llm_module.settings, "ANSWER_FORMAT_FIXES", True)

    assert "Page numbers come from the citations" in _grade_prompt()
    text = _text_prompt()
    assert "Keep a scale word" in text and "Do not put page numbers in the answer" in text
    calculation = _calculation_prompt()
    assert "take it from the table" in calculation and "from 2017 to 2019 means" in calculation


def test_company_context_replaces_the_strict_company_rule(monkeypatch):
    monkeypatch.setattr(llm_module.settings, "GRADE_COMPANY_CONTEXT", True)

    grade = _grade_prompt()

    assert "A passage that names no company may be accepted" in grade
    assert "never accept evidence that belongs to a different company" in grade
    assert "must be identifiably that company's" not in grade


def test_company_context_does_nothing_without_the_company_rule(monkeypatch):
    monkeypatch.setattr(llm_module.settings, "GRADE_COMPANY_CONTEXT", True)
    monkeypatch.setattr(llm_module.settings, "GRADE_REQUIRE_ENTITY_MATCH", False)

    grade = _grade_prompt()

    assert "names a company" not in grade
    assert "sufficient=false.\n\nQuestion: q" in grade


def test_named_table_rule(monkeypatch):
    monkeypatch.setattr(llm_module.settings, "GRADE_REQUIRE_NAMED_TABLE", True)

    assert "names a specific table, statement or section" in _grade_prompt()


def _hit(name, score):
    return {"document_id": name, "page": 1, "section": None, "score": score}


def test_raw_score_merge_drops_a_top_ranked_lexical_hit(monkeypatch):
    """The failure RANK_FUSION_MERGE fixes: BM25's first hit scores on a lower
    scale than the dense hits and loses its place in the merged five."""
    monkeypatch.setattr(graph.settings, "TOP_K_FINAL", 5)
    dense = [_hit(f"dense{i}", 0.9 - i / 10) for i in range(5)]
    lexical = [_hit("bm25-first", 0.3), _hit("bm25-second", 0.2)]

    merged = graph._dedup_and_rank(dense, lexical)

    assert "bm25-first" not in [h["document_id"] for h in merged]


def test_rank_fusion_merge_keeps_each_search_top_hit(monkeypatch):
    monkeypatch.setattr(graph.settings, "TOP_K_FINAL", 5)
    monkeypatch.setattr(graph.settings, "RANK_FUSION_MERGE", True)
    dense = [_hit(f"dense{i}", 0.9 - i / 10) for i in range(5)]
    lexical = [_hit("bm25-first", 0.3), _hit("dense1", 0.2)]

    merged = [h["document_id"] for h in graph._dedup_and_rank(dense, lexical)]

    # dense1 is found by both searches, so it leads; each search's own top
    # hit follows, ties broken by raw score.
    assert merged[:3] == ["dense1", "dense0", "bm25-first"]
    assert len(merged) == 5 and len(set(merged)) == 5


def test_decomposition_prompt_keeps_simple_lookups_whole():
    prompt = decomposition_prompt("What was revenue in 2019?")

    assert "Most questions are a single lookup: return an empty list" in prompt
    assert "do not split by year" in prompt
    assert '{"subqueries": ["..."]}' in prompt


def test_health_reports_every_switch():
    from fastapi.testclient import TestClient

    from app import main

    config = TestClient(main.app).get("/health").json()["config"]

    assert {
        "query_decomposition",
        "answer_format_fixes",
        "grade_company_context",
        "grade_require_named_table",
        "rank_fusion_merge",
    } <= config.keys()
