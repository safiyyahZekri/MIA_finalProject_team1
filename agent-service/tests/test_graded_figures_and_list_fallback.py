"""EXTRACT_GRADED_FIGURES and CALC_LIST_FALLBACK change only what they name.

With both off, extraction is called and prompted exactly as before, so runs
measured without them stay comparable.
"""

from types import SimpleNamespace

import pytest

from app import graph
from app import llm as llm_module
from app.llm import (
    AnthropicLLM,
    Classification,
    EvidenceGrade,
    ExtractionCalculated,
    ExtractionMultiSpan,
)

NOTE = "Plexus Corp.'s 2018 work-in-process is $102,337 thousand, from plexus p1."
EVIDENCE = [{"document_id": "plexus", "page": 1, "text": "Work-in-process | 102,337"}]
CALCULATED = '{"formula": "788742-102337", "operand_count": 2, "evidence_indexes": [1]}'
ADDED = (
    "An evidence check read these passages before you and approved them for this "
    f'question, noting: "{NOTE}"\n'
    "When that note names a figure needed as an operand and the passage it comes "
    "from, take that operand from that passage.\n"
)


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


@pytest.fixture(autouse=True)
def switches_off(monkeypatch):
    monkeypatch.setattr(llm_module.settings, "ANTHROPIC_EFFORT", "")
    monkeypatch.setattr(llm_module.settings, "ANTHROPIC_FALLBACKS", "")
    for flag in ("EXTRACT_GRADED_FIGURES", "CALC_LIST_FALLBACK", "QUERY_DECOMPOSITION"):
        monkeypatch.setattr(graph.settings, flag, False)
    monkeypatch.setattr(graph.settings, "MAX_RETRIES", 0)


def _calculation_prompt(**kwargs):
    llm = AnthropicLLM.__new__(AnthropicLLM)
    messages = FakeMessages(CALCULATED)
    llm._client = SimpleNamespace(beta=SimpleNamespace(messages=messages))
    llm._transform_schema = lambda schema: {"title": schema.__name__}
    llm.last_usage = {}
    llm.extract("How far apart were the balances?", "numerical", EVIDENCE, **kwargs)
    return messages.requests[-1]["messages"][0]["content"]


def test_the_grade_note_only_adds_its_own_block_to_the_calculation_prompt():
    without_note = _calculation_prompt()
    with_note = _calculation_prompt(grade_note=NOTE)

    assert "An evidence check read these passages" not in without_note
    assert ADDED in with_note
    assert with_note.replace(ADDED, "") == without_note


class Recorder:
    """A fake model that approves the evidence and records every extraction."""

    last_usage = {}

    def __init__(self, question_type="numerical", formula="788742-102337"):
        self.question_type = question_type
        self.formula = formula
        self.extractions = []

    def classify(self, question):
        return Classification(question_type=self.question_type, search_query="balances")

    def grade(self, question, evidence):
        return EvidenceGrade(sufficient=True, confidence=0.9, reason=NOTE)

    def reformulate(self, *args):
        return "retry"

    def extract(self, question, question_type, evidence, **kwargs):
        self.extractions.append((question_type, kwargs))
        if question_type == "numerical":
            return ExtractionCalculated(formula=self.formula, operand_count=2, evidence_indexes=[1])
        return ExtractionMultiSpan(
            values=["$12.2 million", "$23.6 million", "$38.6 million"], evidence_indexes=[1]
        )


async def _run(monkeypatch, model):
    monkeypatch.setattr(graph, "get_llm", lambda: model)

    async def search(query, document_id=None, **kwargs):
        return [
            {
                "document_id": "plexus",
                "chunk_id": "c1",
                "page": 1,
                "section": "Inventories",
                "text": "Work-in-process | 102,337",
                "score": 0.9,
            }
        ]

    for name in ("search_documents", "search_tables", "search_bm25"):
        monkeypatch.setattr(graph, name, search)
    return await graph.build_graph().ainvoke({"question": "How far apart?", "trace": []})


@pytest.mark.asyncio
async def test_without_the_switch_extraction_gets_no_grade_note(monkeypatch):
    model = Recorder()

    state = await _run(monkeypatch, model)

    assert model.extractions == [("numerical", {})]
    assert state["answer"]["answer_type"] == "calculated"


@pytest.mark.asyncio
async def test_the_switch_hands_the_approving_note_to_calculation_extraction(monkeypatch):
    monkeypatch.setattr(graph.settings, "EXTRACT_GRADED_FIGURES", True)
    model = Recorder()

    state = await _run(monkeypatch, model)

    assert model.extractions == [("numerical", {"grade_note": NOTE})]
    assert any(step.get("step") == "graded_figures" for step in state["trace"])


@pytest.mark.asyncio
async def test_the_grade_note_is_not_given_to_text_questions(monkeypatch):
    monkeypatch.setattr(graph.settings, "EXTRACT_GRADED_FIGURES", True)
    model = Recorder(question_type="text")

    await _run(monkeypatch, model)

    assert model.extractions == [("text", {})]


@pytest.mark.asyncio
async def test_without_the_fallback_a_list_of_values_is_declined(monkeypatch):
    model = Recorder(formula="38.6, 23.6, 12.2")

    state = await _run(monkeypatch, model)

    assert state["answer"]["answer_type"] == "insufficient_evidence"
    assert "Calculator rejected formula" in state["answer"]["params"]["reason"]
    assert len(model.extractions) == 1


@pytest.mark.asyncio
async def test_the_fallback_answers_a_list_of_values_as_a_list(monkeypatch):
    """A086: the question was classified numerical and extraction listed the
    three values instead of writing a formula."""
    monkeypatch.setattr(graph.settings, "CALC_LIST_FALLBACK", True)
    model = Recorder(formula="38.6, 23.6, 12.2")

    state = await _run(monkeypatch, model)

    assert [question_type for question_type, _ in model.extractions] == ["numerical", "table"]
    assert state["answer"]["answer_type"] == "multi_span"
    assert state["answer"]["params"]["values"] == ["$12.2 million", "$23.6 million", "$38.6 million"]


@pytest.mark.asyncio
async def test_the_fallback_leaves_a_broken_formula_declined(monkeypatch):
    monkeypatch.setattr(graph.settings, "CALC_LIST_FALLBACK", True)
    model = Recorder(formula="(38.6+")

    state = await _run(monkeypatch, model)

    assert state["answer"]["answer_type"] == "insufficient_evidence"
    assert len(model.extractions) == 1


@pytest.mark.parametrize(
    "formula, is_list",
    [
        ("38.6, 23.6, 12.2", True),
        ("12,200, 23,600", True),
        ("5%; 6%", True),
        ("12,200", False),
        ("(3875-3410)/3410*100", False),
        ("-3.5", False),
        ("38.6, 23.6+1", False),
    ],
)
def test_what_counts_as_a_list_of_values(formula, is_list):
    assert graph._is_value_list(formula) is is_list


def test_health_reports_both_switches():
    from fastapi.testclient import TestClient

    from app import main

    config = TestClient(main.app).get("/health").json()["config"]

    assert {"extract_graded_figures", "calc_list_fallback"} <= config.keys()
