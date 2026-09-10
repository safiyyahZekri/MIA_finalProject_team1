import pytest
from app import graph
from app.decomposition import QueryPlan, merge_query_results
from app.llm import (
    Classification,
    EvidenceGrade,
    ExtractionCalculated,
    GroqLLM,
    OllamaLLM,
)
from app.main import _usage_summary
from pydantic import ValidationError


def hit(doc, score=0.9, chunk="1"):
    return {
        "document_id": doc,
        "chunk_id": f"{doc}-{chunk}",
        "page": 1,
        "text": doc,
        "score": score,
        "bbox": [1, 2, 3, 4],
    }


def test_merge_keeps_evidence_from_each_operand_and_confidence():
    first = [hit("one", 0.99, str(i)) for i in range(5)]
    second = [hit("two", 0.6)]
    merged = merge_query_results([first, second, first], 5)
    assert {h["document_id"] for h in merged} == {"one", "two"}
    assert len(merged) == 5
    assert len({h["chunk_id"] for h in merged}) == 5
    assert next(h for h in merged if h["document_id"] == "two")["score"] == 0.6
    assert all(h["bbox"] == [1, 2, 3, 4] for h in merged)


def test_same_page_different_passages_survive_without_chunk_ids():
    a, b = hit("one"), hit("one")
    a.pop("chunk_id")
    b.pop("chunk_id")
    b["text"] = "another operand"
    assert len(merge_query_results([[a, a], [b]], 5)) == 2


def test_plan_is_bounded_and_deduplicated():
    assert QueryPlan(
        subqueries=[" A  revenue ", "a revenue", "B revenue"]
    ).subqueries == ["A revenue", "B revenue"]
    with pytest.raises(ValidationError):
        QueryPlan(subqueries=["a", "b", "c", "d"])
    with pytest.raises(ValidationError):
        QueryPlan(subqueries=["x" * 501])


@pytest.mark.parametrize("provider", [OllamaLLM, GroqLLM])
def test_json_providers_use_validated_query_plan(provider):
    llm = provider.__new__(provider)
    prompts = []

    def generate(prompt):
        prompts.append(prompt)
        return {"subqueries": ["Orion 2021 revenue", "Lyra 2021 revenue"]}

    llm._generate = generate
    assert len(llm.decompose("Compare Orion and Lyra 2021 revenue").subqueries) == 2
    assert "Do not answer" in prompts[0]


class Planner:
    last_usage = {}
    plan_calls = 0
    fail = False

    def classify(self, question):
        self.last_usage = {"prompt_tokens": 1, "completion_tokens": 1}
        return Classification(question_type="numerical", search_query="combined")

    def decompose(self, question):
        self.plan_calls += 1
        self.last_usage = {"prompt_tokens": 10, "completion_tokens": 5}
        if self.fail:
            raise ValueError("bad plan")
        return QueryPlan(subqueries=["Orion revenue", "Lyra revenue"])

    def grade(self, question, evidence):
        self.last_usage = {"prompt_tokens": 1, "completion_tokens": 1}
        sufficient = {"Orion", "Lyra"}.issubset({h["document_id"] for h in evidence})
        return EvidenceGrade(
            sufficient=sufficient, confidence=0.9, reason="both operands required"
        )

    def reformulate(self, *args):
        return "retry"

    def extract(self, *args):
        return ExtractionCalculated(
            formula="100-40", operand_count=2, evidence_indexes=[1, 2]
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("queries", [[], ["one lookup"]])
async def test_unsplit_plan_makes_only_original_retrieval_calls(monkeypatch, queries):
    planner, calls = Planner(), []
    planner.decompose = lambda question: QueryPlan(subqueries=queries)
    monkeypatch.setattr(graph, "get_llm", lambda: planner)
    monkeypatch.setattr(graph.settings, "QUERY_DECOMPOSITION", True)
    monkeypatch.setattr(graph.settings, "MAX_RETRIES", 0)

    async def search(query, document_id=None):
        calls.append(query)
        return []

    monkeypatch.setattr(graph, "search_documents", search)
    monkeypatch.setattr(graph, "search_tables", search)
    state = await graph.build_graph().ainvoke({"question": "Revenue?", "trace": []})
    assert calls == ["combined", "combined"]
    assert state["subqueries"] == []


@pytest.mark.asyncio
async def test_subquery_retrieval_error_is_not_hidden_as_abstention(monkeypatch):
    monkeypatch.setattr(graph, "get_llm", Planner)
    monkeypatch.setattr(graph.settings, "QUERY_DECOMPOSITION", True)

    async def search(query, document_id=None):
        if query == "Orion revenue":
            raise RuntimeError("retrieval service unavailable")
        return []

    monkeypatch.setattr(graph, "search_documents", search)
    monkeypatch.setattr(graph, "search_tables", search)
    with pytest.raises(RuntimeError, match="retrieval service unavailable"):
        await graph.build_graph().ainvoke({"question": "Compare revenues", "trace": []})


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_graph_decomposition_supplies_both_operands_and_preserves_scope(
    monkeypatch, enabled
):
    planner, calls = Planner(), []
    monkeypatch.setattr(graph, "get_llm", lambda: planner)
    monkeypatch.setattr(graph.settings, "QUERY_DECOMPOSITION", enabled)
    monkeypatch.setattr(graph.settings, "MAX_RETRIES", 0)

    async def search(query, document_id=None):
        calls.append((query, document_id))
        return (
            [hit(query.split()[0])]
            if query.endswith("revenue")
            else [hit("distractor")]
        )

    async def filtered(metadata):
        return []

    monkeypatch.setattr(graph, "search_documents", search)
    monkeypatch.setattr(graph, "search_tables", search)
    monkeypatch.setattr(graph, "filter_documents", filtered)
    state = await graph.build_graph().ainvoke(
        {"question": "Compare revenues", "document_id": "user-scope", "trace": []}
    )
    assert all(scope == "user-scope" for _, scope in calls)
    assert planner.plan_calls == int(enabled)
    if enabled:
        assert state["answer"]["params"]["value"] == 60
        assert {e["document_id"] for e in state["answer"]["evidence"]} == {
            "Orion",
            "Lyra",
        }
        # Citations keep the Strict Answer Schema: no bounding boxes.
        assert all("bbox" not in e for e in state["answer"]["evidence"])
        trace = next(s for s in state["trace"] if s["step"] == "decompose")
        assert _usage_summary([trace])["input_tokens"] == 10
    else:
        assert state["answer"]["answer_type"] == "insufficient_evidence"
        assert len(calls) == 2


@pytest.mark.asyncio
async def test_retry_reuses_subquery_evidence_without_replanning(monkeypatch):
    planner, calls = Planner(), []
    planner.grade = lambda *args: EvidenceGrade(
        sufficient=False, confidence=0.9, reason="need more facts"
    )
    monkeypatch.setattr(graph, "get_llm", lambda: planner)
    monkeypatch.setattr(graph.settings, "QUERY_DECOMPOSITION", True)
    monkeypatch.setattr(graph.settings, "MAX_RETRIES", 1)

    async def search(query, document_id=None):
        calls.append((query, document_id))
        return [hit(query)]

    monkeypatch.setattr(graph, "search_documents", search)
    monkeypatch.setattr(graph, "search_tables", search)
    state = await graph.build_graph().ainvoke(
        {"question": "Compare revenues", "trace": []}
    )
    assert planner.plan_calls == 1
    assert calls.count(("Orion revenue", None)) == 2  # vector + table, once each
    assert len(calls) == 8  # two main attempts plus two subqueries
    steps = [s for s in state["trace"] if s["step"] == "retrieve"]
    assert steps[-1]["reused_subquery_evidence"] is True
    assert "Orion revenue" in [h["document_id"] for h in state["evidence"]]
    assert state["answer"]["answer_type"] == "insufficient_evidence"


@pytest.mark.asyncio
async def test_failed_optional_planning_uses_original_search_and_logs_failure(
    monkeypatch,
):
    planner = Planner()
    planner.fail = True
    monkeypatch.setattr(graph, "get_llm", lambda: planner)
    monkeypatch.setattr(graph.settings, "QUERY_DECOMPOSITION", True)
    monkeypatch.setattr(graph.settings, "MAX_RETRIES", 0)

    async def search(*args, **kwargs):
        return []

    monkeypatch.setattr(graph, "search_documents", search)
    monkeypatch.setattr(graph, "search_tables", search)
    state = await graph.build_graph().ainvoke(
        {"question": "Compare revenues", "trace": []}
    )
    plan = next(s for s in state["trace"] if s["step"] == "decompose")
    assert plan["status"] == "fallback"
    assert plan["error_type"] == "ValueError"
    assert state["subqueries"] == []
