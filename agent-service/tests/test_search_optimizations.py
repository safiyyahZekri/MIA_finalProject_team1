import json
from unittest.mock import AsyncMock

import pytest
from app import graph
from app.answer_normalization import normalize_answer
from app.answer_repair import repair_extraction
from app.context_expansion import expand_context
from app.llm import (
    Classification,
    EvidenceGrade,
    ExtractionDirect,
    ExtractionInsufficient,
    ExtractionMultiSpan,
    GroqLLM,
    OllamaLLM,
)
from app.retrieval_client import RetrievalClient
from app.search_policy import (
    diversify,
    policy,
    profiles,
    query_variants,
)


def hit(key, text="Revenue 100 million", score=0.8, page=1):
    return {
        "chunk_id": key,
        "document_id": "company",
        "page": page,
        "section": "Report",
        "text": text,
        "score": score,
    }


@pytest.fixture(autouse=True)
def switches_off(monkeypatch):
    for flag in (
        "ADAPTIVE_TOP_K",
        "QUESTION_TYPE_RETRIEVAL",
        "EVIDENCE_DIVERSIFICATION",
        "QUERY_EXPANSION",
        "TABLE_QUERY_REWRITING",
        "ADJACENT_EVIDENCE",
        "ANSWER_REPAIR",
        "ANSWER_NORMALIZATION",
        "GRADE_CALIBRATION",
        "QUERY_DECOMPOSITION",
        "HYBRID_RERANKING",
        "ENTITY_DOCUMENT_ROUTING",
        "RETRY_EVIDENCE_FUSION",
    ):
        monkeypatch.setattr(graph.settings, flag, False)
    monkeypatch.setattr(graph.settings, "RETRIEVAL_PROFILES_JSON", "{}")
    monkeypatch.setattr(graph.settings, "TOP_K_FINAL", 5)
    monkeypatch.setattr(graph.settings, "TOP_K_OVERRETRIEVE", 30)


def test_adaptive_policy_grows_only_after_failure_and_is_capped(monkeypatch):
    monkeypatch.setattr(graph.settings, "ADAPTIVE_TOP_K", True)
    monkeypatch.setattr(graph.settings, "ADAPTIVE_TOP_K_MAX", 15)
    assert [policy("text", n).top_k for n in range(4)] == [5, 10, 15, 15]
    assert policy("text", 3).candidate_k >= 15


def test_type_profiles_validate_and_do_not_use_gold_answer_types(monkeypatch):
    monkeypatch.setattr(graph.settings, "QUESTION_TYPE_RETRIEVAL", True)
    monkeypatch.setattr(
        graph.settings,
        "RETRIEVAL_PROFILES_JSON",
        json.dumps({"numerical": {"top_k": 8, "candidate_k": 60, "dense_weight": 0.3}}),
    )
    assert policy("numerical").top_k == 8
    assert policy("text").top_k == 5
    monkeypatch.setattr(
        graph.settings, "RETRIEVAL_PROFILES_JSON", '{"unanswerable": {}}'
    )
    with pytest.raises(ValueError):
        profiles()
    monkeypatch.setattr(
        graph.settings, "RETRIEVAL_PROFILES_JSON", '{"text": {"top_k": 999}}'
    )
    with pytest.raises(ValueError):
        profiles()


def test_expansion_preserves_entity_year_units_and_original_query(monkeypatch):
    monkeypatch.setattr(graph.settings, "QUERY_EXPANSION", True)
    monkeypatch.setattr(graph.settings, "TABLE_QUERY_REWRITING", True)
    question = "What was CTS revenue in thousands for 2021?"
    variants = query_variants(question, "revenue", "table")
    assert len(variants) == 2
    assert all(
        all(token in v for token in ("CTS", "2021", "thousands", "revenue"))
        for v in variants
    )
    assert "net sales" in variants[0]
    assert query_variants("Why did CTS close the facility?", "why", "text") == []


def test_diversification_keeps_distinct_numbers_and_entity_anchors():
    hits = [
        hit("one", "Revenue 2021 100 million", 0.95),
        hit("copy", "Revenue 2021 100 million", 0.94),
        hit("two", "Cash flow 2020 90 million", 0.9),
        {**hit("entity", "Revenue 2022 200 million", 0.7), "routed_entity": "Other"},
    ]
    result = diversify(hits, 3)
    assert {h["chunk_id"] for h in result} == {"entity", "one", "two"}
    same_page = [hit("a", "Revenue 2021 100"), hit("b", "Revenue 2022 101")]
    assert len(diversify(same_page, 5)) == 2


@pytest.mark.asyncio
async def test_context_is_bounded_scoped_and_does_not_inherit_filter_confidence():
    async def fetch(metadata, top_k):
        return [
            hit(f"p{metadata['page']}-{n}", page=metadata["page"], score=1.0)
            for n in range(2)
        ] + [{**hit("wrong"), "document_id": "another"}]

    result, details = await expand_context([hit("anchor", page=2)], fetch)
    assert len(result) == 5
    assert details["context_hits"] == 4
    assert all(h["document_id"] == "company" for h in result)
    assert all(h["score"] == 0 and h["context_only"] for h in result[1:])
    assert result[0]["score"] == 0.8


def test_normalization_preserves_units_signs_precision_and_independent_values():
    answer = {
        "answer_type": "multi_span",
        "params": {"values": ["  $1,200  million ", "−0.1100 %", "2021–2022"]},
        "evidence": [],
    }
    result = normalize_answer(answer)
    assert result["params"]["values"] == ["$1,200 million", "-0.1100%", "2021–2022"]
    assert answer["params"]["values"][0].startswith("  ")


class RepairLLM:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    def extract(self, *args):
        self.calls += 1
        return self.response


def test_repair_singleton_shape_and_reject_changed_figures():
    original = ExtractionMultiSpan(values=["100 million"], evidence_indexes=[1])
    llm = RepairLLM(ExtractionDirect(value="100 million", evidence_indexes=[1]))
    fixed, status = repair_extraction(llm, "Revenue?", "text", [hit("1")], original)
    assert (
        isinstance(fixed, ExtractionDirect) and status == "accepted" and llm.calls == 1
    )
    llm = RepairLLM(ExtractionDirect(value="101 million", evidence_indexes=[1]))
    fixed, status = repair_extraction(llm, "Revenue?", "text", [hit("1")], original)
    assert fixed is original and status == "rejected_changed_facts"


def test_repair_rejects_unsupported_citation_and_decimal_change():
    original = ExtractionDirect(value="1,200", evidence_indexes=[])
    llm = RepairLLM(ExtractionDirect(value="1.200", evidence_indexes=[1]))
    assert (
        repair_extraction(llm, "q", "text", [hit("1", "1.200")], original)[1]
        == "rejected_changed_facts"
    )
    llm = RepairLLM(ExtractionDirect(value="1,200", evidence_indexes=[1]))
    assert (
        repair_extraction(llm, "q", "text", [hit("1", "nothing")], original)[1]
        == "rejected_unsupported"
    )


class RetryLLM:
    def classify(self, question):
        return Classification(question_type="text", search_query=question)

    def grade(self, question, evidence):
        enough = any(h["chunk_id"] == "9" for h in evidence)
        return EvidenceGrade(
            sufficient=enough, confidence=0.9, reason="needs last fact"
        )

    def reformulate(self, question, query, attempt):
        return query

    def extract(self, *args):
        return ExtractionDirect(value="100 million", evidence_indexes=[1])


@pytest.mark.asyncio
async def test_graph_retry_widens_and_recovers_missing_evidence(monkeypatch):
    monkeypatch.setattr(graph.settings, "ADAPTIVE_TOP_K", True)
    monkeypatch.setattr(graph.settings, "MAX_RETRIES", 1)
    monkeypatch.setattr(graph, "get_llm", RetryLLM)
    calls = []

    async def search(query, document_id=None, top_k=5, candidate_k=30):
        calls.append(top_k)
        return [hit(str(n), page=n + 1) for n in range(top_k)]

    monkeypatch.setattr(graph, "search_documents", search)
    monkeypatch.setattr(graph, "search_bm25", search)
    state = await graph.build_graph().ainvoke({"question": "Revenue?", "trace": []})
    assert calls == [5, 5, 10, 10]
    assert state["answer"]["answer_type"] == "direct"
    assert len(state["evidence"]) == 10


@pytest.mark.asyncio
async def test_grade_calibration_never_overrides_model_rejection(monkeypatch):
    monkeypatch.setattr(graph.settings, "GRADE_CALIBRATION", True)
    monkeypatch.setattr(graph.settings, "GRADE_MIN_SCORE", 0)
    monkeypatch.setattr(graph.settings, "GRADE_MIN_CONFIDENCE", 0)
    monkeypatch.setattr(graph.settings, "MAX_RETRIES", 0)
    monkeypatch.setattr(graph, "get_llm", RetryLLM)
    monkeypatch.setattr(graph, "search_documents", AsyncMock(return_value=[hit("1")]))
    monkeypatch.setattr(graph, "search_bm25", AsyncMock(return_value=[]))
    state = await graph.build_graph().ainvoke({"question": "q", "trace": []})
    assert state["answer"]["answer_type"] == "insufficient_evidence"
    grade = next(s for s in state["trace"] if s["step"] == "grade")
    assert grade["model_sufficient"] is False and grade["top_score"] == 0.8


@pytest.mark.asyncio
async def test_large_top_k_and_profile_fusion_options_reach_api(monkeypatch):
    client = RetrievalClient()
    post = AsyncMock(return_value=[])
    monkeypatch.setattr(client, "_post", post)
    await client.search_documents("q", top_k=50, candidate_k=70, document_id="scope")
    assert post.call_args.args[1] == {
        "query": "q",
        "top_k": 50,
        "candidate_k": 70,
        "document_id": "scope",
    }
    await client.search_hybrid("q", dense_weight=0.3, rrf_k=20)
    assert post.call_args.args[1]["dense_weight"] == 0.3
    assert post.call_args.args[1]["rrf_k"] == 20


@pytest.mark.parametrize("provider", [GroqLLM, OllamaLLM])
def test_repair_capable_providers_number_passages_and_retain_indexes(
    monkeypatch, provider
):
    monkeypatch.setattr(graph.settings, "ANSWER_REPAIR", True)
    llm = provider.__new__(provider)
    prompts = []

    def generate(prompt):
        prompts.append(prompt)
        return {"shape": "direct", "value": "100 million", "evidence_indexes": [2]}

    llm._generate = generate
    extraction = llm.extract("Revenue?", "text", [hit("1"), hit("2")])
    assert extraction.evidence_indexes == [2]
    assert "[2] company p1" in prompts[0]
    assert "Include evidence_indexes" in prompts[0]


def test_repair_preserves_an_explicit_model_abstention():
    llm = RepairLLM(ExtractionInsufficient(reason="wrong company"))
    fixed, status = repair_extraction(
        llm, "q", "text", [hit("1")], ExtractionDirect(value="100 million")
    )
    assert isinstance(fixed, ExtractionInsufficient) and status == "declined"


def test_repair_rejects_reversing_a_relationship_with_the_same_words():
    original = ExtractionDirect(value="revenue exceeds expense", evidence_indexes=[])
    llm = RepairLLM(
        ExtractionDirect(value="expense exceeds revenue", evidence_indexes=[1])
    )
    fixed, status = repair_extraction(
        llm, "q", "text", [hit("1", "revenue exceeds expense")], original
    )
    assert fixed is original and status == "rejected_changed_facts"


@pytest.mark.asyncio
async def test_scoped_adaptive_queries_keep_budget_and_ignore_metadata_confidence(
    monkeypatch,
):
    monkeypatch.setattr(graph.settings, "ADAPTIVE_TOP_K", True)
    monkeypatch.setattr(graph.settings, "GRADE_CALIBRATION", True)
    monkeypatch.setattr(graph.settings, "QUERY_EXPANSION", True)
    monkeypatch.setattr(graph.settings, "GRADE_MIN_SCORE", 0.9)
    monkeypatch.setattr(graph.settings, "MAX_RETRIES", 1)
    monkeypatch.setattr(graph, "get_llm", RetryLLM)
    calls = []

    async def search(query, document_id=None, top_k=5, candidate_k=30):
        calls.append((query, document_id, top_k))
        return [hit(str(n), page=n + 1) for n in range(top_k)]

    monkeypatch.setattr(graph, "search_documents", search)
    monkeypatch.setattr(graph, "search_bm25", search)
    monkeypatch.setattr(
        graph, "filter_documents", AsyncMock(return_value=[hit("metadata", score=1)])
    )
    state = await graph.build_graph().ainvoke(
        {"question": "CTS revenue 2021?", "document_id": "company", "trace": []}
    )
    assert all(doc == "company" for _, doc, _ in calls)
    assert any("net sales" in query for query, _, _ in calls)
    assert len(state["evidence"]) == 10
    assert state["answer"]["answer_type"] == "insufficient_evidence"
    assert all(
        step["top_score"] == 0.8 for step in state["trace"] if step["step"] == "grade"
    )
