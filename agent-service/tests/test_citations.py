"""Citations must come from the evidence an answer was taken from.

The agent used to cite the top-ranked retrieval hits. In the Opus 5
experiments that produced correct answers whose cited documents did not
contain the figures -- A002's (2.5) and (4.5), A094's 508 / 151 / 397 sat in
lower-ranked chunks the model had read but the citation never pointed at.
"""

import pytest

from app import graph
from app.graph import _citations
from app.llm import Classification, EvidenceGrade, ExtractionDirect

EVIDENCE = [
    {"document_id": "top-ranked", "page": 1, "section": "S", "text": "no figures here"},
    {"document_id": "second", "page": 2, "section": None, "text": "nor here"},
    {"document_id": "holds-the-figures", "page": 3, "section": "Note 5", "text": "(2.5) (4.5)"},
]


def test_cites_the_passages_the_model_named():
    citations, cited_by_model = _citations(EVIDENCE, [3], fallback_n=2)

    assert cited_by_model is True
    assert citations == [{"document_id": "holds-the-figures", "page": 3, "section": "Note 5"}]


def test_out_of_range_repeated_and_boolean_indexes_are_dropped():
    citations, cited_by_model = _citations(EVIDENCE, [0, 4, -1, True, 3, 3], fallback_n=1)

    assert cited_by_model is True
    assert [c["document_id"] for c in citations] == ["holds-the-figures"]


def test_no_usable_index_falls_back_to_top_ranked_and_reports_it():
    citations, cited_by_model = _citations(EVIDENCE, [], fallback_n=2)

    assert cited_by_model is False
    assert [c["document_id"] for c in citations] == ["top-ranked", "second"]


def test_two_chunks_of_the_same_page_are_cited_once():
    evidence = EVIDENCE + [{"document_id": "holds-the-figures", "page": 3, "section": "Note 5", "text": "part 2"}]

    citations, _ = _citations(evidence, [3, 4], fallback_n=1)

    assert len(citations) == 1


def test_citation_keeps_retrieval_bounding_box():
    hit = {
        "document_id": "report",
        "page": 4,
        "section": "Revenue",
        "bbox": [10, 20, 300, 120],
    }

    citations, _ = _citations([hit], [1], fallback_n=1)

    assert citations == [hit]


class NamesSecondPassage:
    """An LLM whose extraction cites the second retrieved passage, not the first."""

    def classify(self, question):
        return Classification(question_type="text", search_query=question)

    def grade(self, question, evidence):
        return EvidenceGrade(sufficient=True, confidence=0.9, reason="enough")

    def reformulate(self, question, previous_query, attempt):
        return previous_query

    def extract(self, question, question_type, evidence):
        return ExtractionDirect(value="Marketing", evidence_indexes=[2])


@pytest.mark.asyncio
async def test_graph_cites_what_extraction_named_not_the_top_hit(monkeypatch):
    """Through the compiled graph, not just the helper: a helper that passes
    its own test while the node never calls it is the failure this guards."""
    monkeypatch.setattr(graph, "get_llm", lambda: NamesSecondPassage())

    final = await graph.build_graph().ainvoke(
        {"question": "Which expense categories increased year-over-year?", "trace": []}
    )

    assert len(final["evidence"]) >= 2, "needs a second passage to cite"
    assert final["answer"]["evidence"] == [
        {k: final["evidence"][1].get(k) for k in ("document_id", "page", "section")}
    ]
    reason = next(step for step in final["trace"] if step["step"] == "reason")
    assert reason["cited_by_model"] is True
