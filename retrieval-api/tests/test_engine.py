from copy import deepcopy
from pathlib import Path

from app.engine import RetrievalEngine
from app.models import IndexDocumentRequest, SearchRequest

from .fakes import FakeEmbedder, FakeReranker


def test_hybrid_table_search_returns_grounded_hit(
    engine, sample_document: dict
) -> None:
    engine.index_document(IndexDocumentRequest.model_validate(sample_document))

    result = engine.search(
        SearchRequest.model_validate(
            {
                "query": "What was CTS finished goods inventory in 2019?",
                "top_k": 1,
                "candidate_k": 2,
                "filters": {"content_type": "table"},
            }
        )
    )

    assert result.reranked is True
    assert result.hits[0].document_id == "doc-cts-2019"
    assert result.hits[0].page == 1
    assert "9,447" in result.hits[0].text
    assert result.hits[0].scores.lexical is not None
    assert result.hits[0].scores.dense is not None
    assert result.hits[0].scores.reranker is not None


def test_reindex_replaces_document_and_persists(
    engine, sample_document: dict, tmp_path: Path
) -> None:
    request = IndexDocumentRequest.model_validate(sample_document)
    first = engine.index_document(request)
    second = engine.index_document(request)
    reloaded = RetrievalEngine(tmp_path, FakeEmbedder(), FakeReranker())

    assert first.chunks_replaced == 0
    assert second.chunks_replaced == first.chunks_indexed
    assert reloaded.stats().documents == 1
    assert reloaded.stats().chunks == first.chunks_indexed


def test_metadata_filter_excludes_other_documents(
    engine, sample_document: dict
) -> None:
    engine.index_document(IndexDocumentRequest.model_validate(sample_document))
    result = engine.search(
        SearchRequest.model_validate(
            {
                "query": "inventory",
                "filters": {"metadata": {"company": "Different Company"}},
            }
        )
    )
    assert result.hits == []


def test_corpus_wide_query_can_return_evidence_from_two_documents(
    engine, sample_document: dict
) -> None:
    jabil = deepcopy(sample_document)
    jabil["document"]["document_id"] = "doc-jabil-2019"
    jabil["source_filename"] = "jabil-circuit-inc_2019.pdf"
    jabil["metadata"]["company"] = "Jabil Inc."
    jabil["document"]["pages"][0]["blocks"][2]["uuid"] = "jabil-table"
    jabil["document"]["pages"][0]["blocks"][2]["cells"][4]["text"] = "314,258"

    engine.index_document(IndexDocumentRequest.model_validate(sample_document))
    engine.index_document(IndexDocumentRequest.model_validate(jabil))
    result = engine.search(
        SearchRequest.model_validate(
            {
                "query": "2019 finished-goods balances reported by CTS and Jabil",
                "top_k": 2,
                "candidate_k": 4,
                "filters": {"content_type": "table"},
            }
        )
    )

    assert {hit.document_id for hit in result.hits} == {
        "doc-cts-2019",
        "doc-jabil-2019",
    }
