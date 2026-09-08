from unittest.mock import AsyncMock

import pytest

from app.llm import MockLLM
from app.retrieval_client import RetrievalClient


@pytest.mark.asyncio
async def test_agent_uses_compatible_vector_endpoint_and_normalized_score(monkeypatch):
    client = RetrievalClient()
    hit = {
        "chunk_id": "chunk-1",
        "document_id": "tat-1",
        "source_doc_uid": "tat-1",
        "filename": "report.pdf",
        "page": 2,
        "section": "Income Statement",
        "content_type": "table",
        "text": "Operating income in 2022 was 125 million.",
        "score": 0.82,
        "scores": {"dense": 0.7, "lexical": None, "fusion": 1.0, "reranker": None},
    }
    post = AsyncMock(return_value=[hit])
    monkeypatch.setattr(client, "_post", post)

    evidence = await client.search_documents(
        "What was operating income in 2022?", top_k=5, document_id="tat-1"
    )

    post.assert_awaited_once_with(
        "/search/vector",
        {
            "query": "What was operating income in 2022?",
            "top_k": 5,
            "document_id": "tat-1",
        },
    )
    grade = MockLLM().grade("What was operating income in 2022?", evidence)
    assert grade.sufficient is True
    assert grade.confidence == pytest.approx(0.82)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("search_documents", "/search/vector"),
        ("search_bm25", "/search/bm25"),
        ("search_tables", "/search/tables"),
    ],
)
async def test_agent_retrieval_paths_are_stable(monkeypatch, method, path):
    client = RetrievalClient()
    post = AsyncMock(return_value=[])
    monkeypatch.setattr(client, "_post", post)

    await getattr(client, method)("revenue", top_k=3)

    assert post.await_args.args[0] == path

