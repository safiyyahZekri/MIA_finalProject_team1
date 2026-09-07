from copy import deepcopy

from fastapi.testclient import TestClient

from app.main import create_app


def test_health_reports_retrieval_port_8002(engine) -> None:
    response = TestClient(create_app(engine)).get("/health")
    assert response.status_code == 200
    assert response.json()["port"] == 8002


def test_index_search_tables_and_stats(engine, sample_document: dict) -> None:
    client = TestClient(create_app(engine))

    indexed = client.post("/documents", json=sample_document)
    result = client.post(
        "/search/tables",
        json={
            "query": "finished goods 2019",
            "top_k": 1,
            "candidate_k": 2,
        },
    )
    stats = client.get("/stats")

    assert indexed.status_code == 201
    assert result.status_code == 200
    assert result.json()["hits"][0]["content_type"] == "table"
    assert result.json()["hits"][0]["page"] == 1
    assert stats.json()["documents"] == 1
    assert stats.json()["tables"] == 1


def test_rejects_candidate_pool_smaller_than_top_k(engine) -> None:
    client = TestClient(create_app(engine))
    response = client.post(
        "/search", json={"query": "revenue", "top_k": 5, "candidate_k": 2}
    )
    assert response.status_code == 422


def test_agent_compatibility_endpoints_and_public_contract(
    engine, sample_document: dict
) -> None:
    client = TestClient(create_app(engine))
    assert client.post("/documents", json=sample_document).status_code == 201

    for path in ("/search", "/search/vector", "/search/bm25", "/search/tables"):
        response = client.post(
            path,
            json={
                "query": "finished goods 2019",
                "top_k": 1,
                "document_id": "doc-cts-2019",
            },
        )
        assert response.status_code == 200, (path, response.text)
        hit = response.json()["hits"][0]
        assert 0.0 <= hit["score"] <= 1.0
        assert hit["score"] > 0.0
        assert hit["content_type"] in {"text", "table"}
        assert hit["text"] == hit["content"]
        assert hit["filename"] == "cts-corporation_2019.pdf"
        assert "scores" in hit


def test_flat_and_nested_document_filters_are_equivalent(
    engine, sample_document: dict
) -> None:
    client = TestClient(create_app(engine))
    client.post("/documents", json=sample_document)
    flat = client.post(
        "/search",
        json={"query": "inventory", "document_id": "missing", "rerank": False},
    )
    nested = client.post(
        "/search",
        json={
            "query": "inventory",
            "filters": {"document_id": "missing"},
            "rerank": False,
        },
    )
    assert flat.status_code == nested.status_code == 200
    assert flat.json()["hits"] == nested.json()["hits"] == []


def test_filter_endpoint_supports_agent_metadata_shape(
    engine, sample_document: dict
) -> None:
    client = TestClient(create_app(engine))
    client.post("/documents", json=sample_document)
    response = client.post(
        "/filter", json={"metadata": {"document_id": "doc-cts-2019"}, "top_k": 10}
    )
    assert response.status_code == 200
    assert response.json()["hits"]
    assert all(hit["score"] == 1.0 for hit in response.json()["hits"])


def test_internal_paragraph_is_exposed_as_public_text(
    engine, sample_document: dict
) -> None:
    client = TestClient(create_app(engine))
    client.post("/documents", json=sample_document)

    response = client.post(
        "/filter",
        json={"document_id": "doc-cts-2019", "content_type": "text", "top_k": 10},
    )

    assert response.status_code == 200
    assert response.json()["hits"]
    assert {hit["content_type"] for hit in response.json()["hits"]} == {"text"}


def test_batch_endpoint_indexes_multiple_documents(engine, sample_document: dict) -> None:
    second = deepcopy(sample_document)
    second["document"]["document_id"] = "doc-jabil"
    second["source_filename"] = "jabil.pdf"
    second["document"]["pages"][0]["blocks"][2]["uuid"] = "jabil-table"
    client = TestClient(create_app(engine))

    response = client.post(
        "/documents/batch", json={"documents": [sample_document, second]}
    )

    assert response.status_code == 201
    assert response.json()["documents_indexed"] == 2
    assert client.get("/stats").json()["documents"] == 2
