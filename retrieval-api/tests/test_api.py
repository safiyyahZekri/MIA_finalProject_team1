from fastapi.testclient import TestClient

from app.main import create_app


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
