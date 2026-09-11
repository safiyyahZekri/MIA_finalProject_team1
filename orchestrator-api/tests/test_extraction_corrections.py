from fastapi.testclient import TestClient

from app import clients
from app.main import app


def _field() -> dict:
    return {
        "chunk_id": "chunk-1",
        "document_id": "doc-1",
        "page": 2,
        "section": "Revenue",
        "content_type": "table",
        "content": "Revenue | 2022 | 4Z",
        "bbox": [10, 20, 300, 100],
        "source_block_ids": ["table-1"],
    }


def test_extracted_field_routes_proxy_retrieval(monkeypatch) -> None:
    captured = {}

    async def get_fields(document_id):
        assert document_id == "doc-1"
        return [_field()]

    async def correct(document_id, payload):
        captured.update(payload)
        return {
            **payload,
            "correction_id": "correction-1",
            "document_id": document_id,
            "original_text": _field()["content"],
            "created_at": "2026-09-10T12:00:00Z",
            "status": "applied",
        }

    async def history(document_id):
        return []

    monkeypatch.setattr(clients, "get_extracted_fields", get_fields)
    monkeypatch.setattr(clients, "apply_extraction_correction", correct)
    monkeypatch.setattr(clients, "list_extraction_corrections", history)
    client = TestClient(app)

    assert client.get("/documents/doc-1/chunks").json() == [_field()]
    response = client.post(
        "/documents/doc-1/corrections",
        json={
            "chunk_id": "chunk-1",
            "corrected_text": "Revenue | 2022 | 42",
            "corrected_by": "reviewer@example.com",
            "comment": "OCR fix",
        },
    )
    assert response.status_code == 200
    assert response.json()["status"] == "applied"
    assert captured["corrected_text"] == "Revenue | 2022 | 42"
    assert client.get("/documents/doc-1/corrections").json() == []


def test_extracted_field_proxy_preserves_upstream_not_found(monkeypatch) -> None:
    async def missing(document_id):
        raise clients.ServiceIntegrationError(
            "retrieval_correction", "document not found", 404
        )

    monkeypatch.setattr(clients, "get_extracted_fields", missing)
    response = TestClient(app).get("/documents/missing/chunks")

    assert response.status_code == 404
    assert response.json()["detail"]["upstream_status"] == 404
