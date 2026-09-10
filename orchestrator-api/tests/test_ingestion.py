from __future__ import annotations

import hashlib

from fastapi.testclient import TestClient

from app import clients, document_store
from app.main import app


PDF_BYTES = b"%PDF-1.4\n% retrieval integration fixture\n%%EOF"


def test_pdf_is_processed_indexed_and_identity_is_preserved(monkeypatch, tmp_path) -> None:
    captured = {}

    async def fake_process(content, filename, document_id, source_doc_uid):
        captured["processor"] = (content, filename, document_id, source_doc_uid)
        return {
            "document_id": document_id,
            "source_doc_uid": source_doc_uid,
            "original_filename": filename,
            "pages": [{"bbox": [0, 0, 10, 10], "page_number": 1, "blocks": []}],
        }

    async def fake_index(
        processed, filename, document_id, source_doc_uid, metadata
    ):
        captured["index"] = (
            processed,
            filename,
            document_id,
            source_doc_uid,
            metadata,
        )
        return {
            "document_id": document_id,
            "chunks_indexed": 3,
            "chunks_replaced": 0,
        }

    monkeypatch.setattr(clients, "process_pdf", fake_process)
    monkeypatch.setattr(clients, "index_processed_document", fake_index)
    monkeypatch.setenv("DOCUMENT_STORE_DIR", str(tmp_path))
    response = TestClient(app).post(
        "/documents/ingest",
        files={"file": ("report.pdf", PDF_BYTES, "application/pdf")},
        data={
            "source_doc_uid": "tatdqa-123",
            "metadata_json": '{"company":"Example"}',
        },
    )

    assert response.status_code == 200
    assert response.json()["document_id"] == "tatdqa-123"
    assert response.json()["status"] == "indexed"
    assert captured["processor"][2:] == ("tatdqa-123", "tatdqa-123")
    assert captured["index"][4]["original_filename"] == "report.pdf"
    assert document_store.pdf_path("tatdqa-123").read_bytes() == PDF_BYTES


def test_stored_source_pdf_can_be_downloaded(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("DOCUMENT_STORE_DIR", str(tmp_path))
    document_store.save_pdf("report-id", PDF_BYTES)

    response = TestClient(app).get("/documents/report-id/file")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content == PDF_BYTES


def test_missing_source_pdf_returns_structured_404(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("DOCUMENT_STORE_DIR", str(tmp_path))

    response = TestClient(app).get("/documents/not-indexed/file")

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "source_pdf_not_found"


def test_generic_upload_uses_stable_content_hash(monkeypatch, tmp_path) -> None:
    identities = []

    async def fake_process(content, filename, document_id, source_doc_uid):
        identities.append(document_id)
        return {
            "document_id": document_id,
            "pages": [{"bbox": [0, 0, 10, 10], "page_number": 1, "blocks": []}],
        }

    async def fake_index(processed, filename, document_id, source_doc_uid, metadata):
        return {
            "document_id": document_id,
            "chunks_indexed": 1,
            "chunks_replaced": 1 if len(identities) > 1 else 0,
        }

    monkeypatch.setattr(clients, "process_pdf", fake_process)
    monkeypatch.setattr(clients, "index_processed_document", fake_index)
    monkeypatch.setenv("DOCUMENT_STORE_DIR", str(tmp_path))
    client = TestClient(app)
    first = client.post(
        "/documents/ingest",
        files={"file": ("report.pdf", PDF_BYTES, "application/pdf")},
    )
    second = client.post(
        "/documents/ingest",
        files={"file": ("renamed.pdf", PDF_BYTES, "application/pdf")},
    )

    expected = f"sha256-{hashlib.sha256(PDF_BYTES).hexdigest()}"
    assert identities == [expected, expected]
    assert first.json()["status"] == "indexed"
    assert second.json()["status"] == "reindexed"


def test_blank_source_uid_falls_back_to_stable_hash(monkeypatch, tmp_path) -> None:
    captured = {}

    async def fake_process(content, filename, document_id, source_doc_uid):
        captured["identity"] = (document_id, source_doc_uid)
        return {
            "document_id": document_id,
            "pages": [{"bbox": [0, 0, 10, 10], "page_number": 1, "blocks": []}],
        }

    async def fake_index(processed, filename, document_id, source_doc_uid, metadata):
        return {"document_id": document_id, "chunks_indexed": 1, "chunks_replaced": 0}

    monkeypatch.setattr(clients, "process_pdf", fake_process)
    monkeypatch.setattr(clients, "index_processed_document", fake_index)
    monkeypatch.setenv("DOCUMENT_STORE_DIR", str(tmp_path))
    response = TestClient(app).post(
        "/documents/ingest",
        files={"file": ("report.pdf", PDF_BYTES, "application/pdf")},
        data={"source_doc_uid": "   "},
    )

    expected = f"sha256-{hashlib.sha256(PDF_BYTES).hexdigest()}"
    assert response.status_code == 200
    assert captured["identity"] == (expected, None)


def test_ingestion_reports_unsupported_and_upstream_errors(monkeypatch) -> None:
    client = TestClient(app)
    unsupported = client.post(
        "/documents/ingest", files={"file": ("notes.txt", b"no", "text/plain")}
    )

    async def failed_processor(*args, **kwargs):
        raise clients.ServiceIntegrationError("document_processor", "OCR failed", 500)

    monkeypatch.setattr(clients, "process_pdf", failed_processor)
    upstream = client.post(
        "/documents/ingest",
        files={"file": ("report.pdf", PDF_BYTES, "application/pdf")},
    )

    assert unsupported.status_code == 415
    assert unsupported.json()["detail"]["code"] == "unsupported_file"
    assert upstream.status_code == 502
    assert upstream.json()["detail"]["code"] == "document_processor_failed"
