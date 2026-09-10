import pymupdf

import app as ui_app


def _one_page_pdf() -> bytes:
    document = pymupdf.open()
    page = document.new_page(width=100, height=100)
    page.insert_text((10, 20), "Revenue 2022: $42 million")
    value = document.tobytes()
    document.close()
    return value


def test_answer_renders_the_returned_evidence_box(monkeypatch) -> None:
    monkeypatch.setattr(
        ui_app.api_client,
        "ask",
        lambda question, document_id: {
            "answer_type": "direct",
            "params": {"value": "$42 million"},
            "evidence": [{"document_id": "report", "page": 1, "section": "Revenue"}],
            "evidence_boxes": [
                {
                    "document_id": "report",
                    "page": 1,
                    "section": "Revenue",
                    "bbox": [20, 20, 120, 60],
                }
            ],
        },
    )
    monkeypatch.setattr(
        ui_app.api_client, "download_document", lambda document_id: _one_page_pdf()
    )

    cleared, history, image, caption, review_context = ui_app.answer_with_evidence(
        "What was revenue?", [], "report"
    )

    assert cleared == ""
    assert history[-1]["role"] == "assistant"
    assert image.size == (200, 200)
    assert "Highlighted evidence" in caption
    assert review_context["question"] == "What was revenue?"
    assert review_context["document_id"] == "report"
    assert review_context["original_answer"]["params"]["value"] == "$42 million"


def test_human_correction_is_sent_with_the_answer_context(monkeypatch) -> None:
    captured = {}

    def save(**payload):
        captured.update(payload)
        return {"review_id": "review-123", "status": "recorded"}

    monkeypatch.setattr(ui_app.api_client, "submit_review", save)
    context = {
        "question": "What was revenue?",
        "document_id": "report",
        "original_answer": {
            "answer_type": "direct",
            "params": {"value": "$40 million"},
            "evidence": [{"document_id": "report", "page": 1}],
        },
    }

    status = ui_app.submit_review(
        context,
        "Needs correction",
        "  $42 million  ",
        "  Wrong table row.  ",
    )

    assert captured["verdict"] == "needs_correction"
    assert captured["corrected_answer"] == "$42 million"
    assert captured["comment"] == "Wrong table row."
    assert captured["original_answer"] == context["original_answer"]
    assert "review-123" in status
    assert "does not bypass answer validation" in status


def test_review_requires_an_answer_verdict_and_correction(monkeypatch) -> None:
    called = False

    def save(**payload):
        nonlocal called
        called = True

    monkeypatch.setattr(ui_app.api_client, "submit_review", save)
    context = {"question": "Q", "original_answer": {}, "document_id": None}

    assert "Ask a question" in ui_app.submit_review(None, "Correct", "", "")
    assert "Choose whether" in ui_app.submit_review(context, None, "", "")
    assert "corrected answer" in ui_app.submit_review(
        context, "Needs correction", "", ""
    )
    assert called is False


def test_dashboard_includes_persisted_human_reviews(monkeypatch) -> None:
    monkeypatch.setattr(
        ui_app.api_client,
        "list_documents",
        lambda: [{"document_id": "d", "filename": "d.pdf", "pages": 1, "tables": 2}],
    )
    monkeypatch.setattr(ui_app.api_client, "recent_queries", lambda: [])
    monkeypatch.setattr(
        ui_app.api_client,
        "list_reviews",
        lambda: [
            {
                "review_id": "r1",
                "verdict": "needs_correction",
                "question": "Revenue?",
                "corrected_answer": "$42 million",
                "created_at": "2026-09-10T10:00:00Z",
            }
        ],
    )

    stats, queries, reviews = ui_app.load_dashboard()

    assert "Human reviews:** 1" in stats
    assert queries == []
    assert reviews[0][:4] == [
        "r1",
        "needs_correction",
        "Revenue?",
        "$42 million",
    ]


def test_load_extracted_fields_populates_editor_and_history(monkeypatch) -> None:
    monkeypatch.setattr(
        ui_app.api_client,
        "get_extracted_fields",
        lambda document_id: [
            {
                "chunk_id": "chunk-1",
                "document_id": document_id,
                "page": 3,
                "section": "Revenue",
                "content_type": "table",
                "content": "Revenue | 2022 | 4Z",
            }
        ],
    )
    monkeypatch.setattr(
        ui_app.api_client, "list_extraction_corrections", lambda document_id: []
    )

    overview, field_map, selector, editor, history, status = (
        ui_app.load_extracted_fields(" doc-1 ")
    )

    assert overview[0][:4] == ["chunk-1", 3, "table", "Revenue"]
    assert field_map["chunk-1"]["content"] == "Revenue | 2022 | 4Z"
    assert selector.value == "chunk-1"
    assert editor == "Revenue | 2022 | 4Z"
    assert history == []
    assert "Loaded 1" in status


def test_save_extraction_correction_reindexes_and_returns_audit(monkeypatch) -> None:
    captured = {}

    def save(**payload):
        captured.update(payload)
        return {"correction_id": "correction-7", **payload}

    monkeypatch.setattr(ui_app.api_client, "submit_extraction_correction", save)
    monkeypatch.setattr(
        ui_app.api_client,
        "list_extraction_corrections",
        lambda document_id: [
            {
                "correction_id": "correction-7",
                "chunk_id": "chunk-1",
                "corrected_by": "reviewer",
                "created_at": "2026-09-10T12:00:00Z",
                "comment": "OCR typo",
            }
        ],
    )

    status, history = ui_app.save_extraction_correction(
        "doc-1", "chunk-1", "Revenue | 2022 | 42", "reviewer", "OCR typo"
    )

    assert captured["corrected_text"] == "Revenue | 2022 | 42"
    assert "retrieval index was rebuilt" in status
    assert history[0][:3] == ["correction-7", "chunk-1", "reviewer"]


def test_extraction_correction_requires_identity_and_text(monkeypatch) -> None:
    assert "Load and select" in ui_app.save_extraction_correction(
        "", "", "value", "reviewer", ""
    )[0]
    assert "cannot be empty" in ui_app.save_extraction_correction(
        "doc", "chunk", "", "reviewer", ""
    )[0]
    assert "name or team" in ui_app.save_extraction_correction(
        "doc", "chunk", "value", "", ""
    )[0]


def test_boxes_are_read_only_from_evidence_boxes(monkeypatch) -> None:
    """Citations never carry coordinates under the strict answer schema."""
    monkeypatch.setattr(
        ui_app.api_client,
        "ask",
        lambda question, document_id: {
            "answer_type": "direct",
            "params": {"value": "$42 million"},
            "evidence": [{"document_id": "report", "page": 1, "bbox": [20, 20, 120, 60]}],
        },
    )

    _, _, image, caption, _ = ui_app.answer_with_evidence("What was revenue?", [], "report")

    assert image is None
    assert "No bounding-box coordinates" in caption
