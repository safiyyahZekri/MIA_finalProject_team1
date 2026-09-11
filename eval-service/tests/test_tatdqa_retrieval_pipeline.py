from __future__ import annotations

from pathlib import Path

from scripts.tatdqa_retrieval_pipeline import source_doc_uid_for_pdf, stable_id


def test_official_tatdqa_filename_becomes_source_uid() -> None:
    uid = "7d631ffe5ff034d0ea9053d89a327ca3"

    assert source_doc_uid_for_pdf(Path(f"/corpus/dev/{uid}.pdf"), {}) == uid
    assert stable_id(b"pdf bytes", uid) == uid


def test_explicit_identity_mapping_takes_precedence() -> None:
    filename = "renamed-report.pdf"

    assert source_doc_uid_for_pdf(
        Path(filename), {filename: "external-dataset-id"}
    ) == "external-dataset-id"


def test_arbitrary_filename_keeps_content_hash_fallback() -> None:
    pdf_bytes = b"generic PDF bytes"

    assert source_doc_uid_for_pdf(Path("annual-report.pdf"), {}) is None
    assert stable_id(pdf_bytes, None).startswith("sha256-")
