from __future__ import annotations

import itertools
import json

import pytest
from fastapi.testclient import TestClient

from app.chunking import ChunkingConfig, build_chunks, rebuild_chunks
from app.engine import RetrievalEngine
from app.index_format import LEGACY, IndexFormat, searchable_text
from app.main import create_app
from app.models import Chunk, IndexDocumentRequest

from .fakes import FakeEmbedder, FakeReranker

CONFIG = ChunkingConfig(max_chars=180, overlap_blocks=1)
ALL_FORMATS = [
    IndexFormat(*flags)
    for flags in itertools.product([False, True], repeat=len(IndexFormat.names()))
]
UID = "f00ffe64cc08889e9c4471c96040345c"


def _block(uuid: str, order: int, text: str = "", content_type: str = "paragraph", cells=None) -> dict:
    return {
        "bbox": [10, 10 * order, 900, 10 * order + 30],
        "uuid": uuid,
        "text": text,
        "order": order,
        "content_type": content_type,
        "cells": cells or [],
    }


def _cells(rows: list[list[str]]) -> list[dict]:
    return [
        {
            "bbox": [c * 100, r * 30, c * 100 + 90, r * 30 + 25],
            "text": value,
            "row_span": [r, r + 1],
            "col_span": [c, c + 1],
        }
        for r, row in enumerate(rows)
        for c, value in enumerate(row)
        if value
    ]


def filing() -> dict:
    """Two pages of a filing: footers naming the issuer, a table with a title
    row that splits, a customer named in prose, a table with no cells and an
    untitled table on a later page."""
    inventory = [["Components of inventories", "", ""], ["", "2019", "2018"]] + [
        [f"Component line {n}", f"{n},100", f"{n},050"] for n in range(1, 7)
    ]
    return {
        "document": {
            "document_id": f"sha256-{'0' * 64}",
            "pages": [
                {
                    "bbox": [0, 0, 1200, 1600],
                    "page_number": 1,
                    "blocks": [
                        _block("footer-1", 1, "CTS CORPORATION 40"),
                        _block("heading-1", 2, "Inventory"),
                        _block("para-1", 3, "Components of inventories are stated at the lower of cost or net realizable value."),
                        _block(
                            "para-2",
                            4,
                            "The following table summarizes the inventories of the company (in thousands):",
                        ),
                        _block("table-1", 5, content_type="table", cells=_cells(inventory)),
                        _block("para-3", 6, "Customers include Nidec Corporation and Applied Materials, Inc."),
                        _block("table-2", 7, "Total inventories 9,447 8,100", content_type="table"),
                    ],
                },
                {
                    "bbox": [0, 0, 1200, 1600],
                    "page_number": 2,
                    "blocks": [
                        _block("footer-2", 1, "CTS CORPORATION 41"),
                        _block(
                            "table-3",
                            2,
                            content_type="table",
                            cells=_cells([["Item", "2019", "2018"], ["Finished goods", "9,447", "8,100"]]),
                        ),
                        _block("heading-2", 3, "REVENUE"),
                        _block(
                            "para-4",
                            4,
                            "Revenue is recognized when control of the promised goods transfers to the "
                            "customer, which generally occurs upon shipment.",
                        ),
                        # The unit phrase spans the context's line break, as in
                        # ten TAT-DQA documents: "Units: in\nthousands".
                        _block(
                            "para-5",
                            5,
                            "The following table reconciles changes in our asset retirement liability (in",
                        ),
                        _block("heading-3", 6, "thousands):"),
                        _block(
                            "table-4",
                            7,
                            content_type="table",
                            cells=_cells([["Balance at start of year", "5,382"], ["Accretion", "310"]]),
                        ),
                    ],
                },
            ],
        },
        "source_filename": f"{UID}.pdf",
        "source_doc_uid": UID,
        "metadata": {"source_path": f"dev\\{UID}.pdf"},
    }


def _request() -> IndexDocumentRequest:
    return IndexDocumentRequest.model_validate(filing())


def _dump(chunks: list[Chunk]) -> list[dict]:
    return [chunk.model_dump() for chunk in chunks]


def _paragraphs(chunks: list[Chunk]) -> list[tuple[str, str]]:
    return [(chunk.chunk_id, chunk.text) for chunk in chunks if chunk.content_type == "paragraph"]


def _engine(path, index_format: IndexFormat | None = None) -> RetrievalEngine:
    return RetrievalEngine(
        data_dir=path,
        embedder=FakeEmbedder(),
        reranker=FakeReranker(),
        chunking=CONFIG,
        index_format=index_format,
    )


def test_the_filing_exercises_every_rebuild_path():
    tables = [chunk for chunk in build_chunks(_request(), CONFIG) if chunk.content_type == "table"]
    assert sum(chunk.source_block_ids == ["table-1"] for chunk in tables) > 1
    assert any("\n" in (chunk.table_context or "") for chunk in tables)
    assert any(chunk.text.startswith("Table: Inventory\n") for chunk in tables)
    assert any("\nUnits: in thousands\n" in chunk.text for chunk in tables)
    assert any("\nUnits: in\nthousands\n" in chunk.text for chunk in tables)
    assert any(not chunk.table_cells for chunk in tables)


@pytest.mark.parametrize("index_format", ALL_FORMATS, ids=lambda value: value.describe())
def test_rebuilding_legacy_chunks_equals_indexing_the_document_in_that_format(index_format):
    legacy = build_chunks(_request(), CONFIG)
    expected = build_chunks(_request(), CONFIG, index_format)
    assert _dump(rebuild_chunks(legacy, CONFIG, index_format)) == _dump(expected)


def test_each_switch_changes_only_what_it_is_for():
    legacy = build_chunks(_request(), CONFIG)
    full = build_chunks(
        _request(), CONFIG, IndexFormat(contextual_text=True, company_metadata=True, table_titles=True)
    )
    assert all("company" not in chunk.metadata for chunk in legacy)
    assert {chunk.metadata.get("company") for chunk in full} == {"CTS CORPORATION"}
    inventory = [chunk for chunk in full if chunk.source_block_ids == ["table-1"]]
    assert all(chunk.table_title == "Components of inventories" for chunk in inventory)
    assert all(chunk.text.startswith("Table: Components of inventories\n") for chunk in inventory)
    untitled = next(chunk for chunk in full if chunk.source_block_ids == ["table-3"])
    assert untitled.table_title == "Inventory"
    assert _paragraphs(full) == _paragraphs(legacy)


def test_legacy_indexed_text_is_unchanged():
    chunk = build_chunks(_request(), CONFIG)[0]
    metadata = " ".join(str(value) for value in chunk.metadata.values() if value is not None)
    legacy_text = f"{chunk.source_filename} {metadata} {chunk.section} {chunk.text}"
    assert searchable_text(chunk, LEGACY) == legacy_text
    assert RetrievalEngine._searchable_text(chunk) == legacy_text


def test_contextual_text_leads_with_company_and_section_and_drops_opaque_ids():
    index_format = IndexFormat(contextual_text=True, company_metadata=True)
    chunk = next(
        chunk
        for chunk in build_chunks(_request(), CONFIG, index_format)
        if chunk.content_type == "paragraph" and chunk.section == "Inventory"
    )
    assert searchable_text(chunk, index_format) == f"CTS CORPORATION Inventory {chunk.text}"


def test_contextual_text_keeps_meaningful_upload_metadata_once(sample_document):
    index_format = IndexFormat(contextual_text=True)
    chunk = build_chunks(IndexDocumentRequest.model_validate(sample_document), CONFIG, index_format)[0]
    assert searchable_text(chunk, index_format) == (
        f"CTS Corporation Inventory cts-corporation_2019.pdf 2019 {chunk.text}"
    )


def test_a_company_supplied_at_upload_is_kept(sample_document):
    chunks = build_chunks(
        IndexDocumentRequest.model_validate(sample_document), CONFIG, IndexFormat(company_metadata=True)
    )
    assert {chunk.metadata["company"] for chunk in chunks} == {"CTS Corporation"}


def test_rebuild_refuses_chunks_the_legacy_format_did_not_build():
    titled = build_chunks(_request(), CONFIG, IndexFormat(table_titles=True))
    with pytest.raises(ValueError):
        rebuild_chunks(titled, CONFIG, LEGACY)


def test_rebuild_takes_one_document_at_a_time(sample_document):
    mixed = build_chunks(_request(), CONFIG) + build_chunks(
        IndexDocumentRequest.model_validate(sample_document), CONFIG
    )
    with pytest.raises(ValueError):
        rebuild_chunks(mixed, CONFIG, LEGACY)


def test_format_names_parse_and_round_trip_through_the_manifest():
    assert IndexFormat.parse("legacy") == LEGACY
    assert IndexFormat.parse("") == LEGACY
    index_format = IndexFormat.parse("table_titles, contextual_text")
    assert index_format == IndexFormat(contextual_text=True, table_titles=True)
    assert index_format.describe() == "contextual_text,table_titles"
    assert IndexFormat.from_manifest(index_format.to_manifest()) == index_format
    assert IndexFormat.from_manifest(None) == LEGACY


@pytest.mark.parametrize("spec", ["bogus", "legacy,table_titles"])
def test_unknown_format_names_are_refused(spec):
    with pytest.raises(ValueError):
        IndexFormat.parse(spec)


@pytest.mark.parametrize("value", [{"sql_tables": True}, {"table_titles": "yes"}, ["table_titles"]])
def test_a_recorded_format_this_code_cannot_serve_is_refused(value):
    with pytest.raises((TypeError, ValueError)):
        IndexFormat.from_manifest(value)


def test_the_index_keeps_its_format_whatever_the_next_process_is_configured_with(tmp_path, sample_document):
    index_format = IndexFormat(contextual_text=True, company_metadata=True)
    _engine(tmp_path, index_format).index_document(_request())
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["index_format"] == index_format.to_manifest()

    reopened = _engine(tmp_path)
    assert reopened.index_format == index_format
    assert reopened.stats().index_format == ["contextual_text", "company_metadata"]

    reopened.index_document(IndexDocumentRequest.model_validate(sample_document))
    stored = [chunk for chunk in reopened._chunks if chunk.document_id == "doc-cts-2019"]
    expected = build_chunks(IndexDocumentRequest.model_validate(sample_document), CONFIG, index_format)
    assert _dump(stored) == _dump(expected)
    assert reopened._index_text(stored[0]) == searchable_text(stored[0], index_format)


def test_an_index_without_a_recorded_format_is_served_as_legacy(tmp_path):
    _engine(tmp_path).index_document(_request())
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["index_format"] == LEGACY.to_manifest()
    del manifest["index_format"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert _engine(tmp_path, IndexFormat(contextual_text=True)).index_format == LEGACY


def test_a_new_empty_index_takes_the_configured_format(tmp_path):
    index_format = IndexFormat(table_titles=True)
    assert _engine(tmp_path, index_format).index_format == index_format


def test_health_reports_the_index_format(engine):
    assert TestClient(create_app(engine)).get("/health").json()["index_format"] == "legacy"
