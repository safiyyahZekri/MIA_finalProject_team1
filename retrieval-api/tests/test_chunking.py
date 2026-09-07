from app.chunking import ChunkingConfig, build_chunks
from app.models import IndexDocumentRequest


def test_table_chunk_preserves_headers_citation_and_parent(
    sample_document: dict,
) -> None:
    request = IndexDocumentRequest.model_validate(sample_document)
    chunks = build_chunks(request, ChunkingConfig(max_chars=180, overlap_blocks=1))

    table = next(chunk for chunk in chunks if chunk.content_type == "table")
    assert "Item | 2019 | 2018" in table.text
    assert "Finished goods | 9,447 | 8,100" in table.text
    assert table.document_id == "doc-cts-2019"
    assert table.source_filename == "cts-corporation_2019.pdf"
    assert table.page == 1
    assert table.section == "Inventory"
    assert table.bbox == (10, 120, 900, 400)
    assert table.source_block_ids == ["table-1"]


def test_chunk_ids_are_deterministic(sample_document: dict) -> None:
    request = IndexDocumentRequest.model_validate(sample_document)
    config = ChunkingConfig(max_chars=180, overlap_blocks=1)
    first = build_chunks(request, config)
    second = build_chunks(request, config)
    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]


def test_large_table_splits_on_rows_and_repeats_header(sample_document: dict) -> None:
    cells = sample_document["document"]["pages"][0]["blocks"][2]["cells"]
    cells.extend(
        [
            {
                "bbox": [10, 200, 200, 230],
                "text": "Work in process",
                "row_span": [2, 3],
                "col_span": [0, 1],
            },
            {
                "bbox": [210, 200, 300, 230],
                "text": "2,500",
                "row_span": [2, 3],
                "col_span": [1, 2],
            },
            {
                "bbox": [310, 200, 400, 230],
                "text": "2,200",
                "row_span": [2, 3],
                "col_span": [2, 3],
            },
            {
                "bbox": [10, 240, 200, 270],
                "text": "Raw materials",
                "row_span": [3, 4],
                "col_span": [0, 1],
            },
            {
                "bbox": [210, 240, 300, 270],
                "text": "1,900",
                "row_span": [3, 4],
                "col_span": [1, 2],
            },
            {
                "bbox": [310, 240, 400, 270],
                "text": "1,700",
                "row_span": [3, 4],
                "col_span": [2, 3],
            },
        ]
    )
    request = IndexDocumentRequest.model_validate(sample_document)
    chunks = build_chunks(request, ChunkingConfig(max_chars=55, overlap_blocks=0))
    tables = [chunk for chunk in chunks if chunk.content_type == "table"]

    assert len(tables) > 1
    assert all("Item | 2019 | 2018" in table.text for table in tables)
    assert sum("Work in process" in table.text for table in tables) == 1


def test_multi_row_merged_headers_and_units_are_preserved(sample_document: dict) -> None:
    table = sample_document["document"]["pages"][0]["blocks"][2]
    table["cells"] = [
        {"bbox": [0, 0, 10, 10], "text": "Metric", "row_span": [0, 2], "col_span": [0, 1]},
        {"bbox": [10, 0, 30, 10], "text": "Year ended", "row_span": [0, 1], "col_span": [1, 3]},
        {"bbox": [10, 10, 20, 20], "text": "2022", "row_span": [1, 2], "col_span": [1, 2]},
        {"bbox": [20, 10, 30, 20], "text": "2021", "row_span": [1, 2], "col_span": [2, 3]},
        {"bbox": [0, 20, 10, 30], "text": "Operating income", "row_span": [2, 3], "col_span": [0, 1]},
        {"bbox": [10, 20, 20, 30], "text": "12,500", "row_span": [2, 3], "col_span": [1, 2]},
        {"bbox": [20, 20, 30, 30], "text": "10,200", "row_span": [2, 3], "col_span": [2, 3]},
    ]
    request = IndexDocumentRequest.model_validate(sample_document)
    chunks = build_chunks(request, ChunkingConfig(max_chars=180, overlap_blocks=0))
    tables = [chunk for chunk in chunks if chunk.content_type == "table"]

    assert tables
    assert all("Metric | Year ended | Year ended" in chunk.text for chunk in tables)
    assert all("Metric | 2022 | 2021" in chunk.text for chunk in tables)
    assert any("in thousands of dollars" in (chunk.table_context or "") for chunk in tables)
    assert all(chunk.table_cells for chunk in tables)


def test_heading_fallback_rejects_short_financial_value_line(sample_document: dict) -> None:
    value_line = sample_document["document"]["pages"][0]["blocks"][1]
    value_line["text"] = "Total Equity 9,447"
    request = IndexDocumentRequest.model_validate(sample_document)
    chunks = build_chunks(request, ChunkingConfig())
    table = next(chunk for chunk in chunks if chunk.content_type == "table")
    assert table.section == "Inventory"
