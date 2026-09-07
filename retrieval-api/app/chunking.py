from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from .models import BBox, Cell, Chunk, DocumentBlock, IndexDocumentRequest

STRUCTURAL_HEADINGS = {
    "assets",
    "cash flows",
    "equity",
    "expenses",
    "financial position",
    "financial statements",
    "income statement",
    "inventory",
    "liabilities",
    "operating expenses",
    "revenue",
    "results of operations",
}
UNIT_PATTERN = re.compile(
    r"(?i)(?:amounts?\s+)?(?:in|expressed in)\s+"
    r"(?:u\.s\.\s+)?(?:dollars?|thousands?|millions?|billions?|percent(?:ages?)?)"
)
YEAR_PATTERN = re.compile(r"^(?:19|20)\d{2}$")


@dataclass(frozen=True)
class ChunkingConfig:
    max_chars: int = 1800
    overlap_blocks: int = 1


@dataclass(frozen=True)
class SerializedTable:
    rows: list[str]
    header_row_count: int
    units: str | None


def _looks_like_heading(block: DocumentBlock, text: str) -> bool:
    """Conservative fallback when the processor has no explicit heading label."""
    if block.content_type == "heading":
        return True
    text = " ".join(text.split())
    words = text.split()
    if not text or len(text) > 100 or len(words) > 12:
        return False
    if re.search(r"[$%\d]", text) or text.endswith((".", ";", ",")):
        return False
    letters = [character for character in text if character.isalpha()]
    if not letters:
        return False
    uppercase_ratio = sum(character.isupper() for character in letters) / len(letters)
    numbered = bool(re.match(r"^(?:item\s+)?(?:\d+(?:\.\d+)*|[IVX]+)[.)]?\s+", text, re.I))
    colon_heading = text.endswith(":") and len(words) <= 8
    known_heading = text.lower().rstrip(":") in STRUCTURAL_HEADINGS
    return uppercase_ratio >= 0.85 or numbered or colon_heading or known_heading


def _bbox_union(blocks: list[DocumentBlock]) -> BBox:
    return (
        min(block.bbox[0] for block in blocks),
        min(block.bbox[1] for block in blocks),
        max(block.bbox[2] for block in blocks),
        max(block.bbox[3] for block in blocks),
    )


def _chunk_id(
    document_id: str,
    page: int,
    section: str,
    content_type: str,
    block_ids: list[str],
    text: str,
) -> str:
    identity = "\x1f".join(
        [document_id, str(page), section, content_type, *block_ids, text]
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]


def _clean(value: str) -> str:
    return " ".join(value.split())


def _infer_header_rows(cells: list[Cell], row_count: int) -> int:
    if row_count <= 1:
        return 1
    first_row = [cell for cell in cells if cell.row_span[0] == 0]
    has_hierarchy = any(
        cell.row_span[1] - cell.row_span[0] > 1
        or cell.col_span[1] - cell.col_span[0] > 1
        for cell in first_row
    )
    second_row = [
        _clean(cell.text)
        for cell in cells
        if cell.row_span[0] == 1 and _clean(cell.text)
    ]
    second_row_is_period_header = bool(second_row) and sum(
        bool(YEAR_PATTERN.match(value.strip("()"))) for value in second_row
    ) >= max(1, len(second_row) // 2)
    return 2 if has_hierarchy or second_row_is_period_header else 1


def _serialize_table(block: DocumentBlock, context: str) -> SerializedTable:
    """Create deterministic span-aware text while retaining the original cells."""
    if not block.cells:
        raw = _clean(block.text)
        units_match = UNIT_PATTERN.search(f"{context} {raw}")
        return SerializedTable(
            rows=[raw] if raw else [],
            header_row_count=1,
            units=units_match.group(0) if units_match else None,
        )

    row_count = max(cell.row_span[1] for cell in block.cells)
    column_count = max(cell.col_span[1] for cell in block.cells)
    grid: list[list[str]] = [["" for _ in range(column_count)] for _ in range(row_count)]
    for cell in sorted(
        block.cells,
        key=lambda value: (value.row_span[0], value.col_span[0], value.row_span[1], value.col_span[1]),
    ):
        value = _clean(cell.text)
        if not value:
            continue
        for row in range(cell.row_span[0], cell.row_span[1]):
            for column in range(cell.col_span[0], cell.col_span[1]):
                grid[row][column] = value

    rows = [
        f"Row {row_index + 1}: " + " | ".join(value or "[empty]" for value in row)
        for row_index, row in enumerate(grid)
        if any(row)
    ]
    serialized = " ".join(rows)
    units_match = UNIT_PATTERN.search(f"{context} {serialized}")
    return SerializedTable(
        rows=rows,
        header_row_count=min(_infer_header_rows(block.cells, len(rows)), len(rows)),
        units=units_match.group(0) if units_match else None,
    )


def _table_parts(rows: list[str], header_row_count: int, max_chars: int) -> list[str]:
    """Split only at row boundaries and repeat all inferred header rows."""
    if not rows:
        return []
    if len("\n".join(rows)) <= max_chars:
        return ["\n".join(rows)]

    headers = rows[:header_row_count]
    data_rows = rows[header_row_count:]
    if not data_rows:
        return ["\n".join(rows)]
    parts: list[str] = []
    current = list(headers)
    for row in data_rows:
        candidate = "\n".join([*current, row])
        if len(candidate) > max_chars and len(current) > len(headers):
            parts.append("\n".join(current))
            current = [*headers, row]
        else:
            current.append(row)
    if len(current) > len(headers):
        parts.append("\n".join(current))
    return parts


def build_chunks(request: IndexDocumentRequest, config: ChunkingConfig) -> list[Chunk]:
    document = request.document
    filename = request.source_filename or document.document_id
    chunks: list[Chunk] = []
    active_section = "Document"

    for page in sorted(document.pages, key=lambda item: item.page_number):
        paragraph_group: list[DocumentBlock] = []
        page_paragraphs: list[tuple[str, list[DocumentBlock]]] = []
        recent_text: list[str] = []

        def flush_paragraphs() -> None:
            nonlocal paragraph_group
            if paragraph_group:
                page_paragraphs.append((active_section, paragraph_group))
                paragraph_group = []

        for block in sorted(page.blocks, key=lambda item: item.order):
            clean_text = _clean(block.text)
            if block.content_type == "table":
                context = "\n".join(recent_text[-2:])
                flush_paragraphs()
                table = _serialize_table(block, context)
                if not table.rows:
                    continue
                table_title = active_section if active_section != "Document" else None
                prefix = []
                if table_title:
                    prefix.append(f"Table: {table_title}")
                if context:
                    prefix.append(f"Context: {context}")
                if table.units:
                    prefix.append(f"Units: {table.units}")
                parent_text = "\n".join([*prefix, *table.rows])
                available_chars = max(100, config.max_chars - len("\n".join(prefix)))
                table_parts = _table_parts(
                    table.rows, table.header_row_count, available_chars
                )
                for index, part in enumerate(table_parts):
                    text = "\n".join([*prefix, part])
                    identity_ids = [block.uuid]
                    if len(table_parts) > 1:
                        identity_ids = [f"{block.uuid}:part:{index + 1}"]
                    chunks.append(
                        Chunk(
                            chunk_id=_chunk_id(
                                document.document_id,
                                page.page_number,
                                active_section,
                                "table",
                                identity_ids,
                                text,
                            ),
                            document_id=document.document_id,
                            source_doc_uid=request.source_doc_uid,
                            source_filename=filename,
                            page=page.page_number,
                            section=active_section,
                            content_type="table",
                            text=text,
                            parent_text=parent_text,
                            bbox=tuple(block.bbox),
                            source_block_ids=[block.uuid],
                            table_title=table_title,
                            table_context=context or None,
                            table_cells=block.cells,
                            metadata=request.metadata,
                        )
                    )
                continue

            if not clean_text:
                continue
            recent_text.append(clean_text)
            if _looks_like_heading(block, clean_text):
                flush_paragraphs()
                active_section = clean_text.rstrip(":")
                paragraph_group = [block]
                continue

            prospective = "\n".join(item.text for item in [*paragraph_group, block])
            if paragraph_group and len(prospective) > config.max_chars:
                flush_paragraphs()
                if config.overlap_blocks:
                    previous_blocks = page_paragraphs[-1][1]
                    paragraph_group = previous_blocks[-config.overlap_blocks :]
            paragraph_group.append(block)
        flush_paragraphs()

        for section, blocks in page_paragraphs:
            text = "\n".join(_clean(block.text) for block in blocks)
            block_ids = [block.uuid for block in blocks]
            chunks.append(
                Chunk(
                    chunk_id=_chunk_id(
                        document.document_id,
                        page.page_number,
                        section,
                        "paragraph",
                        block_ids,
                        text,
                    ),
                    document_id=document.document_id,
                    source_doc_uid=request.source_doc_uid,
                    source_filename=filename,
                    page=page.page_number,
                    section=section,
                    content_type="paragraph",
                    text=text,
                    parent_text=text,
                    bbox=_bbox_union(blocks),
                    source_block_ids=block_ids,
                    metadata=request.metadata,
                )
            )
    return chunks
