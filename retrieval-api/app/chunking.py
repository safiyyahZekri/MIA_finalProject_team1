from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from .company import detect_company
from .index_format import LEGACY, IndexFormat
from .models import (
    DEFAULT_SECTION,
    BBox,
    Cell,
    Chunk,
    DocumentBlock,
    IndexDocumentRequest,
    MetadataValue,
)
from .tables import UNIT_PATTERN, first_row_title, table_grid
from .tables import clean as _clean

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
    numbered = bool(
        re.match(
            r"^(?:item\s+)?(?:\d+(?:\.\d+)*|[IVX]+)[.)]?\s+",
            text,
            re.IGNORECASE,
        )
    )
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


def _table_units(context: str, rows: list[str]) -> str | None:
    match = UNIT_PATTERN.search(f"{context} {' '.join(rows)}")
    return match.group(0) if match else None


def _header_row_count(cells: list[Cell], rows: list[str]) -> int:
    if not cells:
        return 1
    return min(_infer_header_rows(cells, len(rows)), len(rows))


def _serialize_table(block: DocumentBlock, context: str) -> SerializedTable:
    """Create deterministic span-aware text while retaining the original cells."""
    if not block.cells:
        raw = _clean(block.text)
        rows = [raw] if raw else []
        return SerializedTable(
            rows=rows, header_row_count=1, units=_table_units(context, rows)
        )

    rows = [
        f"Row {row_index + 1}: " + " | ".join(value or "[empty]" for value in row)
        for row_index, row in enumerate(table_grid(block.cells))
        if any(row)
    ]
    return SerializedTable(
        rows=rows,
        header_row_count=_header_row_count(block.cells, rows),
        units=_table_units(context, rows),
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


def _table_chunks(
    *,
    document_id: str,
    source_doc_uid: str | None,
    filename: str,
    page: int,
    section: str,
    block_uuid: str,
    bbox: BBox,
    cells: list[Cell],
    table: SerializedTable,
    context: str,
    metadata: dict[str, MetadataValue],
    config: ChunkingConfig,
    index_format: IndexFormat,
) -> list[Chunk]:
    """The chunks of one serialized table; shared by indexing and rebuilds."""
    table_title = section if section != DEFAULT_SECTION else None
    if index_format.table_titles:
        table_title = first_row_title(cells) or table_title
    prefix = []
    if table_title:
        prefix.append(f"Table: {table_title}")
    if context:
        prefix.append(f"Context: {context}")
    if table.units:
        prefix.append(f"Units: {table.units}")
    parent_text = "\n".join([*prefix, *table.rows])
    available_chars = max(100, config.max_chars - len("\n".join(prefix)))
    table_parts = _table_parts(table.rows, table.header_row_count, available_chars)
    chunks: list[Chunk] = []
    for index, part in enumerate(table_parts):
        text = "\n".join([*prefix, part])
        identity_ids = [block_uuid]
        if len(table_parts) > 1:
            identity_ids = [f"{block_uuid}:part:{index + 1}"]
        chunks.append(
            Chunk(
                chunk_id=_chunk_id(document_id, page, section, "table", identity_ids, text),
                document_id=document_id,
                source_doc_uid=source_doc_uid,
                source_filename=filename,
                page=page,
                section=section,
                content_type="table",
                text=text,
                parent_text=parent_text,
                bbox=tuple(bbox),
                source_block_ids=[block_uuid],
                table_title=table_title,
                table_context=context or None,
                table_cells=cells,
                metadata=metadata,
            )
        )
    return chunks


def apply_document_format(chunks: list[Chunk], index_format: IndexFormat) -> list[Chunk]:
    """Document-level format steps, applied the same way at upload and rebuild."""
    if not chunks or not index_format.company_metadata:
        return chunks
    metadata = chunks[0].metadata
    supplied = metadata.get("company")
    if isinstance(supplied, str) and supplied.strip():
        return chunks
    company = detect_company(chunks)
    if company is None:
        return chunks
    enriched = {**metadata, "company": company}
    return [chunk.model_copy(update={"metadata": enriched}) for chunk in chunks]


def _flush_paragraphs(
    paragraph_group: list[DocumentBlock],
    section: str,
    destination: list[tuple[str, list[DocumentBlock]]],
) -> list[DocumentBlock]:
    if paragraph_group:
        destination.append((section, paragraph_group))
    return []


def build_chunks(
    request: IndexDocumentRequest,
    config: ChunkingConfig,
    index_format: IndexFormat = LEGACY,
) -> list[Chunk]:
    document = request.document
    filename = request.source_filename or document.document_id
    chunks: list[Chunk] = []
    active_section = DEFAULT_SECTION

    for page in sorted(document.pages, key=lambda item: item.page_number):
        paragraph_group: list[DocumentBlock] = []
        page_paragraphs: list[tuple[str, list[DocumentBlock]]] = []
        recent_text: list[str] = []

        for block in sorted(page.blocks, key=lambda item: item.order):
            clean_text = _clean(block.text)
            if block.content_type == "table":
                context = "\n".join(recent_text[-2:])
                paragraph_group = _flush_paragraphs(
                    paragraph_group, active_section, page_paragraphs
                )
                table = _serialize_table(block, context)
                if not table.rows:
                    continue
                chunks.extend(
                    _table_chunks(
                        document_id=document.document_id,
                        source_doc_uid=request.source_doc_uid,
                        filename=filename,
                        page=page.page_number,
                        section=active_section,
                        block_uuid=block.uuid,
                        bbox=tuple(block.bbox),
                        cells=block.cells,
                        table=table,
                        context=context,
                        metadata=request.metadata,
                        config=config,
                        index_format=index_format,
                    )
                )
                continue

            if not clean_text:
                continue
            recent_text.append(clean_text)
            if _looks_like_heading(block, clean_text):
                paragraph_group = _flush_paragraphs(
                    paragraph_group, active_section, page_paragraphs
                )
                active_section = clean_text.rstrip(":")
                paragraph_group = [block]
                continue

            prospective = "\n".join(item.text for item in [*paragraph_group, block])
            if paragraph_group and len(prospective) > config.max_chars:
                paragraph_group = _flush_paragraphs(
                    paragraph_group, active_section, page_paragraphs
                )
                if config.overlap_blocks:
                    previous_blocks = page_paragraphs[-1][1]
                    paragraph_group = previous_blocks[-config.overlap_blocks :]
            paragraph_group.append(block)
        _flush_paragraphs(paragraph_group, active_section, page_paragraphs)

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
    return apply_document_format(chunks, index_format)


def _rebuild_table(first: Chunk, config: ChunkingConfig, index_format: IndexFormat) -> list[Chunk]:
    """Re-cut one table from its first legacy chunk, as build_chunks() would."""
    legacy_title = first.section if first.section != DEFAULT_SECTION else None
    if first.table_title != legacy_title or len(first.source_block_ids) != 1:
        raise ValueError(f"table chunk {first.chunk_id} was not built by the legacy format")
    context = first.table_context or ""
    head = "\n".join(
        [
            *([f"Table: {legacy_title}"] if legacy_title else []),
            *([f"Context: {context}"] if context else []),
        ]
    )
    if head and not first.parent_text.startswith(head + "\n"):
        raise ValueError(f"table chunk {first.chunk_id}: parent_text does not start with its title and context")
    lines = (first.parent_text[len(head) + 1 :] if head else first.parent_text).split("\n")
    units = None
    if lines[0].startswith("Units: "):
        # A unit phrase can span the context's line break ("(in\nthousands)"),
        # so the Units line can run over more than one line.
        for count in range(1, len(lines)):
            candidate = "\n".join(lines[:count])[len("Units: ") :]
            if _table_units(context, lines[count:]) == candidate:
                units, lines = candidate, lines[count:]
                break
    if not lines or _table_units(context, lines) != units:
        raise ValueError(f"table chunk {first.chunk_id}: stored rows and units do not agree")
    table = SerializedTable(
        rows=lines,
        header_row_count=_header_row_count(first.table_cells, lines),
        units=units,
    )
    return _table_chunks(
        document_id=first.document_id,
        source_doc_uid=first.source_doc_uid,
        filename=first.source_filename,
        page=first.page,
        section=first.section,
        block_uuid=first.source_block_ids[0],
        bbox=first.bbox,
        cells=first.table_cells,
        table=table,
        context=context,
        metadata=first.metadata,
        config=config,
        index_format=index_format,
    )


def rebuild_chunks(
    chunks: list[Chunk], config: ChunkingConfig, index_format: IndexFormat
) -> list[Chunk]:
    """One document's chunks under another format, from its legacy chunks.

    Every table chunk stores its whole serialized table, cells and context, and
    paragraph chunks do not depend on the format, so this returns what
    build_chunks() returns for the original document, with no OCR. Raises
    ValueError when a stored chunk cannot be read back exactly.
    """
    if not chunks:
        return []
    if len({chunk.document_id for chunk in chunks}) != 1:
        raise ValueError("rebuild_chunks takes the chunks of a single document")
    rebuilt: list[Chunk] = []
    tables_done: set[tuple[int, tuple[str, ...], str]] = set()
    for chunk in chunks:
        if chunk.content_type == "paragraph":
            rebuilt.append(chunk)
            continue
        key = (chunk.page, tuple(chunk.source_block_ids), chunk.parent_text)
        if key in tables_done:
            continue
        tables_done.add(key)
        rebuilt.extend(_rebuild_table(chunk, config, index_format))
    return apply_document_format(rebuilt, index_format)
