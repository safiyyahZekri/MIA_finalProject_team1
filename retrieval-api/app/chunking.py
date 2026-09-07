from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass

from .models import BBox, Chunk, DocumentBlock, IndexDocumentRequest


@dataclass(frozen=True)
class ChunkingConfig:
    max_chars: int = 1800
    overlap_blocks: int = 1


def _looks_like_heading(text: str) -> bool:
    text = " ".join(text.split())
    if not text or len(text) > 120 or len(text.split()) > 14:
        return False
    if text.endswith((".", ";", ",")):
        return False
    letters = [character for character in text if character.isalpha()]
    if not letters:
        return False
    uppercase_ratio = sum(character.isupper() for character in letters) / len(letters)
    title_ratio = sum(word[:1].isupper() for word in text.split()) / len(text.split())
    numbered = bool(re.match(r"^(?:\d+(?:\.\d+)*|[IVX]+)[.)]?\s+", text))
    return uppercase_ratio >= 0.72 or title_ratio >= 0.75 or numbered


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


def _table_rows(block: DocumentBlock) -> list[str]:
    if not block.cells:
        return [" ".join(block.text.split())]

    rows: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for cell in block.cells:
        row = cell.row_span[0]
        column = cell.col_span[0]
        value = " ".join(cell.text.split())
        if value:
            rows[row].append((column, value))
    return [
        " | ".join(value for _, value in sorted(rows[row]))
        for row in sorted(rows)
        if rows[row]
    ]


def _table_parts(rows: list[str], max_chars: int) -> list[str]:
    """Split only at row boundaries and repeat the probable header row."""
    if not rows:
        return []
    whole = "\n".join(rows)
    if len(whole) <= max_chars:
        return [whole]

    header = rows[0]
    parts: list[str] = []
    current = [header]
    for row in rows[1:]:
        candidate = "\n".join([*current, row])
        if len(candidate) > max_chars and len(current) > 1:
            parts.append("\n".join(current))
            current = [header, row]
        else:
            current.append(row)
    if current:
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

        def flush_paragraphs() -> None:
            nonlocal paragraph_group
            if paragraph_group:
                page_paragraphs.append((active_section, paragraph_group))
                paragraph_group = []

        for block in sorted(page.blocks, key=lambda item: item.order):
            clean_text = " ".join(block.text.split())
            if block.content_type == "table":
                flush_paragraphs()
                rows = _table_rows(block)
                if not any(rows):
                    continue
                parent_text = "\n".join(rows)
                table_parts = _table_parts(rows, config.max_chars)
                for index, part in enumerate(table_parts):
                    text = (
                        f"{active_section}\n{part}"
                        if active_section != "Document"
                        else part
                    )
                    block_ids = [block.uuid]
                    if len(table_parts) > 1:
                        block_ids = [f"{block.uuid}:part:{index + 1}"]
                    chunks.append(
                        Chunk(
                            chunk_id=_chunk_id(
                                document.document_id,
                                page.page_number,
                                active_section,
                                "table",
                                block_ids,
                                text,
                            ),
                            document_id=document.document_id,
                            source_filename=filename,
                            page=page.page_number,
                            section=active_section,
                            content_type="table",
                            text=text,
                            parent_text=parent_text,
                            bbox=tuple(block.bbox),
                            source_block_ids=[block.uuid],
                            metadata=request.metadata,
                        )
                    )
                continue

            if not clean_text:
                continue

            if _looks_like_heading(clean_text):
                flush_paragraphs()
                active_section = clean_text
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
            text = "\n".join(" ".join(block.text.split()) for block in blocks)
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
