"""How an index's chunks are built, and which text is embedded for each chunk.

The format belongs to the index: it is stored in manifest.json, and a
retrieval-api serving an index builds and embeds new documents the way the
index already was, so vectors, BM25 and later uploads stay consistent. An index
whose manifest records no format is the original one, LEGACY, and with every
switch off chunks and embedded text are exactly what they always were.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, fields
from typing import Any

from .models import DEFAULT_SECTION, Chunk

# A document id, a TAT-DQA uid, or a filename or path that is only one of them.
_OPAQUE_ID = re.compile(r"sha256-[0-9a-f]{64}|[0-9a-f]{32}", re.IGNORECASE)


@dataclass(frozen=True)
class IndexFormat:
    # Embed and BM25-index "company section metadata text" without opaque ids.
    # The legacy text starts with the hash filename, uid and source path, a
    # median 83 of 255 tokens per chunk in the TAT-DQA index, before any content.
    contextual_text: bool = False
    # metadata["company"]: the issuer the document names in its own page
    # headers (app/company.py). A company supplied at upload is kept.
    company_metadata: bool = False
    # A table whose first row holds only a heading (A037 "Operating costs")
    # is titled by it instead of by the running section.
    table_titles: bool = False

    @classmethod
    def names(cls) -> tuple[str, ...]:
        return tuple(field.name for field in fields(cls))

    @classmethod
    def parse(cls, spec: str) -> IndexFormat:
        """LEDGER_INDEX_FORMAT: "legacy", or switch names separated by commas."""
        requested = {part.strip() for part in spec.split(",") if part.strip()}
        if requested <= {"legacy"}:
            return LEGACY
        unknown = requested - set(cls.names())
        if unknown:
            raise ValueError(
                f"unknown index format {sorted(unknown)}: use 'legacy' or any of "
                f"{', '.join(cls.names())}"
            )
        return cls(**{name: True for name in requested})

    @classmethod
    def from_manifest(cls, value: Any) -> IndexFormat:
        if value is None:
            return LEGACY
        if not isinstance(value, dict):
            raise TypeError("manifest index_format must be an object")
        unknown = set(value) - set(cls.names())
        if unknown:
            raise ValueError(
                "the index was built with format switches this retrieval-api does not "
                f"know: {sorted(unknown)}"
            )
        if not all(isinstance(flag, bool) for flag in value.values()):
            raise ValueError("manifest index_format switches must be true or false")
        return cls(**value)

    def to_manifest(self) -> dict[str, bool]:
        return asdict(self)

    def enabled(self) -> list[str]:
        return [name for name, on in asdict(self).items() if on]

    def describe(self) -> str:
        return ",".join(self.enabled()) or "legacy"


LEGACY = IndexFormat()


def _is_opaque(value: str, chunk: Chunk) -> bool:
    name = re.split(r"[\\/]", value.strip())[-1]
    stem = re.sub(r"\.pdf$", "", name, flags=re.IGNORECASE)
    return bool(_OPAQUE_ID.fullmatch(stem)) or stem in {chunk.document_id, chunk.source_doc_uid}


def searchable_text(chunk: Chunk, index_format: IndexFormat) -> str:
    """The text a chunk is embedded and BM25-indexed with."""
    if not index_format.contextual_text:
        metadata = " ".join(
            str(value) for value in chunk.metadata.values() if value is not None
        )
        return f"{chunk.source_filename} {metadata} {chunk.section} {chunk.text}"

    context: list[str] = []
    seen: set[str] = set()

    def add(value: object) -> None:
        text = " ".join(str(value).split())
        if text and text.casefold() not in seen:
            seen.add(text.casefold())
            context.append(text)

    company = chunk.metadata.get("company")
    if company is not None:
        add(company)
    if chunk.section != DEFAULT_SECTION:
        add(chunk.section)
    for key, value in [("source_filename", chunk.source_filename), *chunk.metadata.items()]:
        if key == "company" or value is None:
            continue
        if isinstance(value, str) and _is_opaque(value, chunk):
            continue
        add(value)
    return " ".join([*context, chunk.text])
