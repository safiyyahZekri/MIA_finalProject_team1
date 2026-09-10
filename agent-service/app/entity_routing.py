"""Question-entity document routing over the existing retrieval index."""

from __future__ import annotations

import re
from collections.abc import Iterable

TOKEN_RE = re.compile(r"[a-z0-9]+")
CORPORATE_SUFFIXES = {
    "co",
    "company",
    "corp",
    "corporation",
    "inc",
    "incorporated",
    "limited",
    "llc",
    "ltd",
    "plc",
}


def entity_tokens(entity: str) -> set[str]:
    tokens = set(TOKEN_RE.findall(entity.casefold()))
    meaningful = tokens - CORPORATE_SUFFIXES
    return meaningful or tokens


def hit_mentions_entity(hit: dict, entity: str) -> bool:
    metadata = hit.get("metadata") if isinstance(hit.get("metadata"), dict) else {}
    searchable = " ".join(
        str(value)
        for value in (
            hit.get("text"),
            hit.get("content"),
            hit.get("filename"),
            hit.get("source_filename"),
            *metadata.values(),
        )
        if value is not None
    )
    return entity_tokens(entity) <= set(TOKEN_RE.findall(searchable.casefold()))


def choose_entity_document(
    entity: str,
    rankings: Iterable[list[dict]],
    excluded_document_ids: set[str] | None = None,
) -> str | None:
    """Fuse verified entity-only rankings and return their best document."""
    excluded = excluded_document_ids or set()
    fusion: dict[str, float] = {}
    confidence: dict[str, float] = {}
    for hits in rankings:
        seen: set[str] = set()
        for rank, hit in enumerate(hits, start=1):
            document_id = hit.get("document_id")
            if (
                not document_id
                or document_id in excluded
                or document_id in seen
                or not hit_mentions_entity(hit, entity)
            ):
                continue
            seen.add(document_id)
            fusion[document_id] = fusion.get(document_id, 0.0) + 1 / (60 + rank)
            confidence[document_id] = max(
                confidence.get(document_id, 0.0), float(hit.get("score") or 0.0)
            )
    if not fusion:
        return None
    return max(
        fusion,
        key=lambda document_id: (
            fusion[document_id],
            confidence[document_id],
            document_id,
        ),
    )


def scoped_entity_query(query: str, entity: str, all_entities: list[str]) -> str:
    """Remove other named companies while retaining the selected company."""
    scoped = query
    for other in all_entities:
        if other.casefold() == entity.casefold():
            continue
        scoped = re.sub(re.escape(other), " ", scoped, flags=re.IGNORECASE)
    scoped = " ".join(scoped.split())
    if not entity_tokens(entity) <= set(TOKEN_RE.findall(scoped.casefold())):
        scoped = f"{entity} {scoped}".strip()
    return scoped
