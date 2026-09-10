"""Bounded search-time policies over the existing corpus; no index writes."""

from __future__ import annotations

import json
import re
from collections import Counter

from pydantic import BaseModel, ConfigDict, Field

from app.answer_normalization import normalize_value
from app.config import settings
from app.decomposition import hit_key


class RetrievalProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    top_k: int = Field(default=5, ge=1, le=50)
    candidate_k: int = Field(default=30, ge=1, le=250)
    dense_weight: float = Field(default=0.55, ge=0, le=1, allow_inf_nan=False)
    rrf_k: int = Field(default=60, ge=1, le=200)
    min_score: float = Field(default=0, ge=0, le=1, allow_inf_nan=False)
    min_grade_confidence: float = Field(default=0, ge=0, le=1, allow_inf_nan=False)


def profiles() -> dict[str, RetrievalProfile]:
    raw = json.loads(settings.RETRIEVAL_PROFILES_JSON)
    if not isinstance(raw, dict) or set(raw) - {"text", "table", "numerical"}:
        raise ValueError("Retrieval profiles must use text, table, or numerical keys")
    return {key: RetrievalProfile.model_validate(value) for key, value in raw.items()}


def policy(question_type: str, retry: int = 0) -> RetrievalProfile:
    profile = RetrievalProfile(
        top_k=settings.TOP_K_FINAL,
        candidate_k=settings.TOP_K_OVERRETRIEVE,
        min_score=settings.GRADE_MIN_SCORE,
        min_grade_confidence=settings.GRADE_MIN_CONFIDENCE,
    )
    if settings.QUESTION_TYPE_RETRIEVAL:
        profile = profiles().get(question_type, profile)
    if settings.ADAPTIVE_TOP_K and retry:
        profile.top_k = min(
            50,
            max(
                profile.top_k,
                min(settings.ADAPTIVE_TOP_K_MAX, profile.top_k * (2**retry)),
            ),
        )
        profile.candidate_k = min(250, max(profile.candidate_k, profile.top_k * 3))
    profile.candidate_k = max(profile.top_k, profile.candidate_k)
    return profile


# Expansions retain the original question, including companies, years and units.
# These are search alternatives, never financial equivalences used in calculation.
SYNONYMS = (
    (r"\brevenue\b|\bnet sales\b", "revenue net sales"),
    (r"\boperating income\b|\boperating profit\b", "operating income operating profit"),
    (r"\bR&D\b|\bresearch and development\b", "research and development R&D"),
    (r"\bEPS\b|\bearnings per share\b", "earnings per share EPS"),
    (r"\bSG&A\b|\bselling general", "selling general and administrative SG&A"),
)


def query_variants(question: str, query: str, question_type: str) -> list[str]:
    variants = []
    if settings.QUERY_EXPANSION:
        terms = [
            value
            for pattern, value in SYNONYMS
            if re.search(pattern, question, re.IGNORECASE)
        ]
        if terms:
            variants.append(f"{question} {' '.join(terms)}")
    if settings.TABLE_QUERY_REWRITING and question_type in {"table", "numerical"}:
        # Removing interrogative boilerplate leaves row labels and all operands.
        compact = re.sub(
            r"\b(what|which|how|was|were|is|are|did|does|please|calculate)\b",
            " ",
            question,
            flags=re.IGNORECASE,
        )
        variants.append(" ".join(compact.split()) + " financial table row year period")
    return list(dict.fromkeys(v for v in variants if v.strip() and v != query))[:2]


def diversify(hits: list[dict], limit: int) -> list[dict]:
    """Greedy lexical MMR; preserve numeric differences and per-entity anchors."""
    unique = {}
    for hit in hits:
        unique.setdefault(hit_key(hit), hit)
    remaining = list(unique.values())
    picked: list[dict] = []
    covered: set[str] = set()

    token_cache = {
        hit_key(hit): set(
            re.findall(r"\w+", (hit.get("text") or hit.get("content") or "").casefold())
        )
        for hit in remaining
    }

    def tokens(hit):
        return token_cache[hit_key(hit)]

    while remaining and len(picked) < limit:

        def utility(hit):
            words = tokens(hit)
            redundancy = max(
                (
                    len(words & tokens(old)) / max(1, len(words | tokens(old)))
                    for old in picked
                ),
                default=0,
            )
            # Never erase different values/periods as duplicate prose.
            score = float(hit.get("score") or 0)
            entity_bonus = (
                1
                if hit.get("routed_entity") and hit["routed_entity"] not in covered
                else 0
            )
            return entity_bonus + 0.75 * score - 0.25 * redundancy

        chosen = max(remaining, key=utility)
        remaining.remove(chosen)
        picked.append(chosen)
        if chosen.get("routed_entity"):
            covered.add(chosen["routed_entity"])
    return picked


def material_tokens(values: list[str]) -> Counter:
    """Used by repair to reject altered facts, numbers, units or qualifiers."""
    return Counter(material_sequence(values))


def material_sequence(values: list[str]) -> list[str]:
    return re.findall(
        r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?|[\w]+|[%$€£+-]",
        " ".join(normalize_value(v) for v in values).casefold(),
    )
