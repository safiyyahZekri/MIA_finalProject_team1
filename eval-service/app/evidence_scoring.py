"""Separate answer citations from the ranked evidence the agent actually saw."""

from __future__ import annotations

import re
from typing import Any

from .retrieval_benchmark import _hit_identity

METRIC_DEFINITIONS = {
    "citation": "Final answer citations; an abstention has no citations and scores zero.",
    "agent_retrieval": "Ranked hits in agent retrieve steps, independently of its answer.",
    "first_attempt": "Initial retrieval before query reformulation.",
    "final_attempt": "Last retrieval passed to evidence grading; not the best attempt.",
    "any_attempt": "Union coverage across retries; NOT Recall@K of a single ranked search.",
    "precision_at_k": "Relevant returned hits divided by K, including unfilled slots as misses.",
    "relevance": "Document labels only; a document match does not prove the right facts were retrieved.",
    "legacy_retrieval": "Deprecated citation-based fields retained unchanged for compatibility.",
}
_LEGACY_HIT = re.compile(r"^(.+):p(\d+)$")


def ranked_metrics(ids: list[str | None], relevant: set[str], k: int) -> dict:
    if k < 1:
        raise ValueError("retrieval k must be positive")
    top = ids[:k]
    hits = relevant.intersection(top)
    return {
        "recall_at_k": len(hits) / len(relevant) if relevant else None,
        "precision_at_k": sum(i in relevant for i in top) / k,
        "hit_rate": float(bool(hits)),
        "reciprocal_rank": next(
            (1 / rank for rank, i in enumerate(top, 1) if i in relevant), 0.0
        ),
    }


def score_evidence(
    answer: dict | None, trace: Any, relevant: set[str], k: int, aliases: dict[str, str]
) -> dict:
    """Read both legacy ``id:p1`` traces and metadata-rich retrieval traces.

    Missing/malformed traces are unknown, never inferred from citations.
    Local identity links come only from returned hit metadata, never gold.
    """
    identity_map = dict(aliases)
    attempts: list[list[dict] | None] = []
    issues: list[str] = []
    for step in trace if isinstance(trace, list) else []:
        if not isinstance(step, dict) or step.get("step") != "retrieve":
            continue
        raw_hits = step.get("retrieval_hits", step.get("hits"))
        hits = []
        if not isinstance(raw_hits, list):
            attempts.append(None)
            issues.append("retrieve step has no ranked hit list")
            continue
        malformed = False
        for raw in raw_hits:
            if isinstance(raw, dict) and _hit_identity(raw, "document", identity_map):
                hits.append(raw)
            elif isinstance(raw, str) and (match := _LEGACY_HIT.fullmatch(raw)):
                hits.append({"document_id": match[1], "page": int(match[2])})
            else:
                malformed = True
        attempts.append(None if malformed else hits)
        if malformed:
            issues.append(
                "retrieve step contains an unrecognised hit; its ranking is unavailable"
            )
        for hit in hits:
            identity = _hit_identity(hit, "document", identity_map)
            document_id = hit.get("document_id")
            if document_id and identity and identity != document_id:
                identity_map[document_id] = identity

    unresolved: set[str] = set()

    def resolve(hits: list[dict]) -> list[str | None]:
        ids = [_hit_identity(hit, "document", identity_map) for hit in hits]
        unresolved.update(i for i in ids if i and i.startswith("sha256-"))
        return ids

    ranked = [resolve(hits) if hits is not None else None for hits in attempts]
    raw_citations = (answer or {}).get("evidence")
    citations = resolve(
        [e for e in raw_citations if isinstance(e, dict)]
        if isinstance(raw_citations, list)
        else []
    )
    if not relevant:
        return {"citation": None, "agent_retrieval": None}

    def attempt_result(ids: list[str | None] | None) -> dict | None:
        if ids is None:
            return None
        return {"document_ids": ids[:k], **ranked_metrics(ids, relevant, k)}

    union = {
        identity
        for ids in ranked
        if ids is not None
        for identity in ids[:k]
        if identity
    }
    complete_trace = bool(ranked) and all(ids is not None for ids in ranked)
    retrieval = {
        "k": k,
        "attempt_count": len(ranked),
        "first_attempt": attempt_result(ranked[0]) if ranked else None,
        "final_attempt": attempt_result(ranked[-1]) if ranked else None,
        "any_attempt_hit_rate": float(bool(union & relevant))
        if complete_trace
        else None,
        "any_attempt_recall": len(union & relevant) / len(relevant)
        if complete_trace
        else None,
        "unresolved_document_ids": sorted(unresolved),
        "trace_issues": issues,
    }
    return {
        "citation": {
            "k": k,
            "document_ids": citations[:k],
            **ranked_metrics(citations, relevant, k),
        },
        "agent_retrieval": retrieval,
    }


def summarize_evidence(rows: list[dict]) -> dict:
    """Use explicit denominators; unavailable traces do not become zero scores."""

    def aggregate(values: list[dict]) -> dict:
        metrics = ("recall_at_k", "precision_at_k", "hit_rate", "reciprocal_rank")
        return {
            "scored_questions": len(values),
            **{
                name: sum(v[name] for v in values) / len(values) if values else None
                for name in metrics
            },
        }

    retrieval = [
        r["agent_retrieval"] for r in rows if r.get("agent_retrieval") is not None
    ]
    complete = [r for r in retrieval if r["any_attempt_hit_rate"] is not None]
    return {
        "citation": aggregate(
            [r["citation"] for r in rows if r.get("citation") is not None]
        ),
        "agent_retrieval": {
            "first_attempt": aggregate(
                [
                    r["first_attempt"]
                    for r in retrieval
                    if r["first_attempt"] is not None
                ]
            ),
            "final_attempt": aggregate(
                [
                    r["final_attempt"]
                    for r in retrieval
                    if r["final_attempt"] is not None
                ]
            ),
            "any_attempt_scored_questions": len(complete),
            "any_attempt_hit_rate": sum(r["any_attempt_hit_rate"] for r in complete)
            / len(complete)
            if complete
            else None,
            "any_attempt_recall": sum(r["any_attempt_recall"] for r in complete)
            / len(complete)
            if complete
            else None,
            "unresolved_document_ids": sorted(
                {i for r in retrieval for i in r["unresolved_document_ids"]}
            ),
            "questions_with_trace_issues": sum(
                bool(r["trace_issues"]) for r in retrieval
            ),
        },
    }
