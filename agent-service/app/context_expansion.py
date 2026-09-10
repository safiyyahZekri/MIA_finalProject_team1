"""Read bounded page context from already indexed documents."""

from app.config import settings
from app.decomposition import hit_key


async def expand_context(hits: list[dict], fetch) -> tuple[list[dict], dict]:
    result = list(hits)
    pages = set()
    seen = {hit_key(h) for h in hits}
    added = 0
    for anchor in hits[:2]:
        if float(anchor.get("score") or 0) < settings.MIN_EVIDENCE_SCORE:
            continue
        if not anchor.get("document_id") or not isinstance(anchor.get("page"), int):
            continue
        for page in (anchor["page"], anchor["page"] - 1, anchor["page"] + 1):
            key = (anchor["document_id"], page)
            if page < 1 or key in pages or added >= 4 or len(result) >= 50:
                continue
            pages.add(key)
            nearby = await fetch({"document_id": key[0], "page": page}, top_k=2)
            for hit in nearby:
                if hit.get("document_id") != key[0] or hit.get("page") != page:
                    continue
                if hit_key(hit) in seen or added >= 4 or len(result) >= 50:
                    continue
                # A metadata match's score=1 is not query confidence. Context
                # may support reasoning but cannot raise the calibrated gate.
                result.append(
                    {
                        **hit,
                        "score": 0.0,
                        "context_only": True,
                        "context_anchor_chunk_id": anchor.get("chunk_id"),
                    }
                )
                seen.add(hit_key(hit))
                added += 1
    return result, {"pages_requested": len(pages), "context_hits": added}
