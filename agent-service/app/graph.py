"""
LangGraph "brain" for LEDGER's agent-service.

Graph shape (real conditional branches, not a fixed chain):

    START -> classify -> retrieve -> grade ----sufficient---> reason -> END
                              ^                 |
                              |            retry (< MAX_RETRIES)
                              |                 v
                              +---------- reformulate
                                          |
                                  insufficient (retries exhausted)
                                          v
                                 build_insufficient -> END

- `classify` branches retrieval strategy on text vs. table vs. numerical.
- When enabled, `decompose` runs once after classification; its subquery
  results merge into the same evidence-grading step.
- `grade` branches on sufficient vs. insufficient evidence, with a bounded
  retry loop that reformulates the query on weak evidence.
- `reason` branches on the extracted answer shape (direct / calculated /
  multi_span / insufficient) and is the ONLY place arithmetic happens --
  always via the deterministic calculate() tool, never the LLM directly.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict

from langgraph.graph import END, START, StateGraph

from app.answer_normalization import normalize_answer
from app.calculator import CalculatorError, calculate
from app.config import settings
from app.decomposition import QueryPlan, merge_query_results
from app.entity_routing import choose_entity_document, scoped_entity_query
from app.llm import (
    ExtractionCalculated,
    ExtractionDirect,
    ExtractionInsufficient,
    ExtractionMultiSpan,
    get_llm,
)
from app.schemas import validate_answer_dict
from app.search_policy import (
    diversify,
    policy,
    profiles,
    query_variants,
)
from app.tools import (
    filter_documents,
    search_bm25,
    search_documents,
    search_hybrid,
    search_tables,
)


class AgentState(TypedDict, total=False):
    question: str
    document_id: Optional[str]
    question_type: str
    current_query: str
    retry_count: int
    evidence: List[dict]
    sufficient: bool
    grade_reason: str
    answer: Dict[str, Any]
    trace: List[dict]
    subqueries: List[str]
    subquery_evidence: List[List[dict]]
    retry_evidence: List[List[dict]]
    entities: List[str]
    entity_document_ids: Dict[str, str]
    entity_routing_resolved: bool


def _log(state: AgentState, step: str, **detail) -> None:
    state.setdefault("trace", []).append({"step": step, **detail})


def _log_with_usage(state: AgentState, step: str, llm, **detail) -> None:
    """Same as _log, but also attaches per-call token usage when the active
    LLM provider exposes it (OllamaLLM does, via `last_usage`) -- gives
    eval-service/Langfuse something to show for prompt/token observability
    even when running against a local model instead of a metered API."""
    usage = getattr(llm, "last_usage", None)
    if usage:
        detail["usage"] = usage
    _log(state, step, **detail)


def _dedup_and_rank(*hit_lists: List[dict]) -> List[dict]:
    """Merge ranked hit lists into the top TOP_K_FINAL unique passages.

    By default hits are ordered by raw score. Dense, lexical and table
    searches score on different scales, so a passage ranked high by one can
    lose its place to lower-ranked hits of another: in A013 the gold
    passage was 4th in BM25 and dropped from the merged five. With
    RANK_FUSION_MERGE, hits are ordered by reciprocal rank fusion over the
    lists instead, with raw score breaking ties.
    """
    best: Dict[tuple, dict] = {}
    fusion: Dict[tuple, float] = {}
    for hits in hit_lists:
        seen = set()
        for rank, h in enumerate(hits, start=1):
            key = (h.get("document_id"), h.get("page"), h.get("section"))
            if key not in best or h.get("score", 0) > best[key].get("score", 0):
                best[key] = h
            if key not in seen:
                seen.add(key)
                fusion[key] = fusion.get(key, 0.0) + 1.0 / (60 + rank)
    if settings.RANK_FUSION_MERGE:
        order = sorted(best, key=lambda k: (-fusion[k], -best[k].get("score", 0)))
        ranked = [best[k] for k in order]
    else:
        ranked = sorted(best.values(), key=lambda h: h.get("score", 0), reverse=True)
    return ranked[: settings.TOP_K_FINAL]


def _top_evidence(hits: List[dict], n: int) -> List[dict]:
    # The Strict Answer Schema allows exactly these fields per citation, and
    # answer-validator-api rejects any other. Coordinates for highlighting
    # belong in a separate field next to the answer, not in the citation.
    return [
        {"document_id": h["document_id"], "page": h["page"], "section": h.get("section")}
        for h in hits[:n]
    ]


def _citations(
    evidence: List[dict], indexes: List[int], fallback_n: int
) -> Tuple[List[dict], bool]:
    """Cite the passages the model says the answer was taken from.

    Citing the top-ranked hits instead let a citation point at a document that
    does not contain the answer: in the Opus 5 experiments every correct answer
    with wrong citations had its figures in a lower-ranked chunk
    (eval-service/EXPERIMENTS.md, finding 5). Indexes are 1-based, as numbered
    in the extraction prompt; out-of-range and repeated ones are dropped.

    Returns (citations, cited_by_model). When the model names no usable
    passage the old top-ranked citation is used, and cited_by_model is False
    so the fallback is counted rather than hidden.
    """
    picked: List[dict] = []
    seen = set()
    for index in indexes or []:
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 1 <= index <= len(evidence)
        ):
            continue
        hit = evidence[index - 1]
        key = (hit["document_id"], hit["page"], hit.get("section"))
        if key not in seen:
            seen.add(key)
            picked.append(hit)
    if picked:
        return _top_evidence(picked, len(picked)), True
    return _top_evidence(evidence, fallback_n), False


_NUMBER_ONLY = re.compile(r"^\(?[-\u2212]?\$?\d[\d,]*(?:\.\d+)?%?\)?$")


def _is_value_list(formula: str) -> bool:
    """Two or more plain numbers separated by ", " or ";" and no arithmetic,
    such as '38.6, 23.6, 12.2'. The calculator reads commas as thousands
    separators, so such a list can never be evaluated; '12,200' stays one
    number."""
    parts = [part.strip() for part in re.split(r";|,\s+", formula.strip())]
    return len(parts) >= 2 and all(_NUMBER_ONLY.match(part) for part in parts)


def build_graph():
    llm = get_llm()
    profiles()
    policy("text")

    # ------------------------------------------------------------- nodes --
    async def classify_node(state: AgentState) -> AgentState:
        result = llm.classify(state["question"])
        state["question_type"] = result.question_type
        state["current_query"] = result.search_query
        state["entities"] = result.entities if settings.ENTITY_DOCUMENT_ROUTING else []
        state["entity_document_ids"] = {}
        state["entity_routing_resolved"] = False
        state["retry_count"] = 0
        _log_with_usage(
            state,
            "classify",
            llm,
            question_type=result.question_type,
            query=result.search_query,
            entities=state["entities"],
        )
        return state

    async def decompose_node(state: AgentState) -> AgentState:
        state["subqueries"] = []
        state["subquery_evidence"] = []
        started = time.perf_counter()
        if hasattr(llm, "last_usage"):
            llm.last_usage = {}
        try:
            plan = QueryPlan.model_validate(llm.decompose(state["question"]))
            if 2 <= len(plan.subqueries) <= settings.TOP_K_FINAL:
                state["subqueries"] = plan.subqueries
            status = "split" if state["subqueries"] else "single_query"
            _log_with_usage(
                state,
                "decompose",
                llm,
                status=status,
                subqueries=state["subqueries"],
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        except Exception as exc:
            # Planning is optional. Fall back to the ordinary query visibly;
            # failures in actual retrieval, grading and extraction still raise.
            logging.getLogger("agent-service").warning(
                "Query decomposition failed (%s); using the original query",
                type(exc).__name__,
            )
            _log_with_usage(
                state,
                "decompose",
                llm,
                status="fallback",
                error_type=type(exc).__name__,
                subqueries=[],
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        return state

    async def retrieve_query(
        query: str, qtype: str, doc_id: Optional[str], retry: int = 0
    ) -> tuple:
        profile = policy(qtype, retry)
        tuned = (
            settings.ADAPTIVE_TOP_K
            or settings.QUESTION_TYPE_RETRIEVAL
            or settings.EVIDENCE_DIVERSIFICATION
        )
        search_options = (
            {
                "top_k": min(50, profile.candidate_k)
                if settings.EVIDENCE_DIVERSIFICATION
                else profile.top_k,
                "candidate_k": profile.candidate_k,
            }
            if tuned
            else {}
        )
        if settings.HYBRID_RERANKING:
            fusion_options = (
                {"dense_weight": profile.dense_weight, "rrf_k": profile.rrf_k}
                if settings.QUESTION_TYPE_RETRIEVAL
                else {}
            )
            hybrid = await search_hybrid(
                query,
                top_k=search_options.get("top_k", profile.top_k),
                candidate_k=profile.candidate_k,
                document_id=doc_id,
                rerank=True,
                **fusion_options,
            )
            # Tables are a useful independent view for numeric/table questions;
            # text questions trust retrieval-api's already fused final ranking.
            if qtype in {"numerical", "table"}:
                return hybrid, await search_tables(
                    query, document_id=doc_id, **search_options
                )
            return (hybrid,)
        if qtype == "numerical":
            return await search_documents(
                query, document_id=doc_id, **search_options
            ), await search_tables(query, document_id=doc_id, **search_options)
        if qtype == "table":
            return await search_tables(
                query, document_id=doc_id, **search_options
            ), await search_documents(query, document_id=doc_id, **search_options)
        return await search_documents(
            query, document_id=doc_id, **search_options
        ), await search_bm25(query, document_id=doc_id, **search_options)

    async def retrieve_node(state: AgentState) -> AgentState:
        query = state["current_query"]
        doc_id = state.get("document_id")
        qtype = state["question_type"]

        retry = state.get("retry_count", 0)
        profile = policy(qtype, retry)
        rankings = list(await retrieve_query(query, qtype, doc_id, retry))
        variants = query_variants(state["question"], query, qtype)
        for variant in variants:
            rankings.extend(await retrieve_query(variant, qtype, doc_id, retry))
        enhanced = (
            settings.ADAPTIVE_TOP_K
            or settings.QUESTION_TYPE_RETRIEVAL
            or settings.EVIDENCE_DIVERSIFICATION
            or bool(variants)
        )
        pool_limit = (
            profile.candidate_k if settings.EVIDENCE_DIVERSIFICATION else profile.top_k
        )
        combined = (
            merge_query_results(rankings, pool_limit)
            if enhanced
            else _dedup_and_rank(*rankings)
        )

        routed_rankings: list[list[dict]] = []
        entities = state.get("entities", [])
        if settings.ENTITY_DOCUMENT_ROUTING and entities and not doc_id:
            entity_documents = state.get("entity_document_ids", {})
            if not state.get("entity_routing_resolved"):

                async def entity_candidates(entity: str):
                    started = time.perf_counter()
                    dense, lexical = await asyncio.gather(
                        search_documents(
                            entity, top_k=settings.ENTITY_ROUTING_CANDIDATES
                        ),
                        search_bm25(entity, top_k=settings.ENTITY_ROUTING_CANDIDATES),
                    )
                    return (
                        entity,
                        dense,
                        lexical,
                        (time.perf_counter() - started) * 1000,
                    )

                candidate_sets = await asyncio.gather(
                    *(entity_candidates(entity) for entity in entities)
                )
                used_documents: set[str] = set()
                for entity, dense, lexical, latency_ms in candidate_sets:
                    selected = choose_entity_document(
                        entity, [dense, lexical], used_documents
                    )
                    if selected:
                        entity_documents[entity] = selected
                        used_documents.add(selected)
                    _log(
                        state,
                        "route_entity",
                        entity=entity,
                        document_id=selected,
                        dense_candidates=len(dense),
                        lexical_candidates=len(lexical),
                        latency_ms=latency_ms,
                    )
                state["entity_document_ids"] = entity_documents
                state["entity_routing_resolved"] = True

            async def retrieve_routed(entity: str, routed_document_id: str):
                routed_query = scoped_entity_query(query, entity, entities)
                lists = await retrieve_query(
                    routed_query, qtype, routed_document_id, retry
                )
                hits = sorted(
                    (hit for group in lists for hit in group),
                    key=lambda hit: hit.get("score", 0),
                    reverse=True,
                )
                return [{**hit, "routed_entity": entity} for hit in hits]

            routed_rankings = list(
                await asyncio.gather(
                    *(
                        retrieve_routed(entity, routed_document_id)
                        for entity, routed_document_id in entity_documents.items()
                    )
                )
            )
            if routed_rankings:
                combined = merge_query_results([*routed_rankings, combined], pool_limit)

        # Direct metadata lookup (non-vector path #2): when the caller scoped
        # the question to a specific document, pull everything indexed for
        # it and fold that in too -- cheap, exact, and doesn't depend on the
        # embedding/BM25 query matching the right words.
        if doc_id:
            filter_hits = await filter_documents({"document_id": doc_id})
            if enhanced or settings.GRADE_CALIBRATION:
                filter_hits = [
                    {**h, "score": 0.0, "context_only": True} for h in filter_hits
                ]
                combined = merge_query_results([combined + filter_hits], pool_limit)
            else:
                combined = _dedup_and_rank(combined, filter_hits)

        subqueries = state.get("subqueries", [])
        reused = bool(state.get("subquery_evidence"))
        if subqueries:
            if not reused:
                rankings = []
                for subquery in subqueries:
                    started = time.perf_counter()
                    lists = await retrieve_query(subquery, qtype, doc_id, retry)
                    hits = sorted(
                        (h for group in lists for h in group),
                        key=lambda h: h.get("score", 0),
                        reverse=True,
                    )
                    rankings.append(hits)
                    _log(
                        state,
                        "retrieve_subquery",
                        query=subquery,
                        latency_ms=(time.perf_counter() - started) * 1000,
                        candidate_count=len(hits),
                        document_ids=list(
                            dict.fromkeys(h.get("document_id") for h in hits)
                        ),
                    )
                state["subquery_evidence"] = rankings
            # Within this request only: retain operand evidence through retries
            # while the existing reformulator searches for additional context.
            combined = merge_query_results(
                [*state["subquery_evidence"], combined], pool_limit
            )

        raw_attempt_hits = combined
        retry_fused = False
        if settings.RETRY_EVIDENCE_FUSION:
            # Retrieval retries previously replaced the entire evidence set.
            # Keep each attempt's ranking within this request and fuse only
            # after a retry. This preserves useful earlier passages without
            # making another service/model call or changing the shared index.
            history = [*state.get("retry_evidence", []), raw_attempt_hits]
            state["retry_evidence"] = history
            if len(history) > 1:
                combined = merge_query_results(
                    history,
                    max(pool_limit, settings.RETRY_FUSION_MAX_HITS),
                )
                retry_fused = True

        if settings.EVIDENCE_DIVERSIFICATION:
            combined = diversify(
                combined,
                max(
                    profile.top_k, settings.RETRY_FUSION_MAX_HITS if retry_fused else 0
                ),
            )
        if settings.ADJACENT_EVIDENCE and retry:
            from app.context_expansion import expand_context

            combined, detail = await expand_context(combined, filter_documents)
            _log(state, "adjacent_evidence", **detail)
        state["evidence"] = combined
        _log(
            state,
            "retrieve",
            question_type=qtype,
            query=query,
            query_variants=variants,
            retrieval_profile=profile.model_dump(),
            n_hits=len(combined),
            raw_attempt_hit_count=len(raw_attempt_hits),
            retry_evidence_fusion=retry_fused,
            retry_evidence_attempts=len(state.get("retry_evidence", [])),
            subqueries=subqueries,
            reused_subquery_evidence=reused,
            entities=entities,
            entity_document_ids=state.get("entity_document_ids", {}),
            # What was retrieved, in rank order, so a citation can be checked
            # against the evidence the model actually saw.
            hits=[f"{h.get('document_id')}:p{h.get('page')}" for h in combined],
            # Preserve identity for evaluation without logging passage text.
            # Legacy hits above remain available to existing trace consumers.
            retrieval_hits=[
                {
                    **{
                        key: h.get(key)
                        for key in (
                            "chunk_id",
                            "document_id",
                            "source_doc_uid",
                            "filename",
                            "source_filename",
                            "page",
                            "score",
                        )
                        if h.get(key) is not None
                    },
                    "metadata": {
                        key: h["metadata"][key]
                        for key in ("source_doc_uid", "original_filename")
                        if isinstance(h.get("metadata"), dict)
                        and h["metadata"].get(key) is not None
                    },
                }
                for h in combined
            ],
        )
        return state

    async def grade_node(state: AgentState) -> AgentState:
        grade = llm.grade(state["question"], state.get("evidence", []))
        profile = policy(state["question_type"], state.get("retry_count", 0))
        top_score = max(
            (
                float(h.get("score") or 0)
                for h in state.get("evidence", [])
                if not h.get("context_only")
            ),
            default=0,
        )
        gate = not settings.GRADE_CALIBRATION or (
            top_score >= profile.min_score
            and grade.confidence >= profile.min_grade_confidence
        )
        state["sufficient"] = grade.sufficient and gate
        state["grade_reason"] = grade.reason
        if not gate:
            state["grade_reason"] += " Calibrated evidence gate rejected this attempt."
        _log_with_usage(
            state,
            "grade",
            llm,
            sufficient=state["sufficient"],
            model_sufficient=grade.sufficient,
            top_score=top_score,
            min_score=profile.min_score,
            min_grade_confidence=profile.min_grade_confidence,
            confidence=grade.confidence,
            reason=grade.reason,
        )
        return state

    def route_after_grade(
        state: AgentState,
    ) -> Literal["sufficient", "retry", "insufficient"]:
        if state.get("sufficient"):
            return "sufficient"
        if state.get("retry_count", 0) < settings.MAX_RETRIES:
            return "retry"
        return "insufficient"

    async def reformulate_node(state: AgentState) -> AgentState:
        state["retry_count"] = state.get("retry_count", 0) + 1
        new_query = llm.reformulate(
            state["question"], state["current_query"], state["retry_count"]
        )
        state["current_query"] = new_query
        _log_with_usage(
            state, "reformulate", llm, attempt=state["retry_count"], new_query=new_query
        )
        return state

    async def reason_node(state: AgentState) -> AgentState:
        evidence = state.get("evidence", [])
        # EXTRACT_GRADED_FIGURES: only an approving grade reaches this node, and
        # its note names the figures and passages the grader checked.
        grade_note = (
            state.get("grade_reason", "")
            if settings.EXTRACT_GRADED_FIGURES
            and state.get("sufficient")
            and state["question_type"] == "numerical"
            else ""
        )
        if grade_note:
            _log(state, "graded_figures", note=grade_note)
            extraction = llm.extract(
                state["question"], state["question_type"], evidence, grade_note=grade_note
            )
        else:
            extraction = llm.extract(state["question"], state["question_type"], evidence)
        _log_with_usage(state, "extract", llm)
        if settings.ANSWER_REPAIR:
            from app.answer_repair import repair_extraction

            extraction, repair_status = repair_extraction(
                llm, state["question"], state["question_type"], evidence, extraction
            )
            if repair_status != "not_needed":
                _log_with_usage(state, "answer_repair", llm, status=repair_status)
        if (
            settings.CALC_LIST_FALLBACK
            and isinstance(extraction, ExtractionCalculated)
            and _is_value_list(extraction.formula)
        ):
            # A086: a "respectively" list question was classified numerical, and
            # extraction wrote its three values as '38.6, 23.6, 12.2', which the
            # calculator rejects. Answer it as a list instead of declining.
            rejected_formula = extraction.formula
            extraction = llm.extract(state["question"], "table", evidence)
            _log_with_usage(state, "list_fallback", llm, rejected_formula=rejected_formula)
        # True when citations are the passages extraction named, False when
        # they fell back to top-ranked hits, None when nothing was cited.
        cited_by_model = None

        try:
            if isinstance(extraction, ExtractionCalculated):
                try:
                    value = calculate(extraction.formula)
                except CalculatorError as e:
                    answer = {
                        "answer_type": "insufficient_evidence",
                        "evidence": [],
                        "params": {
                            "reason": f"Calculator rejected formula '{extraction.formula}': {e}"
                        },
                    }
                    validate_answer_dict(answer)
                    state["answer"] = answer
                    _log(
                        state, "reason", shape="insufficient", cause="calculator_error"
                    )
                    return state

                citations, cited_by_model = _citations(
                    evidence,
                    extraction.evidence_indexes,
                    max(2, extraction.operand_count),
                )
                answer = {
                    "answer_type": "calculated",
                    "evidence": citations or _top_evidence(evidence, 1),
                    "params": {"value": round(value, 6), "formula": extraction.formula},
                }

            elif isinstance(extraction, ExtractionDirect):
                citations, cited_by_model = _citations(
                    evidence, extraction.evidence_indexes, 1
                )
                answer = {
                    "answer_type": "direct",
                    "evidence": citations,
                    "params": {"value": extraction.value},
                }

            elif isinstance(extraction, ExtractionMultiSpan):
                citations, cited_by_model = _citations(
                    evidence, extraction.evidence_indexes, max(1, len(evidence))
                )
                answer = {
                    "answer_type": "multi_span",
                    "evidence": citations,
                    "params": {"values": extraction.values},
                }

            else:  # ExtractionInsufficient
                reason = (
                    extraction.reason
                    if isinstance(extraction, ExtractionInsufficient)
                    else "Insufficient evidence."
                )
                answer = {
                    "answer_type": "insufficient_evidence",
                    "evidence": [],
                    "params": {"reason": reason},
                }

            if settings.ANSWER_NORMALIZATION:
                answer = normalize_answer(answer)
            validate_answer_dict(answer)  # local schema self-check before returning

        except Exception as e:  # noqa: BLE001 - final safety net: never return a malformed/hallucinated answer
            cited_by_model = None
            answer = {
                "answer_type": "insufficient_evidence",
                "evidence": [],
                "params": {
                    "reason": f"Internal validation failure, refusing to guess: {e}"
                },
            }

        state["answer"] = answer
        _log(
            state,
            "reason",
            answer_type=answer["answer_type"],
            cited_by_model=cited_by_model,
        )
        return state

    async def build_insufficient_node(state: AgentState) -> AgentState:
        answer = {
            "answer_type": "insufficient_evidence",
            "evidence": [],
            "params": {
                "reason": state.get(
                    "grade_reason",
                    "No sufficient evidence found after retrying retrieval.",
                )
            },
        }
        validate_answer_dict(answer)
        state["answer"] = answer
        _log(state, "build_insufficient", reason=answer["params"]["reason"])
        return state

    # ------------------------------------------------------------- wiring --
    graph = StateGraph(AgentState)
    graph.add_node("classify", classify_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("reformulate", reformulate_node)
    graph.add_node("reason", reason_node)
    graph.add_node("build_insufficient", build_insufficient_node)

    graph.add_edge(START, "classify")
    if settings.QUERY_DECOMPOSITION:
        graph.add_node("decompose", decompose_node)
        graph.add_edge("classify", "decompose")
        graph.add_edge("decompose", "retrieve")
    else:
        graph.add_edge("classify", "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade",
        route_after_grade,
        {
            "sufficient": "reason",
            "retry": "reformulate",
            "insufficient": "build_insufficient",
        },
    )
    graph.add_edge("reformulate", "retrieve")
    graph.add_edge("reason", END)
    graph.add_edge("build_insufficient", END)

    return graph.compile()


_compiled_graph = None


def get_compiled_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


async def run_agent(question: str, document_id: Optional[str] = None) -> AgentState:
    graph = get_compiled_graph()
    initial_state: AgentState = {
        "question": question,
        "document_id": document_id,
        "trace": [],
    }
    final_state = await graph.ainvoke(initial_state)
    return final_state
