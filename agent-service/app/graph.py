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

import logging
import time
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict

from langgraph.graph import StateGraph, START, END

from app.calculator import CalculatorError, calculate
from app.config import settings
from app.decomposition import QueryPlan, merge_query_results
from app.llm import (
    ExtractionCalculated,
    ExtractionDirect,
    ExtractionInsufficient,
    ExtractionMultiSpan,
    get_llm,
)
from app.schemas import validate_answer_dict
from app.tools import filter_documents, search_bm25, search_documents, search_tables


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


def _citations(evidence: List[dict], indexes: List[int], fallback_n: int) -> Tuple[List[dict], bool]:
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
        if isinstance(index, bool) or not isinstance(index, int) or not 1 <= index <= len(evidence):
            continue
        hit = evidence[index - 1]
        key = (hit["document_id"], hit["page"], hit.get("section"))
        if key not in seen:
            seen.add(key)
            picked.append(hit)
    if picked:
        return _top_evidence(picked, len(picked)), True
    return _top_evidence(evidence, fallback_n), False


def build_graph():
    llm = get_llm()

    # ------------------------------------------------------------- nodes --
    async def classify_node(state: AgentState) -> AgentState:
        result = llm.classify(state["question"])
        state["question_type"] = result.question_type
        state["current_query"] = result.search_query
        state["retry_count"] = 0
        _log_with_usage(state, "classify", llm, question_type=result.question_type, query=result.search_query)
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
            _log_with_usage(state, "decompose", llm, status=status,
                            subqueries=state["subqueries"],
                            latency_ms=(time.perf_counter() - started) * 1000)
        except Exception as exc:
            # Planning is optional. Fall back to the ordinary query visibly;
            # failures in actual retrieval, grading and extraction still raise.
            logging.getLogger("agent-service").warning(
                "Query decomposition failed (%s); using the original query", type(exc).__name__
            )
            _log_with_usage(state, "decompose", llm, status="fallback",
                            error_type=type(exc).__name__, subqueries=[],
                            latency_ms=(time.perf_counter() - started) * 1000)
        return state

    async def retrieve_query(query: str, qtype: str, doc_id: Optional[str]) -> tuple:
        if qtype == "numerical":
            return await search_documents(query, document_id=doc_id), await search_tables(query, document_id=doc_id)
        if qtype == "table":
            return await search_tables(query, document_id=doc_id), await search_documents(query, document_id=doc_id)
        return await search_documents(query, document_id=doc_id), await search_bm25(query, document_id=doc_id)

    async def retrieve_node(state: AgentState) -> AgentState:
        query = state["current_query"]
        doc_id = state.get("document_id")
        qtype = state["question_type"]

        combined = _dedup_and_rank(*(await retrieve_query(query, qtype, doc_id)))

        # Direct metadata lookup (non-vector path #2): when the caller scoped
        # the question to a specific document, pull everything indexed for
        # it and fold that in too -- cheap, exact, and doesn't depend on the
        # embedding/BM25 query matching the right words.
        if doc_id:
            filter_hits = await filter_documents({"document_id": doc_id})
            combined = _dedup_and_rank(combined, filter_hits)

        subqueries = state.get("subqueries", [])
        reused = bool(state.get("subquery_evidence"))
        if subqueries:
            if not reused:
                rankings = []
                for subquery in subqueries:
                    started = time.perf_counter()
                    lists = await retrieve_query(subquery, qtype, doc_id)
                    hits = sorted((h for group in lists for h in group),
                                  key=lambda h: h.get("score", 0), reverse=True)
                    rankings.append(hits)
                    _log(state, "retrieve_subquery", query=subquery,
                         latency_ms=(time.perf_counter() - started) * 1000,
                         candidate_count=len(hits),
                         document_ids=list(dict.fromkeys(h.get("document_id") for h in hits)))
                state["subquery_evidence"] = rankings
            # Within this request only: retain operand evidence through retries
            # while the existing reformulator searches for additional context.
            combined = merge_query_results(
                [*state["subquery_evidence"], combined], settings.TOP_K_FINAL
            )

        state["evidence"] = combined
        _log(
            state, "retrieve", question_type=qtype, query=query, n_hits=len(combined),
            subqueries=subqueries, reused_subquery_evidence=reused,
            # What was retrieved, in rank order, so a citation can be checked
            # against the evidence the model actually saw.
            hits=[f"{h.get('document_id')}:p{h.get('page')}" for h in combined],
            # Preserve identity for evaluation without logging passage text.
            # Legacy hits above remain available to existing trace consumers.
            retrieval_hits=[
                {
                    **{key: h.get(key) for key in (
                        "chunk_id", "document_id", "source_doc_uid", "filename",
                        "source_filename", "page", "score",
                    ) if h.get(key) is not None},
                    "metadata": {
                        key: h["metadata"][key]
                        for key in ("source_doc_uid", "original_filename")
                        if isinstance(h.get("metadata"), dict) and h["metadata"].get(key) is not None
                    },
                }
                for h in combined
            ],
        )
        return state

    async def grade_node(state: AgentState) -> AgentState:
        grade = llm.grade(state["question"], state.get("evidence", []))
        state["sufficient"] = grade.sufficient
        state["grade_reason"] = grade.reason
        _log_with_usage(
            state, "grade", llm,
            sufficient=grade.sufficient, confidence=grade.confidence, reason=grade.reason,
        )
        return state

    def route_after_grade(state: AgentState) -> Literal["sufficient", "retry", "insufficient"]:
        if state.get("sufficient"):
            return "sufficient"
        if state.get("retry_count", 0) < settings.MAX_RETRIES:
            return "retry"
        return "insufficient"

    async def reformulate_node(state: AgentState) -> AgentState:
        state["retry_count"] = state.get("retry_count", 0) + 1
        new_query = llm.reformulate(state["question"], state["current_query"], state["retry_count"])
        state["current_query"] = new_query
        _log_with_usage(state, "reformulate", llm, attempt=state["retry_count"], new_query=new_query)
        return state

    async def reason_node(state: AgentState) -> AgentState:
        evidence = state.get("evidence", [])
        extraction = llm.extract(state["question"], state["question_type"], evidence)
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
                        "params": {"reason": f"Calculator rejected formula '{extraction.formula}': {e}"},
                    }
                    validate_answer_dict(answer)
                    state["answer"] = answer
                    _log(state, "reason", shape="insufficient", cause="calculator_error")
                    return state

                citations, cited_by_model = _citations(
                    evidence, extraction.evidence_indexes, max(2, extraction.operand_count)
                )
                answer = {
                    "answer_type": "calculated",
                    "evidence": citations or _top_evidence(evidence, 1),
                    "params": {"value": round(value, 6), "formula": extraction.formula},
                }

            elif isinstance(extraction, ExtractionDirect):
                citations, cited_by_model = _citations(evidence, extraction.evidence_indexes, 1)
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
                reason = extraction.reason if isinstance(extraction, ExtractionInsufficient) else "Insufficient evidence."
                answer = {
                    "answer_type": "insufficient_evidence",
                    "evidence": [],
                    "params": {"reason": reason},
                }

            validate_answer_dict(answer)  # local schema self-check before returning

        except Exception as e:  # noqa: BLE001 - final safety net: never return a malformed/hallucinated answer
            cited_by_model = None
            answer = {
                "answer_type": "insufficient_evidence",
                "evidence": [],
                "params": {"reason": f"Internal validation failure, refusing to guess: {e}"},
            }

        state["answer"] = answer
        _log_with_usage(state, "reason", llm, answer_type=answer["answer_type"], cited_by_model=cited_by_model)
        return state

    async def build_insufficient_node(state: AgentState) -> AgentState:
        answer = {
            "answer_type": "insufficient_evidence",
            "evidence": [],
            "params": {
                "reason": state.get(
                    "grade_reason", "No sufficient evidence found after retrying retrieval."
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
        {"sufficient": "reason", "retry": "reformulate", "insufficient": "build_insufficient"},
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
