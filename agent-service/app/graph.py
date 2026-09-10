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
- `grade` branches on sufficient vs. insufficient evidence, with a bounded
  retry loop that reformulates the query on weak evidence.
- `reason` branches on the extracted answer shape (direct / calculated /
  multi_span / insufficient) and is the ONLY place arithmetic happens --
  always via the deterministic calculate() tool, never the LLM directly.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict

from langgraph.graph import StateGraph, START, END

from app.calculator import CalculatorError, calculate
from app.config import settings
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
    best: Dict[tuple, dict] = {}
    for hits in hit_lists:
        for h in hits:
            key = (h.get("document_id"), h.get("page"), h.get("section"))
            if key not in best or h.get("score", 0) > best[key].get("score", 0):
                best[key] = h
    ranked = sorted(best.values(), key=lambda h: h.get("score", 0), reverse=True)
    return ranked[: settings.TOP_K_FINAL]


def _top_evidence(hits: List[dict], n: int) -> List[dict]:
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

    async def retrieve_node(state: AgentState) -> AgentState:
        query = state["current_query"]
        doc_id = state.get("document_id")
        qtype = state["question_type"]

        if qtype == "numerical":
            # numbers often live in tables, but may also be narrated in text
            vec_hits, table_hits = await search_documents(query, document_id=doc_id), await search_tables(query, document_id=doc_id)
            combined = _dedup_and_rank(vec_hits, table_hits)
        elif qtype == "table":
            table_hits, vec_hits = await search_tables(query, document_id=doc_id), await search_documents(query, document_id=doc_id)
            combined = _dedup_and_rank(table_hits, vec_hits)
        else:  # "text" -> hybrid dense + lexical, the required non-vector path
            vec_hits, bm25_hits = await search_documents(query, document_id=doc_id), await search_bm25(query, document_id=doc_id)
            combined = _dedup_and_rank(vec_hits, bm25_hits)

        # Direct metadata lookup (non-vector path #2): when the caller scoped
        # the question to a specific document, pull everything indexed for
        # it and fold that in too -- cheap, exact, and doesn't depend on the
        # embedding/BM25 query matching the right words.
        if doc_id:
            filter_hits = await filter_documents({"document_id": doc_id})
            combined = _dedup_and_rank(combined, filter_hits)

        state["evidence"] = combined
        _log(
            state, "retrieve", question_type=qtype, query=query, n_hits=len(combined),
            # What was retrieved, in rank order, so a citation can be checked
            # against the evidence the model actually saw.
            hits=[f"{h.get('document_id')}:p{h.get('page')}" for h in combined],
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
