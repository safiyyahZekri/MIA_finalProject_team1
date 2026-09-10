"""
agent-service: the LangGraph "brain" of Project LEDGER.

POST /agent/query -> the orchestrator's contract: runs the graph and
                      returns ONLY the raw answer object
                      ({"answer_type", "evidence", "params"}), nothing else.
POST /answer       -> same graph, but wrapped with question_type/
                      retries_used/trace -- for your own testing and for
                      eval-service, which needs the trace.
GET  /health       -> liveness check.

NOTE on architecture: per the spec, the *orchestrator* is responsible for
forwarding the produced answer to answer-validator-api before it reaches the
user. This service already self-validates the answer against the same
schema (see app.schemas) so a malformed answer never leaves this service in
the first place. If ANSWER_VALIDATOR_URL is set, it will *also* call the
validator directly and include its verdict in the response, which is
convenient when running/demoing agent-service standalone.
"""
from __future__ import annotations

import logging

import httpx
from fastapi import FastAPI, HTTPException

from app.config import settings
from app.graph import run_agent
from app.schemas import AnswerRequest, AnswerResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agent-service")

app = FastAPI(title="LEDGER agent-service", version="1.0.0")


@app.get("/health")
async def health():
    body = {"status": "ok", "llm_provider": settings.LLM_PROVIDER}
    # The settings experiments vary, so a saved run records exactly what ran.
    body["config"] = {
        "top_k_final": settings.TOP_K_FINAL,
        "max_retries": settings.MAX_RETRIES,
        "query_decomposition": settings.QUERY_DECOMPOSITION,
        "grade_require_entity_match": settings.GRADE_REQUIRE_ENTITY_MATCH,
        "answer_format_fixes": settings.ANSWER_FORMAT_FIXES,
        "grade_company_context": settings.GRADE_COMPANY_CONTEXT,
        "grade_require_named_table": settings.GRADE_REQUIRE_NAMED_TABLE,
        "rank_fusion_merge": settings.RANK_FUSION_MERGE,
    }
    if settings.LLM_PROVIDER == "anthropic":
        body["config"].update(
            model=settings.ANTHROPIC_MODEL,
            effort=settings.ANTHROPIC_EFFORT or "api-default",
            max_tokens=settings.ANTHROPIC_MAX_TOKENS,
            fallbacks=settings.ANTHROPIC_FALLBACKS or None,
        )
    if settings.LLM_PROVIDER == "gemini":
        body["config"].update(
            model=settings.GEMINI_MODEL,
            max_output_tokens=settings.GEMINI_MAX_OUTPUT_TOKENS,
            temperature=settings.GEMINI_TEMPERATURE,
        )
        if not settings.GEMINI_API_KEY:
            body["status"] = "degraded"
            body["hint"] = "GEMINI_API_KEY is not set in .env -- create a key in Google AI Studio"
    if settings.LLM_PROVIDER == "ollama":
        from app.llm import OllamaLLM  # local import: avoid httpx.Client at module load

        reachable = OllamaLLM().ping()
        body["ollama"] = {
            "reachable": reachable,
            "base_url": settings.OLLAMA_BASE_URL,
            "model": settings.OLLAMA_MODEL,
        }
        if not reachable:
            body["status"] = "degraded"
            body["hint"] = (
                f"Can't reach Ollama at {settings.OLLAMA_BASE_URL}. "
                f"Run `ollama serve` and `ollama pull {settings.OLLAMA_MODEL}`, "
                "or the graph will fall back to the offline mock heuristic per call."
            )
    elif settings.LLM_PROVIDER == "groq":
        from app.llm import GroqError, GroqLLM  # local import: avoid httpx.Client at module load

        if not settings.GROQ_API_KEY:
            body["status"] = "degraded"
            body["groq"] = {"reachable": False, "model": settings.GROQ_MODEL}
            body["hint"] = "GROQ_API_KEY is not set in .env -- get one from https://console.groq.com"
        else:
            try:
                reachable = GroqLLM().ping()
            except GroqError:
                reachable = False
            body["groq"] = {"reachable": reachable, "model": settings.GROQ_MODEL}
            if not reachable:
                body["status"] = "degraded"
                body["hint"] = (
                    "Can't reach Groq or the API key was rejected. Check GROQ_API_KEY "
                    "and GROQ_MODEL in .env, or the graph will fall back to the "
                    "offline mock heuristic per call."
                )
    return body


async def _call_validator(answer: dict) -> dict | None:
    if not settings.ANSWER_VALIDATOR_URL:
        return None
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.post(
                f"{settings.ANSWER_VALIDATOR_URL.rstrip('/')}/validate_answer", json=answer
            )
            return {"status_code": resp.status_code, "body": resp.json() if resp.content else None}
    except httpx.HTTPError as exc:
        logger.warning("answer-validator-api call failed: %s", exc)
        return {"error": str(exc)}


def _usage_summary(trace: list) -> dict:
    """LLM calls, tokens and approximate cost for one question, read from the
    per-call usage the graph already attaches to its trace steps."""
    calls = [step["usage"] for step in trace if isinstance(step, dict) and step.get("usage")]
    input_tokens = sum(usage.get("prompt_tokens") or 0 for usage in calls)
    output_tokens = sum(usage.get("completion_tokens") or 0 for usage in calls)
    cost_usd = None
    rates = {
        "anthropic": (settings.ANTHROPIC_INPUT_USD_PER_MTOK, settings.ANTHROPIC_OUTPUT_USD_PER_MTOK),
        "gemini": (settings.GEMINI_INPUT_USD_PER_MTOK, settings.GEMINI_OUTPUT_USD_PER_MTOK),
    }.get(settings.LLM_PROVIDER)
    if rates and any(rates):
        cost_usd = round(
            input_tokens / 1_000_000 * rates[0] + output_tokens / 1_000_000 * rates[1],
            6,
        )
    return {
        "llm_calls": len(calls),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "tokens_used": input_tokens + output_tokens,
        "cost_usd": cost_usd,
    }


async def _run_and_validate(req: AnswerRequest) -> dict:
    """Shared logic for both endpoints: run the graph, guarantee an
    `answer`, and optionally call answer-validator-api directly. Returns
    the full final_state (answer + question_type + retry_count + trace)
    -- callers pick what to expose."""
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=422, detail="question must not be empty")

    try:
        final_state = await run_agent(req.question, req.document_id)
    except Exception as exc:  # noqa: BLE001 -- re-raised as an HTTP error, not swallowed
        # An LLM or retrieval failure the SDK's own retries could not absorb.
        # Report it with its cause so a caller records an error -- never an
        # answer that could be scored as if the model had seen the question.
        logger.exception("agent run failed")
        raise HTTPException(status_code=503, detail=f"{type(exc).__name__}: {exc}") from exc
    answer = final_state.get("answer")
    if answer is None:
        # Should be unreachable: every graph path sets `answer`. Fail safe rather
        # than return a hallucinated/empty result.
        answer = {
            "answer_type": "insufficient_evidence",
            "evidence": [],
            "params": {"reason": "Agent graph terminated without producing an answer."},
        }
    final_state["answer"] = answer

    validator_result = await _call_validator(answer)
    trace = final_state.get("trace", [])
    if validator_result is not None:
        trace.append({"step": "answer_validator_api", **validator_result})
    final_state["trace"] = trace

    return final_state


@app.post("/agent/query")
async def agent_query(req: AnswerRequest):
    """Orchestrator's contract: input {"question", "document_id"} ->
    output is ONLY the answer object itself, e.g.
    {"answer_type": "direct", "evidence": [...], "params": {...}} --
    no wrapper, no trace, nothing else at the top level."""
    final_state = await _run_and_validate(req)
    return final_state["answer"]


@app.post("/answer", response_model=AnswerResponse)
async def answer_question(req: AnswerRequest):
    """Same graph as /agent/query, but wrapped with question_type/
    retries_used/trace -- useful for your own testing and for
    eval-service, which needs the trace. The orchestrator should call
    /agent/query instead."""
    final_state = await _run_and_validate(req)
    trace = final_state.get("trace", [])
    return AnswerResponse(
        answer=final_state["answer"],
        question_type=final_state.get("question_type", "unknown"),
        retries_used=final_state.get("retry_count", 0),
        trace=trace,
        **_usage_summary(trace),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=False)
