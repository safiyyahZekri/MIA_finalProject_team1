import json
import logging
from pathlib import Path
from typing import Any, List, Optional, Union

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

from . import tracing
from .benchmark import BenchmarkConfig, load_report, run_benchmark
from .retrieval_benchmark import (
    RetrievalBenchmarkConfig,
    VARIANTS,
    run_ablation,
    run_retrieval_benchmark,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("eval-service")

app = FastAPI(
    title="LEDGER Evaluation & Observability Service",
    description="Centralized Langfuse tracing + automated TAT-DQA benchmark runner.",
    version="1.0.0",
)


@app.get("/health")
def health():
    return {"status": "ok", "langfuse_enabled": tracing.is_langfuse_enabled()}


# ---------------------------------------------------------------------------
# Tracing — other services POST here instead of each holding Langfuse creds
# ---------------------------------------------------------------------------


class StartTraceRequest(BaseModel):
    name: str
    metadata: Optional[dict] = None


class StepRequest(BaseModel):
    trace_id: str
    name: str
    input: Optional[Any] = None
    output: Optional[Any] = None
    latency_ms: Optional[float] = None
    tokens: Optional[dict] = None
    metadata: Optional[dict] = None


class EndTraceRequest(BaseModel):
    trace_id: str
    output: Optional[Any] = None


@app.post("/trace/start")
def trace_start(req: StartTraceRequest):
    trace_id = tracing.start_trace(req.name, req.metadata)
    return {"trace_id": trace_id}


@app.post("/trace/step")
def trace_step(req: StepRequest):
    tracing.log_step(
        req.trace_id, req.name, req.input, req.output, req.latency_ms, req.tokens, req.metadata
    )
    return {"status": "logged"}


@app.post("/trace/end")
def trace_end(req: EndTraceRequest):
    tracing.end_trace(req.trace_id, req.output)
    return {"status": "closed"}


@app.get("/trace/{trace_id}")
def trace_get(trace_id: str):
    trace = tracing.get_trace(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="trace not found")
    trace = dict(trace)
    trace["trace_url"] = tracing.get_trace_url(trace_id)
    return trace


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


class BenchmarkRunRequest(BaseModel):
    system_url: str
    questions: List[dict]
    validator_url: Optional[str] = None
    answer_key: Optional[str] = "answer"
    retrieval_k: int = 5
    question_field: str = "question_text"
    gold_field: str = "ground_truth_answer"
    scale_field: str = "scale"
    scope_to_gold_document: bool = False
    timeout_s: float = 60.0
    delay_between_questions_s: float = 0.0


@app.post("/benchmark/run")
def benchmark_run(req: BenchmarkRunRequest):
    config = BenchmarkConfig(
        system_url=req.system_url,
        questions=req.questions,
        validator_url=req.validator_url,
        answer_key=req.answer_key,
        retrieval_k=req.retrieval_k,
        question_field=req.question_field,
        gold_field=req.gold_field,
        scale_field=req.scale_field,
        scope_to_gold_document=req.scope_to_gold_document,
        timeout_s=req.timeout_s,
        delay_between_questions_s=req.delay_between_questions_s,
    )
    report = run_benchmark(config)
    return {"run_id": report["run_id"], "summary": report["summary"]}


@app.post("/benchmark/run_from_file")
async def benchmark_run_from_file(
    system_url: str = Form(...),
    validator_url: Optional[str] = Form(None),
    answer_key: Optional[str] = Form("answer"),
    retrieval_k: int = Form(5),
    question_field: str = Form("question_text"),
    gold_field: str = Form("ground_truth_answer"),
    scale_field: str = Form("scale"),
    scope_to_gold_document: bool = Form(False),
    timeout_s: float = Form(60.0),
    delay_between_questions_s: float = Form(0.0),
    file: UploadFile = File(...),
):
    """Convenience endpoint: upload the 100-question practice set (JSON
    array or JSONL) directly instead of inlining it in the request body."""
    raw = (await file.read()).decode("utf-8")
    if file.filename and file.filename.endswith(".jsonl"):
        questions = [json.loads(line) for line in raw.splitlines() if line.strip()]
    else:
        questions = json.loads(raw)

    config = BenchmarkConfig(
        system_url=system_url,
        questions=questions,
        validator_url=validator_url,
        answer_key=answer_key,
        retrieval_k=retrieval_k,
        question_field=question_field,
        gold_field=gold_field,
        scale_field=scale_field,
        scope_to_gold_document=scope_to_gold_document,
    )
    report = run_benchmark(config)
    return {"run_id": report["run_id"], "summary": report["summary"]}


@app.get("/benchmark/results/{run_id}")
def benchmark_results(run_id: str):
    report = load_report(run_id)
    if report is None:
        raise HTTPException(status_code=404, detail="run_id not found")
    return report


class RetrievalBenchmarkRunRequest(BaseModel):
    retrieval_url: str = "http://localhost:8002"
    questions: List[dict]
    candidate_k: int = 30
    top_k: int = 10
    variant: str = "hybrid_reranker"


@app.post("/benchmark/retrieval/run")
def retrieval_benchmark_run(req: RetrievalBenchmarkRunRequest):
    variants = {variant.name: variant for variant in VARIANTS}
    if req.variant not in variants:
        raise HTTPException(
            status_code=422,
            detail=f"variant must be one of: {', '.join(variants)}",
        )
    report = run_retrieval_benchmark(
        RetrievalBenchmarkConfig(
            retrieval_url=req.retrieval_url,
            questions=req.questions,
            variant=variants[req.variant],
            candidate_k=req.candidate_k,
            top_k=req.top_k,
            scope_to_gold_document=False,
        )
    )
    return report


@app.post("/benchmark/retrieval/ablation")
def retrieval_ablation_run(req: RetrievalBenchmarkRunRequest):
    return run_ablation(
        req.retrieval_url,
        req.questions,
        candidate_k=req.candidate_k,
        top_k=req.top_k,
    )
