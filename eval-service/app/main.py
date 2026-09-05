import json
import logging
from pathlib import Path
from typing import Any, List, Optional, Union

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

from . import tracing
from .benchmark import BenchmarkConfig, load_report, run_benchmark

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
    return trace


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


class BenchmarkRunRequest(BaseModel):
    system_url: str
    questions: List[dict]
    validator_url: Optional[str] = None
    answer_key: Optional[str] = None
    retrieval_k: int = 5
    gold_field: str = "answer"
    relevant_docs_field: str = "relevant_document_ids"


@app.post("/benchmark/run")
def benchmark_run(req: BenchmarkRunRequest):
    config = BenchmarkConfig(
        system_url=req.system_url,
        questions=req.questions,
        validator_url=req.validator_url,
        answer_key=req.answer_key,
        retrieval_k=req.retrieval_k,
        gold_field=req.gold_field,
        relevant_docs_field=req.relevant_docs_field,
    )
    report = run_benchmark(config)
    return {"run_id": report["run_id"], "summary": report["summary"]}


@app.post("/benchmark/run_from_file")
async def benchmark_run_from_file(
    system_url: str = Form(...),
    validator_url: Optional[str] = Form(None),
    answer_key: Optional[str] = Form(None),
    retrieval_k: int = Form(5),
    gold_field: str = Form("answer"),
    relevant_docs_field: str = Form("relevant_document_ids"),
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
        gold_field=gold_field,
        relevant_docs_field=relevant_docs_field,
    )
    report = run_benchmark(config)
    return {"run_id": report["run_id"], "summary": report["summary"]}


@app.get("/benchmark/results/{run_id}")
def benchmark_results(run_id: str):
    report = load_report(run_id)
    if report is None:
        raise HTTPException(status_code=404, detail="run_id not found")
    return report
