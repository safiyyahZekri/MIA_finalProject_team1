"""
Automated benchmark harness (spec: "a held-out subset must be used to
automatically score the complete system end to end: question -> your system
-> predicted answer -> compare to ground truth -> metrics").

Input format: the practice-question record schema (confirmed against
`questions_setA_practice.json` + `record_schema_guide_ar.md`), NOT the
generic TAT-DQA shape. Each record looks like:

    {
      "question_id": "A001",
      "question_text": "...",
      "ground_truth_answer": 304811,           # number | string | list | null
      "answer_type": "arithmetic",             # span/multi-span/arithmetic/count/unanswerable
      "scale": "thousand",                     # "", "thousand", "million", "billion", "percent"
      "source_doc_uid": null,                  # identity/scoped-debug id, not used by default
      "gold_evidence": [
        {"source_doc_uid": "7d631ffe...", "source_document": "...", ...},
        ...
      ]
    }

There is no flat `relevant_document_ids` list in the real data — the set of
relevant document ids for retrieval metrics is derived from
`gold_evidence[*].source_doc_uid`. There is no `document_id` field either;
the optional single-document diagnostic-scoping id is `source_doc_uid`,
which is `null` for multi-document/derived questions. Normal benchmark runs
are explicitly corpus-wide and do not send this gold ID to the system.

Confirmed system contract: `system_url` should point at agent-service's
`POST /answer` endpoint, built specifically for eval-service's use. It takes
`{"question": ..., "document_id": ...}` and returns:

    {
      "answer": {"answer_type": "...", "evidence": [...], "params": {...}},
      "question_type": "...",
      "retries_used": 2,
      "trace": [...]
    }

This was confirmed end-to-end against the real `feature/agent-service`
branch (100 practice questions, 0 errors, 100% schema validity), so
`answer_key="answer"` is the default here — override it only if a future
system response shape doesn't nest the answer under `"answer"`.
`trace` is confirmed to be a list (the agent's internal step history) but
its per-item shape isn't part of the confirmed contract, so it's still
treated as opaque: read verbatim if present, stored alongside the result,
never required and never validated. It is also never forwarded to
answer-validator-api, since that endpoint's schema forbids unknown keys —
only the bare `{answer_type, evidence, params}` object is sent there.
`retries_used`, when present, is captured as a system-performance metric
alongside `llm_calls` / `tokens_used` / `cost_usd`.

Independent of whatever agent-service reports, this harness opens its own
Langfuse trace for every question (via app.tracing) so every benchmarked
question is diagnosable from eval-service's own records regardless of
whether agent-service's tracing is wired up. Each result carries an
eval-service `trace_id` plus, separately and clearly labeled, whatever
`trace` payload the system under test returned.

For each question the harness:
  1. Opens an eval-service trace and times the round trip.
  2. Extracts the core answer object and, if present, the optional
     `trace` / `question_type` fields.
  3. Sends the core answer object to answer-validator-api to confirm schema
     validity.
  4. Scores EM / F1 / numerical accuracy against the gold answer (with
     scale normalization), or, for `unanswerable` gold questions, whether
     the system correctly abstained.
  5. If gold evidence document ids are available, scores Recall@K /
     Precision@K / Hit Rate / MRR against the predicted evidence.
  6. Logs question, response, and scoring outcome as trace steps, then
     closes the trace.
  7. Aggregates everything, including a short list of failed examples with
     their trace_id, for failure-analysis follow-up.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from . import metrics as m
from . import tracing

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

_CORE_ANSWER_KEYS = ("answer_type", "evidence", "params")


def _extract_answer_value(answer: dict):
    params = answer.get("params", {}) if isinstance(answer, dict) else {}
    if "values" in params:
        return params["values"]
    if "value" in params:
        return params["value"]
    if "reason" in params:
        return None  # insufficient_evidence has no comparable value
    return None


def _extract_retrieved_doc_ids(answer: dict) -> List[str]:
    evidence = answer.get("evidence", []) if isinstance(answer, dict) else []
    return [e.get("document_id") for e in evidence if isinstance(e, dict) and e.get("document_id")]


def _extract_relevant_doc_ids(record: dict) -> List[str]:
    """The real schema has no flat `relevant_document_ids` field — the gold
    document ids live inside `gold_evidence[*].source_doc_uid`."""
    evidence = record.get("gold_evidence") or []
    ids = [e.get("source_doc_uid") for e in evidence if isinstance(e, dict) and e.get("source_doc_uid")]
    return ids


def _extract_scoping_doc_id(record: dict) -> Optional[str]:
    """Single-document scoping id, when the question has exactly one
    source. `null`/absent for cross-document and source-identification
    questions, matching the real record schema."""
    return record.get("source_doc_uid")


def _dig(payload: dict, dotted_key: Optional[str]) -> dict:
    if not dotted_key:
        return payload
    node = payload
    for part in dotted_key.split("."):
        node = node.get(part, {}) if isinstance(node, dict) else {}
    return node


def _split_core_answer_and_metadata(body: Any, answer_key: Optional[str]):
    """Given a raw system response, return (core_answer, system_trace,
    question_type).

    `core_answer` is the bare `{answer_type, evidence, params}` object,
    stripped of any sibling fields (like agent-service's `trace` /
    `question_type`) so it is safe to forward to answer-validator-api,
    which rejects unknown top-level keys. If `answer_key` is set, the
    answer is dug out from that nested path first (metadata is then read
    from the top-level body, not from inside the nested object, matching
    the confirmed agent-service shape where trace/question_type are
    siblings of the answer fields, not nested inside them).
    """
    if not isinstance(body, dict):
        return None, None, None

    nested = _dig(body, answer_key)
    if not isinstance(nested, dict) or "answer_type" not in nested:
        return None, body.get("trace"), body.get("question_type")

    core_answer = {k: nested[k] for k in _CORE_ANSWER_KEYS if k in nested}

    system_trace = body.get("trace") if not answer_key else nested.get("trace", body.get("trace"))
    question_type = body.get("question_type") if not answer_key else nested.get(
        "question_type", body.get("question_type")
    )
    return core_answer, system_trace, question_type


@dataclass
class BenchmarkConfig:
    system_url: str
    questions: List[dict]
    validator_url: Optional[str] = None
    answer_key: Optional[str] = "answer"  # confirmed default: agent-service nests under "answer"
    retrieval_k: int = 5
    timeout_s: float = 60.0
    # Field names matching the confirmed practice-question record schema.
    question_field: str = "question_text"
    gold_field: str = "ground_truth_answer"
    scale_field: str = "scale"
    # Corpus-wide is the production/default behavior. Gold scoping is retained
    # only as an explicit diagnostic mode and must not be used for headline
    # retrieval metrics.
    scope_to_gold_document: bool = False


@dataclass
class QuestionResult:
    question_id: str
    question: str
    predicted_answer: Optional[dict]
    gold: Any
    exact_match: Optional[float]
    f1: Optional[float]
    numerical_accuracy: Optional[float]
    retrieval: Optional[dict]
    latency_ms: float
    schema_valid: Optional[bool]
    validator_reason: Optional[str]
    trace_id: Optional[str] = None
    system_trace: Optional[Any] = None
    question_type: Optional[str] = None
    error: Optional[str] = None
    extra_perf: dict = field(default_factory=dict)

    @property
    def is_failure(self) -> bool:
        if self.error:
            return True
        if self.schema_valid is False:
            return True
        if self.exact_match is not None and self.exact_match < 1.0:
            return True
        return False


def run_benchmark(config: BenchmarkConfig) -> dict:
    run_id = str(uuid.uuid4())
    results: List[QuestionResult] = []

    with httpx.Client(timeout=config.timeout_s) as client:
        for q in config.questions:
            qid = q.get("question_id", str(uuid.uuid4()))
            question_text = q.get(config.question_field, "")
            gold = q.get(config.gold_field)
            gold_scale = q.get(config.scale_field) or ""
            relevant_docs = _extract_relevant_doc_ids(q)
            scoping_doc_id = (
                _extract_scoping_doc_id(q) if config.scope_to_gold_document else None
            )

            trace_id = tracing.start_trace(
                name=f"benchmark:{qid}",
                metadata={"question_id": qid, "question_text": question_text, "run_id": run_id},
            )

            start = time.perf_counter()
            error = None
            answer_obj: Optional[dict] = None
            system_trace = None
            question_type = None
            perf_extra: dict = {}
            request_payload = {"question": question_text, "document_id": scoping_doc_id}
            try:
                resp = client.post(config.system_url, json=request_payload)
                resp.raise_for_status()
                body = resp.json()
                answer_obj, system_trace, question_type = _split_core_answer_and_metadata(
                    body, config.answer_key
                )
                if isinstance(body, dict):
                    for perf_field in ("llm_calls", "tokens_used", "cost_usd", "retries_used"):
                        if perf_field in body:
                            perf_extra[perf_field] = body[perf_field]
            except Exception as exc:  # network error, bad JSON, non-2xx, etc.
                error = str(exc)
            latency_ms = (time.perf_counter() - start) * 1000

            tracing.log_step(
                trace_id,
                "call_system",
                input=request_payload,
                output=answer_obj if answer_obj is not None else {"error": error},
                latency_ms=latency_ms,
            )

            schema_valid = None
            validator_reason = None
            if answer_obj is not None and config.validator_url:
                try:
                    v_resp = client.post(config.validator_url, json=answer_obj)
                    v_body = v_resp.json()
                    schema_valid = v_body.get("valid")
                    validator_reason = v_body.get("reason")
                except Exception as exc:
                    schema_valid = False
                    validator_reason = f"validator call failed: {exc}"
                tracing.log_step(
                    trace_id,
                    "validate_schema",
                    input=answer_obj,
                    output={"valid": schema_valid, "reason": validator_reason},
                )

            em = f1 = num_acc = None
            retrieval_scores = None
            is_unanswerable = q.get("answer_type") == "unanswerable" or gold is None
            if answer_obj is not None and is_unanswerable:
                # Gold is "no answer" — score whether the system correctly
                # abstained (insufficient_evidence) instead of running
                # text/numeric comparisons against a null gold value.
                predicted_type = answer_obj.get("answer_type") if isinstance(answer_obj, dict) else None
                correct_abstain = 1.0 if predicted_type == "insufficient_evidence" else 0.0
                em = f1 = correct_abstain
            elif answer_obj is not None and gold is not None:
                predicted_value = _extract_answer_value(answer_obj)
                em = m.exact_match(predicted_value, gold)
                f1 = m.f1_score(predicted_value, gold)
                num_acc = m.numerical_accuracy(predicted_value, gold, gold_scale=gold_scale)

            if answer_obj is not None and relevant_docs:
                retrieved_ids = _extract_retrieved_doc_ids(answer_obj)
                k = config.retrieval_k
                retrieval_scores = {
                    "recall_at_k": m.recall_at_k(retrieved_ids, relevant_docs, k),
                    "precision_at_k": m.precision_at_k(retrieved_ids, relevant_docs, k),
                    "hit_rate": m.hit_rate(retrieved_ids, relevant_docs, k),
                    "reciprocal_rank": m.reciprocal_rank(retrieved_ids, relevant_docs),
                }

            tracing.log_step(
                trace_id,
                "score",
                input={"gold": gold, "gold_scale": gold_scale},
                output={
                    "exact_match": em,
                    "f1": f1,
                    "numerical_accuracy": num_acc,
                    "retrieval": retrieval_scores,
                },
            )
            tracing.end_trace(trace_id, output={"exact_match": em, "schema_valid": schema_valid})

            results.append(
                QuestionResult(
                    question_id=qid,
                    question=question_text,
                    predicted_answer=answer_obj,
                    gold=gold,
                    exact_match=em,
                    f1=f1,
                    numerical_accuracy=num_acc,
                    retrieval=retrieval_scores,
                    latency_ms=latency_ms,
                    schema_valid=schema_valid,
                    validator_reason=validator_reason,
                    trace_id=trace_id,
                    system_trace=system_trace,
                    question_type=question_type,
                    error=error,
                    extra_perf=perf_extra,
                )
            )

    summary = _summarize(results)
    report = {
        "run_id": run_id,
        "system_url": config.system_url,
        "num_questions": len(config.questions),
        "summary": summary,
        "results": [r.__dict__ for r in results],
    }
    (RESULTS_DIR / f"{run_id}.json").write_text(json.dumps(report, indent=2, default=str))
    return report


def _summarize(results: List[QuestionResult]) -> dict:
    n = len(results)
    n_errors = sum(1 for r in results if r.error)
    n_valid_schema = sum(1 for r in results if r.schema_valid is True)
    n_checked_schema = sum(1 for r in results if r.schema_valid is not None)

    retrieval_recalls = [r.retrieval["recall_at_k"] for r in results if r.retrieval]
    retrieval_precisions = [r.retrieval["precision_at_k"] for r in results if r.retrieval]
    retrieval_hit_rates = [r.retrieval["hit_rate"] for r in results if r.retrieval]
    reciprocal_ranks = [r.retrieval["reciprocal_rank"] for r in results if r.retrieval]

    perf_llm_calls = [r.extra_perf.get("llm_calls") for r in results if "llm_calls" in r.extra_perf]
    perf_tokens = [r.extra_perf.get("tokens_used") for r in results if "tokens_used" in r.extra_perf]
    perf_cost = [r.extra_perf.get("cost_usd") for r in results if "cost_usd" in r.extra_perf]
    perf_retries = [r.extra_perf.get("retries_used") for r in results if "retries_used" in r.extra_perf]

    failed = [r for r in results if r.is_failure]
    failed_examples = [
        {
            "question_id": r.question_id,
            "trace_id": r.trace_id,
            "trace_url": tracing.get_trace_url(r.trace_id) if r.trace_id else None,
            "reason": r.error or r.validator_reason or "exact_match miss",
        }
        for r in failed[:5]
    ]

    return {
        "num_questions": n,
        "num_errors": n_errors,
        "error_rate": n_errors / n if n else None,
        "exact_match": m.mean([r.exact_match for r in results]),
        "f1": m.mean([r.f1 for r in results]),
        "numerical_accuracy": m.mean([r.numerical_accuracy for r in results]),
        "schema_validity_rate": (n_valid_schema / n_checked_schema) if n_checked_schema else None,
        "retrieval_recall_at_k": m.mean(retrieval_recalls),
        "retrieval_precision_at_k": m.mean(retrieval_precisions),
        "retrieval_hit_rate": m.mean(retrieval_hit_rates),
        "retrieval_mrr": m.mean(reciprocal_ranks),
        "avg_latency_ms": m.mean([r.latency_ms for r in results]),
        "avg_llm_calls": m.mean(perf_llm_calls),
        "avg_tokens_used": m.mean(perf_tokens),
        "avg_cost_usd": m.mean(perf_cost),
        "avg_retries_used": m.mean(perf_retries),
        "num_failed_examples": len(failed),
        "failed_examples": failed_examples,
    }


def load_report(run_id: str) -> Optional[dict]:
    path = RESULTS_DIR / f"{run_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())
