"""Direct, corpus-wide retrieval evaluation and reranking ablations."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import httpx

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
KS = (1, 5, 10)


class HTTPClient(Protocol):
    def post(self, url: str, json: dict[str, Any]): ...


@dataclass(frozen=True)
class RetrievalVariant:
    name: str
    path: str
    rerank: bool


VARIANTS = (
    RetrievalVariant("dense", "/search/vector", False),
    RetrievalVariant("bm25", "/search/bm25", False),
    RetrievalVariant("hybrid", "/search", False),
    RetrievalVariant("hybrid_reranker", "/search", True),
)


@dataclass
class RetrievalBenchmarkConfig:
    retrieval_url: str
    questions: list[dict]
    variant: RetrievalVariant = VARIANTS[-1]
    candidate_k: int = 30
    top_k: int = 10
    timeout_seconds: float = 60.0
    question_field: str = "question_text"
    relevance_unit: Literal["document", "document_page"] = "document"
    # Off by default: gold-document scoping is diagnostic only.
    scope_to_gold_document: bool = False
    output_dir: Path = field(default_factory=lambda: RESULTS_DIR)
    save_results: bool = True


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _gold_identity(evidence: dict, unit: str) -> str | None:
    identity = evidence.get("source_doc_uid") or evidence.get("source_document")
    if not identity:
        return None
    if unit == "document_page":
        page = evidence.get("source_page")
        return f"{identity}::page:{page}" if page is not None else None
    return str(identity)


def _hit_identity(hit: dict, unit: str) -> str | None:
    metadata = hit.get("metadata") if isinstance(hit.get("metadata"), dict) else {}
    identity = (
        hit.get("source_doc_uid")
        or metadata.get("source_doc_uid")
        or hit.get("document_id")
        or hit.get("filename")
        or hit.get("source_filename")
    )
    if not identity:
        return None
    if unit == "document_page":
        page = hit.get("page")
        return f"{identity}::page:{page}" if page is not None else None
    return str(identity)


def _ordered_unique(values: list[str | None]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _metrics(retrieved: list[str], relevant: set[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    for k in KS:
        top = retrieved[:k]
        hits = len(set(top) & relevant)
        result[f"recall_at_{k}"] = hits / len(relevant)
        result[f"precision_at_{k}"] = hits / len(top) if top else 0.0
        result[f"hit_rate_at_{k}"] = 1.0 if hits else 0.0
    result["reciprocal_rank"] = next(
        (1.0 / rank for rank, item in enumerate(retrieved, 1) if item in relevant),
        0.0,
    )
    return result


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _summarize(rows: list[dict]) -> dict[str, Any]:
    scored = [row for row in rows if row.get("metrics")]
    latencies = [float(row["latency_ms"]) for row in rows if row.get("error") is None]
    summary: dict[str, Any] = {
        "queries": len(rows),
        "scored_queries": len(scored),
        "errors": sum(row.get("error") is not None for row in rows),
    }
    for metric in (
        *(f"recall_at_{k}" for k in KS),
        *(f"precision_at_{k}" for k in KS),
        *(f"hit_rate_at_{k}" for k in KS),
        "reciprocal_rank",
    ):
        values = [float(row["metrics"][metric]) for row in scored]
        summary["mrr" if metric == "reciprocal_rank" else metric] = _mean(values)
    summary.update(
        {
            "mean_latency_ms": _mean(latencies),
            "median_latency_ms": statistics.median(latencies) if latencies else None,
            "p95_latency_ms": _percentile(latencies, 0.95),
        }
    )
    return summary


def _write_report(report: dict, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"retrieval-{report['variant']}-{report['run_id']}"
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    fieldnames = [
        "question_id",
        "question",
        "latency_ms",
        "retrieved",
        "relevant",
        "error",
        *(f"recall_at_{k}" for k in KS),
        *(f"precision_at_{k}" for k in KS),
        *(f"hit_rate_at_{k}" for k in KS),
        "reciprocal_rank",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in report["results"]:
            writer.writerow(
                {
                    **{key: row.get(key) for key in fieldnames},
                    **(row.get("metrics") or {}),
                    "retrieved": json.dumps(row.get("retrieved", [])),
                    "relevant": json.dumps(row.get("relevant", [])),
                }
            )
    return json_path, csv_path


def run_retrieval_benchmark(
    config: RetrievalBenchmarkConfig, client: HTTPClient | None = None
) -> dict[str, Any]:
    """Evaluate ranked retrieval directly; no agent answer citations are used."""
    owns_client = client is None
    actual_client = client or httpx.Client(timeout=config.timeout_seconds)
    rows: list[dict[str, Any]] = []
    try:
        for index, question in enumerate(config.questions):
            query = str(question.get(config.question_field, "")).strip()
            relevant = {
                identity
                for evidence in question.get("gold_evidence", []) or []
                if (identity := _gold_identity(evidence, config.relevance_unit))
            }
            payload: dict[str, Any] = {
                "query": query,
                "top_k": config.top_k,
                "candidate_k": config.candidate_k,
                "rerank": config.variant.rerank,
            }
            if config.scope_to_gold_document and len(relevant) == 1:
                payload["document_id"] = next(iter(relevant)).split("::page:")[0]
            started = time.perf_counter()
            error = None
            retrieved: list[str] = []
            try:
                response = actual_client.post(
                    f"{config.retrieval_url.rstrip('/')}{config.variant.path}",
                    json=payload,
                )
                response.raise_for_status()
                body = response.json()
                retrieved = _ordered_unique(
                    [
                        _hit_identity(hit, config.relevance_unit)
                        for hit in body.get("hits", [])
                        if isinstance(hit, dict)
                    ]
                )
            except Exception as exc:  # retained per-query in the machine-readable report
                error = f"{type(exc).__name__}: {exc}"
            latency_ms = (time.perf_counter() - started) * 1000
            rows.append(
                {
                    "question_id": question.get("question_id", str(index)),
                    "question": query,
                    "latency_ms": latency_ms,
                    "retrieved": retrieved,
                    "relevant": sorted(relevant),
                    "metrics": _metrics(retrieved, relevant) if relevant and not error else None,
                    "error": error,
                    "request_document_id": payload.get("document_id"),
                }
            )
    finally:
        if owns_client:
            actual_client.close()

    report: dict[str, Any] = {
        "run_id": str(uuid.uuid4()),
        "variant": config.variant.name,
        "retrieval_url": config.retrieval_url,
        "scope_to_gold_document": config.scope_to_gold_document,
        "relevance_unit": config.relevance_unit,
        "summary": _summarize(rows),
        "results": rows,
    }
    if config.save_results:
        json_path, csv_path = _write_report(report, config.output_dir)
        report["json_path"] = str(json_path)
        report["csv_path"] = str(csv_path)
    return report


def run_ablation(
    retrieval_url: str,
    questions: list[dict],
    *,
    candidate_k: int = 30,
    top_k: int = 10,
    output_dir: Path = RESULTS_DIR,
    client: HTTPClient | None = None,
    save_results: bool = True,
) -> dict[str, Any]:
    reports = {}
    for variant in VARIANTS:
        reports[variant.name] = run_retrieval_benchmark(
            RetrievalBenchmarkConfig(
                retrieval_url=retrieval_url,
                questions=questions,
                variant=variant,
                candidate_k=candidate_k,
                top_k=top_k,
                output_dir=output_dir,
                save_results=save_results,
                scope_to_gold_document=False,
            ),
            client=client,
        )
    hybrid = reports["hybrid"]["summary"]
    reranked = reports["hybrid_reranker"]["summary"]
    delta_fields = ("recall_at_5", "recall_at_10", "mrr", "mean_latency_ms", "p95_latency_ms")
    deltas = {
        field: (
            reranked[field] - hybrid[field]
            if reranked.get(field) is not None and hybrid.get(field) is not None
            else None
        )
        for field in delta_fields
    }
    comparison = {
        "run_id": str(uuid.uuid4()),
        "scope_to_gold_document": False,
        "variants": {name: report["summary"] for name, report in reports.items()},
        "hybrid_reranker_minus_hybrid": deltas,
        "reports": reports,
    }
    if save_results:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"retrieval-ablation-{comparison['run_id']}.json"
        path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
        comparison["comparison_path"] = str(path)
    return comparison


def format_comparison(comparison: dict[str, Any]) -> str:
    columns = (
        "recall_at_1",
        "recall_at_5",
        "recall_at_10",
        "precision_at_1",
        "precision_at_5",
        "precision_at_10",
        "hit_rate_at_1",
        "hit_rate_at_5",
        "hit_rate_at_10",
        "mrr",
        "mean_latency_ms",
        "p95_latency_ms",
    )
    header = ["variant", *columns]
    lines = [" | ".join(header), " | ".join("---" for _ in header)]
    for name, summary in comparison["variants"].items():
        values = [name]
        for column in columns:
            value = summary.get(column)
            values.append("n/a" if value is None else f"{value:.4f}")
        lines.append(" | ".join(values))
    return "\n".join(lines)


def _load_questions(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("questions file must contain a JSON array or JSONL records")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-url", default="http://localhost:8002")
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--variant", choices=[variant.name for variant in VARIANTS])
    args = parser.parse_args()
    questions = _load_questions(args.questions)
    if args.variant:
        variant = next(item for item in VARIANTS if item.name == args.variant)
        report = run_retrieval_benchmark(
            RetrievalBenchmarkConfig(
                retrieval_url=args.retrieval_url,
                questions=questions,
                variant=variant,
                candidate_k=args.candidate_k,
                top_k=args.top_k,
                output_dir=args.output_dir,
            )
        )
        print(json.dumps(report["summary"], indent=2))
    else:
        comparison = run_ablation(
            args.retrieval_url,
            questions,
            candidate_k=args.candidate_k,
            top_k=args.top_k,
            output_dir=args.output_dir,
        )
        print(format_comparison(comparison))


if __name__ == "__main__":
    main()
