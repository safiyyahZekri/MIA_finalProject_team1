"""Run reproducible corpus-wide fusion/reranker sweeps on development queries."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.retrieval_benchmark import (
    RetrievalBenchmarkConfig,
    RetrievalVariant,
    _load_questions,
    run_retrieval_benchmark,
)


def sweep(
    url: str,
    questions: list[dict],
    output: Path,
    weights: list[float],
    constants: list[int],
    candidates: list[int],
    reranking: list[bool],
    client=None,
) -> dict:
    combinations = list(itertools.product(weights, constants, candidates, reranking))
    if not combinations or len(combinations) > 48:
        raise ValueError("Use between 1 and 48 configurations")
    if (
        any(not 0 <= w <= 1 for w in weights)
        or any(not 1 <= k <= 200 for k in constants)
        or any(not 10 <= k <= 250 for k in candidates)
    ):
        raise ValueError("Invalid weight/RRF constant/candidate count")
    reports = []
    for weight, constant, count, rerank in combinations:
        name = f"hybrid-w{weight}-rrf{constant}-c{count}-rerank{int(rerank)}"
        report = run_retrieval_benchmark(
            RetrievalBenchmarkConfig(
                retrieval_url=url,
                questions=questions,
                variant=RetrievalVariant(name, "/search", rerank),
                candidate_k=count,
                dense_weight=weight,
                rrf_k=constant,
                verify_search_settings=True,
                output_dir=output,
                scope_to_gold_document=False,
            ),
            client=client,
        )
        reports.append(report)
        print(name, json.dumps(report["summary"]))
    eligible = [
        r
        for r in reports
        if r["summary"]["errors"] == 0 and r["summary"]["scored_queries"] > 0
    ]
    best = max(
        eligible,
        key=lambda r: (
            r["summary"]["mrr"],
            r["summary"]["recall_at_5"],
            -r["summary"]["mean_latency_ms"],
        ),
        default=None,
    )
    comparison = {
        "scope": "corpus-wide",
        "reports": reports,
        "development_winner": best["variant"] if best else None,
        "note": "Select on development queries; evaluate the chosen configuration on held-out queries. Keep corpus fixed throughout.",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "tuning-comparison.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )
    lines = [
        "| Configuration | Recall@5 | Recall@10 | MRR | Mean ms | p95 ms | Errors |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in reports:
        s = r["summary"]
        lines.append(
            "| "
            + " | ".join(
                str(x)
                for x in [
                    r["variant"],
                    s["recall_at_5"],
                    s["recall_at_10"],
                    s["mrr"],
                    s["mean_latency_ms"],
                    s["p95_latency_ms"],
                    s["errors"],
                ]
            )
            + " |"
        )
    (output / "tuning-comparison.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-url", required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", type=float, nargs="+", default=[0.35, 0.55])
    parser.add_argument("--rrf-constants", type=int, nargs="+", default=[60])
    parser.add_argument("--candidates", type=int, nargs="+", default=[30, 60])
    args = parser.parse_args()
    with httpx.Client(timeout=120) as client:
        comparison = sweep(
            args.retrieval_url,
            _load_questions(args.questions),
            args.output,
            args.weights,
            args.rrf_constants,
            args.candidates,
            [False, True],
            client,
        )
    if any(r["summary"]["errors"] for r in comparison["reports"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
