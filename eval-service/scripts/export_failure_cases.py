#!/usr/bin/env python3
"""Extract the per-question retrieval outcomes from an ablation run.

The ablation writes its full report under results/, which is gitignored and
large. Failure analysis only needs, per question, what was asked, which
documents were gold, and what each variant actually retrieved -- so emit
exactly that, small enough to commit and to work through without access to
the machine hosting the index.

Every uid here is a corpus filename stem, so a case can be inspected by
opening `<uid>.pdf` from the extracted corpus.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TOP_N = 10


def export(comparison_path: Path, top_n: int = TOP_N) -> dict:
    report = json.loads(comparison_path.read_text(encoding="utf-8"))
    reports = report.get("reports") or {}

    per_question: dict[str, dict] = {}
    for variant, payload in reports.items():
        for row in payload.get("results") or []:
            qid = str(row.get("question_id"))
            entry = per_question.setdefault(
                qid,
                {
                    "question_id": qid,
                    "question": row.get("question"),
                    "gold": list(row.get("relevant") or []),
                    "variants": {},
                },
            )
            metrics = row.get("metrics") or {}
            retrieved = list(row.get("retrieved") or [])[:top_n]
            gold = set(entry["gold"])
            entry["variants"][variant] = {
                "retrieved": retrieved,
                "hit_at_1": bool(gold & set(retrieved[:1])),
                "hit_at_5": bool(gold & set(retrieved[:5])),
                "hit_at_10": bool(gold & set(retrieved)),
                "recall_at_10": metrics.get("recall_at_10"),
                "latency_ms": row.get("latency_ms"),
            }

    questions = sorted(per_question.values(), key=lambda q: q["question_id"])
    variants = sorted(reports)

    # Questions no variant retrieved are the most informative failures: the
    # document is in the index, so nothing about ranking explains the miss.
    missed_by_all = [
        q["question_id"]
        for q in questions
        if q["gold"] and all(not q["variants"].get(v, {}).get("hit_at_10") for v in variants)
    ]
    # Cases where lexical and dense disagree say what each is actually good at.
    bm25_only = [
        q["question_id"]
        for q in questions
        if q["variants"].get("bm25", {}).get("hit_at_10")
        and not q["variants"].get("dense", {}).get("hit_at_10")
    ]
    dense_only = [
        q["question_id"]
        for q in questions
        if q["variants"].get("dense", {}).get("hit_at_10")
        and not q["variants"].get("bm25", {}).get("hit_at_10")
    ]
    # The reranker demoting a document out of the top 5 that hybrid had there.
    reranker_regressions = [
        q["question_id"]
        for q in questions
        if q["variants"].get("hybrid", {}).get("hit_at_5")
        and not q["variants"].get("hybrid_reranker", {}).get("hit_at_5")
    ]

    return {
        "run_id": report.get("run_id"),
        "variants": variants,
        "question_count": len(questions),
        "buckets": {
            "missed_by_all_variants": missed_by_all,
            "bm25_hit_dense_miss": bm25_only,
            "dense_hit_bm25_miss": dense_only,
            "reranker_dropped_from_top5": reranker_regressions,
        },
        "questions": questions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "failure_cases.json",
    )
    parser.add_argument("--top-n", type=int, default=TOP_N)
    args = parser.parse_args()

    data = export(args.comparison, args.top_n)
    args.output.write_text(json.dumps(data, indent=2), encoding="utf-8")

    b = data["buckets"]
    print(f"run_id                     : {data['run_id']}")
    print(f"questions                  : {data['question_count']}")
    print(f"missed by every variant    : {len(b['missed_by_all_variants'])}")
    print(f"bm25 hit / dense miss      : {len(b['bm25_hit_dense_miss'])}")
    print(f"dense hit / bm25 miss      : {len(b['dense_hit_bm25_miss'])}")
    print(f"reranker dropped from top5 : {len(b['reranker_dropped_from_top5'])}")
    print(f"written to                 : {args.output}")


if __name__ == "__main__":
    main()
