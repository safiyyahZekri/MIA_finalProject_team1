#!/usr/bin/env python3
"""Split answer-level failures by the stage that actually broke.

A wrong answer has two very different causes, and the fix differs
completely between them:

  RETRIEVAL_MISS      the evidence never reached the model -- no prompt or
                      model change helps, the document has to be found
  GENERATION_ERROR    the right document was retrieved and the answer is
                      still wrong -- reading the wrong row, botching the
                      arithmetic, losing the scale
  SCHEMA_INVALID      the answer did not satisfy the response contract
  SYSTEM_ERROR        the request itself failed

Reading only the retrieval ablation cannot tell these apart, and reading
only the answer scores cannot either. Joining them can, which is what
makes the failure analysis cover the whole system rather than half of it.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def _gold_retrieved(row: dict) -> bool | None:
    """Did the evidence the model cited include a gold document?"""
    retrieval = row.get("retrieval")
    if not isinstance(retrieval, dict):
        return None
    for key in ("hit_rate", "hit_rate_at_k", "recall_at_k", "recall"):
        value = retrieval.get(key)
        if isinstance(value, (int, float)):
            return value > 0
    return None


def _answer_correct(row: dict) -> bool:
    """Numeric accuracy is authoritative when present: most TAT-DQA answers
    are figures, where exact string match fails on formatting alone."""
    for key in ("numerical_accuracy", "exact_match"):
        value = row.get(key)
        if isinstance(value, (int, float)):
            return value >= 1.0
    return False


def classify(row: dict) -> str:
    if row.get("error"):
        return "SYSTEM_ERROR"
    if row.get("schema_valid") is False:
        return "SCHEMA_INVALID"
    if _answer_correct(row):
        return "CORRECT"
    retrieved = _gold_retrieved(row)
    if retrieved is False:
        return "RETRIEVAL_MISS"
    if retrieved is True:
        return "GENERATION_ERROR"
    return "UNCLASSIFIED"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True, help="results/<run_id>.json")
    parser.add_argument(
        "--retrieval-cases",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "failure_cases.json",
        help="retrieval-stage export, used to cross-check the retrieval verdict",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "failure_cases_answer_level.json",
    )
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    rows = report.get("results") or []

    retrieval_lookup: dict[str, dict] = {}
    if args.retrieval_cases.exists():
        cases = json.loads(args.retrieval_cases.read_text(encoding="utf-8"))
        retrieval_lookup = {q["question_id"]: q for q in cases.get("questions", [])}

    classified = []
    for row in rows:
        qid = str(row.get("question_id"))
        verdict = classify(row)
        baseline = retrieval_lookup.get(qid, {})
        classified.append(
            {
                "question_id": qid,
                "verdict": verdict,
                "question": row.get("question") or baseline.get("question"),
                "gold": baseline.get("gold"),
                "exact_match": row.get("exact_match"),
                "numerical_accuracy": row.get("numerical_accuracy"),
                "f1": row.get("f1"),
                "schema_valid": row.get("schema_valid"),
                "error": row.get("error"),
                "trace_id": row.get("trace_id"),
                # What the retrieval-only baseline said, so a disagreement
                # between the two layers is visible rather than hidden.
                "baseline_bm25_hit_at_10": (baseline.get("variants", {}).get("bm25") or {}).get("hit_at_10"),
            }
        )

    counts = Counter(c["verdict"] for c in classified)
    buckets: dict[str, list[str]] = {}
    for c in classified:
        buckets.setdefault(c["verdict"], []).append(c["question_id"])

    out = {
        "run_id": report.get("run_id"),
        "question_count": len(classified),
        "counts": dict(counts),
        "buckets": buckets,
        "questions": classified,
    }
    args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"run_id     : {out['run_id']}")
    print(f"questions  : {out['question_count']}")
    for verdict, n in counts.most_common():
        print(f"  {verdict:<18} {n}")
    print(f"written to : {args.output}")
    if counts.get("UNCLASSIFIED"):
        print(
            "\nUNCLASSIFIED means the report carried no retrieval metrics for "
            "those questions -- usually gold evidence was absent, so the stage "
            "cannot be attributed either way."
        )


if __name__ == "__main__":
    main()
