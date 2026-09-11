#!/usr/bin/env python3
"""Audit saved answers and retrieval traces offline, without changing the run.

No OCR, indexing, network requests, or LLM calls. Uses the same evidence
scorer as new benchmark runs. Missing traces are reported as unavailable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import metrics as m  # noqa: E402
from app.benchmark import _extract_answer_value, _extract_relevant_doc_ids  # noqa: E402
from app.evidence_scoring import (  # noqa: E402
    METRIC_DEFINITIONS,
    score_evidence,
    summarize_evidence,
)
from app.retrieval_benchmark import IDENTITY_ALIASES_PATH  # noqa: E402


def audit_report(
    report: dict, questions: list[dict], aliases: dict[str, str], k: int = 5
) -> dict:
    if k < 1:
        raise ValueError("k must be positive")
    by_id = {q["question_id"]: q for q in questions}
    rows = []
    seen = set()
    for source in report["results"]:
        qid = source["question_id"]
        if qid in seen:
            raise ValueError(f"Duplicate result question_id: {qid}")
        seen.add(qid)
        question = by_id[qid]
        if source.get("gold") != question.get("ground_truth_answer"):
            raise ValueError(f"Gold differs from the supplied question set: {qid}")
        if source.get("question") != question.get("question_text"):
            raise ValueError(
                f"Question text differs from the supplied question set: {qid}"
            )
        answer = source.get("predicted_answer")
        relevant = set(_extract_relevant_doc_ids(question, aliases))
        scored = score_evidence(
            answer, source.get("system_trace"), relevant, k, aliases
        )
        row = {"question_id": qid, "relevant_document_ids": sorted(relevant), **scored}
        gold = question.get("ground_truth_answer")
        abstained = (answer or {}).get("answer_type") == "insufficient_evidence"
        unanswerable = question.get("answer_type") == "unanswerable" or gold is None
        value = _extract_answer_value(answer)
        em = f1 = num = None
        if answer is not None:
            if unanswerable:
                em = f1 = float(abstained)
            else:
                em = m.exact_match(value, gold)
                f1 = m.f1_score(value, gold)
                num = m.numerical_accuracy(
                    value, gold, gold_scale=question.get("scale") or ""
                )
        row.update(
            {
                "exact_match": em,
                "f1": f1,
                "numerical_accuracy": num,
                "answerable_abstention": abstained and not unanswerable,
                "numeric_gold": question.get("answer_type") in ("arithmetic", "count"),
                "saved_citation_hit_rate": (source.get("retrieval") or {}).get(
                    "hit_rate"
                ),
                "error": source.get("error"),
            }
        )
        rows.append(row)

    summary = summarize_evidence(rows)
    numeric = [r for r in rows if r["numeric_gold"]]
    abstained = [r for r in rows if r["answerable_abstention"]]
    found_then_declined = [
        r["question_id"]
        for r in abstained
        if r["agent_retrieval"] and r["agent_retrieval"]["any_attempt_hit_rate"] == 1
    ]
    never_found = [
        r["question_id"]
        for r in abstained
        if r["agent_retrieval"] and r["agent_retrieval"]["any_attempt_hit_rate"] == 0
    ]
    citation_changes = [
        r["question_id"]
        for r in rows
        if r["citation"] is not None
        and r["saved_citation_hit_rate"] is not None
        and r["saved_citation_hit_rate"] != r["citation"]["hit_rate"]
    ]
    summary.update(
        {
            "questions": len(rows),
            "errors": sum(bool(r["error"]) for r in rows),
            "answer_quality": {
                "exact_match": m.mean([r["exact_match"] for r in rows]),
                "f1": m.mean([r["f1"] for r in rows]),
                "numerical_accuracy_comparable": m.mean(
                    [r["numerical_accuracy"] for r in rows]
                ),
                "numerical_accuracy_comparable_questions": sum(
                    r["numerical_accuracy"] is not None for r in rows
                ),
                "numerical_accuracy_strict": sum(
                    r["numerical_accuracy"] == 1 for r in numeric
                )
                / len(numeric)
                if numeric
                else None,
                "numeric_gold_questions": len(numeric),
            },
            "answerable_abstentions": len(abstained),
            "gold_document_found_but_abstained": found_then_declined,
            "gold_document_never_found_and_abstained": never_found,
            "saved_citation_hit_rate_changes": citation_changes,
        }
    )
    return {
        "metric_definitions": METRIC_DEFINITIONS,
        "summary": summary,
        "results": rows,
    }


def render_report(report: dict) -> str:
    summary = report["summary"]
    retrieval = summary["agent_retrieval"]
    quality = summary["answer_quality"]

    def fmt(value):
        return "unavailable" if value is None else f"{value:.4f}"

    lines = [
        "# Saved evaluation audit",
        "",
        "Offline recomputation. The original answers, scores and corpus are unchanged.",
        "",
        f"Questions: {summary['questions']}; errors: {summary['errors']}.",
        f"Exact Match: {fmt(quality['exact_match'])}; F1: {fmt(quality['f1'])}.",
        f"Numerical accuracy: {fmt(quality['numerical_accuracy_comparable'])} over "
        f"{quality['numerical_accuracy_comparable_questions']} comparable answers; "
        f"{fmt(quality['numerical_accuracy_strict'])} over all {quality['numeric_gold_questions']} numeric-gold questions.",
        "",
        "| Evidence source | Questions | Recall@K | Precision@K | Hit rate | MRR@K |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, scores in (
        ("Final answer citations", summary["citation"]),
        ("Initial agent retrieval", retrieval["first_attempt"]),
        ("Final agent retrieval", retrieval["final_attempt"]),
    ):
        cells = [
            fmt(scores[key])
            for key in ("recall_at_k", "precision_at_k", "hit_rate", "reciprocal_rank")
        ]
        lines.append(
            f"| {name} | {scores['scored_questions']} | " + " | ".join(cells) + " |"
        )
    lines.extend(
        [
            "",
            f"Gold-document hit in any attempt: {fmt(retrieval['any_attempt_hit_rate'])} "
            f"over {retrieval['any_attempt_scored_questions']} questions. This is retry coverage, not single-search Recall@K.",
            f"Unresolved hash IDs: {len(retrieval['unresolved_document_ids'])}.",
            f"Malformed trace questions: {retrieval['questions_with_trace_issues']}.",
            f"Saved citation hit-rate changes after identity resolution: {len(summary['saved_citation_hit_rate_changes'])}.",
            f"Answerable questions declined: {summary['answerable_abstentions']}.",
            f"Gold document found, then declined: {len(summary['gold_document_found_but_abstained'])} "
            f"({', '.join(summary['gold_document_found_but_abstained'])}).",
            f"Gold document never found, then declined: {len(summary['gold_document_never_found_and_abstained'])}.",
            "",
            "Precision uses K as the denominator; older reports divided by the number of returned citations.",
            "Document-level matches do not establish that the retrieved passage contains the answer or company attribution.",
            "Strict EM is unchanged. Paraphrases, rounding and omitted units are not automatically relabelled correct.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--questions", type=Path, default=ROOT / "questions_setA_practice.json"
    )
    parser.add_argument("--identity-aliases", type=Path, default=IDENTITY_ALIASES_PATH)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    # Fail visibly on a missing/malformed alias file; an empty map is allowed
    # explicitly, and unresolved hash IDs remain visible in the report.
    aliases = json.loads(args.identity_aliases.read_text(encoding="utf-8"))
    if not isinstance(aliases, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in aliases.items()
    ):
        raise ValueError(
            "Identity aliases must be a JSON object mapping strings to strings"
        )
    result = audit_report(
        json.loads(args.report.read_text(encoding="utf-8")),
        json.loads(args.questions.read_text(encoding="utf-8")),
        aliases,
        args.k,
    )
    result["provenance"] = {
        "k": args.k,
        **{
            name: {
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for name, path in (
                ("report", args.report),
                ("questions", args.questions),
                ("aliases", args.identity_aliases),
            )
        },
    }
    destinations = [args.output_dir / "audit.json", args.output_dir / "audit.md"]
    sources = {
        p.resolve() for p in (args.report, args.questions, args.identity_aliases)
    }
    if any(p.resolve() in sources for p in destinations):
        raise ValueError("Audit output must not overwrite an input file")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destinations[0].write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    text = render_report(result)
    destinations[1].write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
