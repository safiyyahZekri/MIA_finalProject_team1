"""Measure presentation normalization on saved answers with the existing scorer.

This replays only deterministic output normalization, not retrieval or the LLM.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.retrieval_benchmark import load_identity_aliases
from scripts.audit_answer_eval import audit_report


def replay(report: dict, questions: list[dict], aliases: dict) -> dict:
    path = ROOT.parent / "agent-service/app/answer_normalization.py"
    spec = importlib.util.spec_from_file_location("ledger_answer_normalization", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    candidate = copy.deepcopy(report)
    changed = []
    for row in candidate["results"]:
        if row.get("predicted_answer"):
            before = row["predicted_answer"]
            row["predicted_answer"] = module.normalize_answer(before)
            if before != row["predicted_answer"]:
                changed.append(row["question_id"])
    before = audit_report(report, questions, aliases)["summary"]["answer_quality"]
    after = audit_report(candidate, questions, aliases)["summary"]["answer_quality"]
    return {
        "kind": "offline_normalization_replay",
        "questions": len(report["results"]),
        "changed_answers": changed,
        "baseline": before,
        "candidate": after,
        "delta": {
            key: after[key] - before[key]
            for key in ("exact_match", "f1")
            if before[key] is not None and after[key] is not None
        },
        "note": "Same saved answers and unchanged scorer. No new retrieval/model evaluation.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--questions", type=Path, default=ROOT / "questions_setA_practice.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() in {args.report.resolve(), args.questions.resolve()}:
        raise ValueError("Output cannot overwrite original inputs")
    result = replay(
        json.loads(args.report.read_text()),
        json.loads(args.questions.read_text()),
        load_identity_aliases(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
