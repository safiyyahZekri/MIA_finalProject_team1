"""Compare one isolated agent optimization off/on without service calls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.retrieval_benchmark import IDENTITY_ALIASES_PATH  # noqa: E402
from scripts.audit_answer_eval import audit_report  # noqa: E402

FEATURES = (
    "query_decomposition",
    "hybrid_reranking",
    "answer_shape_guidance",
    "retry_evidence_fusion",
    "entity_document_routing",
    "adaptive_top_k",
    "evidence_diversification",
    "query_expansion",
    "table_query_rewriting",
    "adjacent_evidence",
    "answer_repair",
    "answer_normalization",
    "question_type_retrieval",
    "grade_calibration",
)


def load_run(directory: Path, enabled: bool, feature: str = FEATURES[0]) -> dict:
    if feature not in FEATURES:
        raise ValueError(f"Unknown optimization flag: {feature}")
    info = json.loads((directory / "run_info.json").read_text(encoding="utf-8"))
    configs = []
    for snapshot in [info, *info.get("resumed", [])]:
        agent = snapshot["agent"]
        if agent.get("llm_provider") == "mock":
            raise ValueError("Mock-model runs cannot establish an EM/F1 improvement")
        config = dict(agent["config"])
        actual = config.pop(feature, False)
        if actual is not enabled:
            raise ValueError(f"{directory}: expected {feature}={enabled}")
        configs.append({"provider": agent.get("llm_provider"), **config})
    if any(c != configs[0] for c in configs):
        raise ValueError(f"{directory}: settings changed during a resumed run")
    return {
        "config": configs[0],
        "report": json.loads((directory / "combined.json").read_text(encoding="utf-8")),
    }


def compare_runs(
    baseline: dict,
    candidate: dict,
    questions: list[dict],
    aliases: dict,
    optimization: str = FEATURES[0],
) -> dict:
    if baseline["config"] != candidate["config"]:
        raise ValueError(
            f"Agent settings differ beyond {optimization}; not an isolated comparison"
        )
    sets = [
        {r["question_id"] for r in run["report"]["results"]}
        for run in (baseline, candidate)
    ]
    if not sets[0] or sets[0] != sets[1]:
        raise ValueError("Both runs must contain the same non-empty question set")
    audits = [
        audit_report(run["report"], questions, aliases) for run in (baseline, candidate)
    ]

    def metrics(run, audit):
        rows = run["report"]["results"]
        summary = audit["summary"]

        def mean(values):
            return (
                sum(values) / len(values)
                if values and all(v is not None for v in values)
                else None
            )

        return {
            # Count errors as zero on both sides so failed calls cannot improve EM.
            "exact_match_all": sum(r["exact_match"] or 0 for r in audit["results"])
            / len(rows),
            "f1_all": sum(r["f1"] or 0 for r in audit["results"]) / len(rows),
            "numerical_accuracy_strict": summary["answer_quality"][
                "numerical_accuracy_strict"
            ],
            "answerable_abstentions": summary["answerable_abstentions"],
            "errors": summary["errors"],
            "retrieval_recall_at_5": summary["agent_retrieval"]["final_attempt"][
                "recall_at_k"
            ],
            "retrieval_mrr_at_5": summary["agent_retrieval"]["final_attempt"][
                "reciprocal_rank"
            ],
            "mean_latency_ms": mean([r.get("latency_ms") for r in rows]),
            "mean_cost_usd": mean(
                [(r.get("extra_perf") or {}).get("cost_usd") for r in rows]
            ),
        }

    before, after = [
        metrics(run, audit) for run, audit in zip((baseline, candidate), audits)
    ]
    return {
        "questions": len(sets[0]),
        "optimization": optimization,
        "config": baseline["config"],
        "corpus_note": "Keep the shared index unchanged between runs; these agent snapshots do not fingerprint the corpus.",
        "baseline": before,
        "candidate": after,
        "delta": {
            key: after[key] - before[key]
            if after[key] is not None and before[key] is not None
            else None
            for key in before
        },
        "baseline_audit": audits[0],
        "candidate_audit": audits[1],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--optimization", choices=FEATURES, default=FEATURES[0])
    parser.add_argument(
        "--questions", type=Path, default=ROOT / "questions_setA_practice.json"
    )
    parser.add_argument("--identity-aliases", type=Path, default=IDENTITY_ALIASES_PATH)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protected = [args.questions, args.identity_aliases]
    protected += [
        p / name
        for p in (args.baseline, args.candidate)
        for name in ("combined.json", "run_info.json")
    ]
    if args.output.resolve() in {p.resolve() for p in protected}:
        raise ValueError("Comparison output must not overwrite its inputs")
    report = compare_runs(
        load_run(args.baseline, False, args.optimization),
        load_run(args.candidate, True, args.optimization),
        json.loads(args.questions.read_text(encoding="utf-8")),
        json.loads(args.identity_aliases.read_text(encoding="utf-8")),
        args.optimization,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"metric | baseline | {args.optimization} | delta")
    for key in report["baseline"]:
        values = [report[part][key] for part in ("baseline", "candidate", "delta")]
        print(
            key
            + " | "
            + " | ".join("unavailable" if v is None else f"{v:.4f}" for v in values)
        )
    print(report["corpus_note"])


if __name__ == "__main__":
    main()
