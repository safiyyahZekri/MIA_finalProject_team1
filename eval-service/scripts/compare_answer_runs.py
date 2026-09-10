#!/usr/bin/env python3
"""Compare answer-level evaluation runs made on the same questions.

Prints headline metrics, an outcome for every question in every run, and
which answers changed. Exact match alone misreads experiments: it scores a
fact wrapped in labels as wrong, and a right number cited from documents that
do not contain it as right. The outcome labels separate those cases.

"Cited correctly" means a cited document contains the question's gold facts.
It is checked against gold_facts rather than the answer's own formula: a count
comes back as "1+1+1", and a "1" appears in any document.

Needs the index's chunks.jsonl -- from the index export, `tar -xzf
ledger-index.tar.gz ./chunks.jsonl` -- to read what cited documents contain.

Usage, from eval-service/:
  python scripts/compare_answer_runs.py path/to/chunks.jsonl \\
      baseline=results/answer-eval/combined.json \\
      format=results/answer-eval/experiments/A1-format/combined.json
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import metrics as m  # noqa: E402
from app.retrieval_benchmark import load_identity_aliases  # noqa: E402

OUTCOMES = "CUFAWE"
LEGEND = (
    "C correct, cited correctly   U correct, but no cited document holds the gold facts   "
    "F right content, wrong form   A abstained   W wrong   E error"
)
THOUSANDS = re.compile(r"(?<=\d),(?=\d)")
NUMBER = re.compile(r"\d+(?:\.\d+)?")


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _fmt(value) -> str:
    return "  -  " if value is None else f"{value:5.2f}"


class Judge:
    def __init__(self, chunks_path: Path) -> None:
        self.aliases = load_identity_aliases()
        self.questions = {
            q["question_id"]: q
            for q in json.loads((ROOT / "questions_setA_practice.json").read_text(encoding="utf-8"))
        }
        self.doc_text: dict[str, list[str]] = defaultdict(list)
        with chunks_path.open(encoding="utf-8") as fh:
            for line in fh:
                chunk = json.loads(line)
                self.doc_text[self._canon(chunk["document_id"])].append(chunk["text"])

    def _canon(self, document_id: str) -> str:
        return self.aliases.get(document_id, document_id)

    def cited_correctly(self, answer: dict, question: dict) -> bool:
        raw = " ".join(
            text
            for evidence in answer.get("evidence") or []
            for text in self.doc_text.get(self._canon(evidence["document_id"]), [])
        )
        plain, norm = THOUSANDS.sub("", raw), m._normalize_text(raw)
        checks = []
        for fact in (f for e in question.get("gold_evidence") or [] for f in (e.get("gold_facts") or [])):
            digits = THOUSANDS.sub("", str(fact)).strip("()$%-+ ")
            if NUMBER.fullmatch(digits):
                if len(digits.replace(".", "")) < 2:  # a bare divisor like "2" proves nothing
                    continue
                checks.append(bool(re.search(rf"(?<![\d.]){re.escape(digits)}(?![\d])", plain)))
            else:
                text = m._normalize_text(str(fact))
                if len(text) >= 3:
                    checks.append(text in norm)
        if checks:
            return all(checks)
        params = answer.get("params") or {}
        values = params.get("values") or [params.get("value")]
        return all(m._normalize_text(str(v)) and m._normalize_text(str(v)) in norm for v in values)

    def outcome(self, row: dict) -> str:
        question = self.questions[row["question_id"]]
        answer = row.get("predicted_answer") or {}
        if row.get("error"):
            return "E"
        abstained = answer.get("answer_type") == "insufficient_evidence"
        if question["answer_type"] == "unanswerable":
            return "C" if abstained else "W"
        if abstained:
            return "A"
        if row.get("numerical_accuracy") == 1.0 or row.get("exact_match") == 1.0:
            return "C" if self.cited_correctly(answer, question) else "U"
        params = answer.get("params") or {}
        predicted = m._normalize_text(" ".join(str(v) for v in (params.get("values") or [params.get("value")])))
        golds = row["gold"] if isinstance(row["gold"], list) else [row["gold"]]
        contained = bool(golds) and all(
            m._normalize_text(str(g)) and m._normalize_text(str(g)) in predicted for g in golds
        )
        # A span paraphrased by a word or two ("Because the Company..." for "as
        # the Company...") is the right content in the wrong form, not a miss.
        if contained or (row.get("f1") or 0) >= 0.8:
            return "F"
        return "W"


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    judge = Judge(Path(sys.argv[1]))
    runs = {}
    for spec in sys.argv[2:]:
        label, path = spec.split("=", 1)
        runs[label] = {r["question_id"]: r for r in json.loads(Path(path).read_text(encoding="utf-8"))["results"]}
    ids = sorted(set.intersection(*(set(rows) for rows in runs.values())))
    print(f"questions compared: {len(ids)}  {ids}\n")

    def citation_fallbacks(rows) -> str:
        """Answers cited from top-ranked hits because the model named no
        evidence. '-' for runs made before citations came from the model."""
        flags = [
            step["cited_by_model"]
            for row in rows
            for step in row.get("system_trace") or []
            if step.get("step") == "reason" and step.get("cited_by_model") is not None
        ]
        return str(sum(1 for flag in flags if flag is False)) if flags else "-"

    header = (
        f"{'run':<18}{'EM':>6}{'F1':>6}{'numAns':>7}{'numAll':>7}"
        + "".join(f"{k:>4}" for k in OUTCOMES)
        + f"{'fb':>4}{'cost$':>8}{'in_tok':>8}{'lat_s':>7}"
    )
    print(header)
    print("-" * len(header))
    outcomes = {}
    for label, rows in runs.items():
        selected = [rows[i] for i in ids]
        outcomes[label] = {i: judge.outcome(rows[i]) for i in ids}
        counts = {k: sum(1 for o in outcomes[label].values() if o == k) for k in OUTCOMES}
        numeric = [r for r in selected if judge.questions[r["question_id"]]["answer_type"] in ("arithmetic", "count")]
        perf = [r.get("extra_perf") or {} for r in selected]
        print(
            f"{label:<18}{_fmt(_mean([r.get('exact_match') for r in selected])):>6}"
            f"{_fmt(_mean([r.get('f1') for r in selected])):>6}"
            f"{_fmt(_mean([r.get('numerical_accuracy') for r in selected])):>7}"
            f"{_fmt(_mean([1.0 if r.get('numerical_accuracy') == 1.0 else 0.0 for r in numeric])):>7}"
            + "".join(f"{counts[k]:>4}" for k in OUTCOMES)
            + f"{citation_fallbacks(selected):>4}"
            + f"{sum(p.get('cost_usd') or 0 for p in perf):>8.3f}"
            f"{(_mean([p.get('input_tokens') for p in perf]) or 0):>8.0f}"
            f"{(_mean([r.get('latency_ms') for r in selected]) or 0) / 1000:>7.0f}"
        )
    print(
        "\nnumAns: numerical accuracy over questions that produced a number. "
        "numAll: over every numeric-gold question, abstentions counted as wrong."
    )
    print(LEGEND)

    print(f"\n{'qid':<6}{'gold type':<13}" + "".join(f"{label[:14]:>16}" for label in runs))
    for i in ids:
        print(f"{i:<6}{judge.questions[i]['answer_type']:<13}" + "".join(f"{outcomes[label][i]:>16}" for label in runs))

    first = next(iter(runs))
    print(f"\nanswers whose outcome differs from {first}:")
    for i in ids:
        for label in list(runs)[1:]:
            if outcomes[first][i] != outcomes[label][i]:
                before = runs[first][i].get("predicted_answer") or {}
                after = runs[label][i].get("predicted_answer") or {}
                shown = [
                    (a.get("params") or {}).get("values") or (a.get("params") or {}).get("value") or a.get("answer_type")
                    for a in (before, after)
                ]
                print(
                    f"  {i} {label}: {outcomes[first][i]}->{outcomes[label][i]}   "
                    f"{json.dumps(shown[0], ensure_ascii=False)[:60]}  ->  {json.dumps(shown[1], ensure_ascii=False)[:60]}"
                )


if __name__ == "__main__":
    main()
