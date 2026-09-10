#!/usr/bin/env python3
"""Run the answer-level evaluation in small, resumable batches.

Drives eval-service's /benchmark/run against agent-service one batch at a
time and saves each batch's full per-question report on the host. The
eval-service container keeps reports only in its own filesystem, so without
this they are lost on the next rebuild.

Built for a long run on a rate-limited API key:
  * batches are small and run back to back, so a failure costs one batch;
  * every batch report is written, and folded into the combined report,
    before anything is printed -- a crash cannot lose an answered question,
    and the next invocation folds any stray batch file back in;
  * --resume skips questions already answered in the output directory;
  * --retry-errors re-runs only the questions that errored;
  * --stop-after-errors ends the run after N errors in a row (default 3), so
    an exhausted API quota does not turn every remaining question into one;
  * --delay spaces questions out to stay under a per-minute token limit, and
    the summary prints measured input tokens per minute to size it.

Examples, from eval-service/:
  python scripts/run_answer_eval.py --sample 10
  python scripts/run_answer_eval.py --ids A001,A011
  python scripts/run_answer_eval.py --all --batch-size 5 --delay 15
  python scripts/run_answer_eval.py --all --resume --dry-run
  python scripts/run_answer_eval.py --retry-errors
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "questions_setA_practice.json"
DEFAULT_OUT = ROOT / "results" / "answer-eval"
TYPE_ORDER = ("arithmetic", "span", "multi-span", "count", "unanswerable")


def stratified_sample(questions: list[dict], n: int) -> list[dict]:
    """Round-robin across answer types, in file order, so even a tiny batch
    exercises every scoring path."""
    by_type: dict[str, list[dict]] = defaultdict(list)
    for q in questions:
        by_type[q["answer_type"]].append(q)
    queues = [list(by_type[t]) for t in TYPE_ORDER if by_type[t]]
    picked: list[dict] = []
    while len(picked) < n and any(queues):
        for queue in queues:
            if queue and len(picked) < n:
                picked.append(queue.pop(0))
    return picked


def load_saved_rows(out_dir: Path) -> dict[str, dict]:
    """Every result already on disk, keyed by question id.

    The combined report is rebuilt from the batch reports as well, so a batch
    written just before a crash is not lost. Batches are applied oldest first,
    so a re-run question replaces its earlier row.
    """
    rows: dict[str, dict] = {}
    combined = out_dir / "combined.json"
    if combined.exists():
        for row in json.loads(combined.read_text(encoding="utf-8"))["results"]:
            rows[row["question_id"]] = row
    for path in sorted(out_dir.glob("batch-*.json"), key=lambda p: p.stat().st_mtime):
        report = json.loads(path.read_text(encoding="utf-8"))
        for row in report["results"]:
            rows[row["question_id"]] = {**row, "run_id": report["run_id"]}
    return rows


def write_combined(path: Path, rows: dict[str, dict], questions: list[dict]) -> None:
    ordered = [rows[q["question_id"]] for q in questions if q["question_id"] in rows]
    path.write_text(json.dumps({"results": ordered}, indent=2, ensure_ascii=False), encoding="utf-8")


def select_questions(args, questions: list[dict], rows: dict[str, dict]) -> list[dict]:
    by_id = {q["question_id"]: q for q in questions}
    if args.sample:
        selected = stratified_sample(questions, args.sample)
    elif args.all:
        selected = list(questions)
    else:
        if args.retry_errors:
            ids = [qid for qid, row in rows.items() if row.get("error")]
            if not ids:
                raise SystemExit("no errored questions in the saved results")
        else:
            ids = [i.strip() for i in args.ids.split(",") if i.strip()]
        unknown = [i for i in ids if i not in by_id]
        if unknown:
            raise SystemExit(f"unknown question ids: {', '.join(unknown)}")
        selected = [by_id[i] for i in ids]
    if args.resume:
        selected = [
            q for q in selected if not (q["question_id"] in rows and not rows[q["question_id"]].get("error"))
        ]
    return selected


def run_batch(args, batch: list[dict]) -> dict:
    payload = {
        "system_url": args.system_url,
        "validator_url": args.validator_url,
        "questions": batch,
        "timeout_s": args.timeout,
        "delay_between_questions_s": args.delay,
    }
    # The eval call is synchronous: allow every question its full timeout.
    budget = len(batch) * (args.timeout + args.delay) + 120
    with httpx.Client(timeout=budget) as client:
        resp = client.post(f"{args.eval_url}/benchmark/run", json=payload)
        resp.raise_for_status()
        run_id = resp.json()["run_id"]
        report = client.get(f"{args.eval_url}/benchmark/results/{run_id}")
        report.raise_for_status()
        return report.json()


def _value(answer: dict | None):
    params = (answer or {}).get("params") or {}
    for key in ("values", "value", "reason"):
        if key in params:
            return params[key]
    return None


def _short(value, width: int) -> str:
    text = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 3] + "..."


def _fmt(score) -> str:
    return "  -  " if score is None else f"{score:5.2f}"


def print_rows(rows: list[dict], types: dict[str, str]) -> None:
    print(f"{'id':<6}{'type':<13}{'EM':>5} {'num':>5}  {'answer_type':<22}{'calls':>5}{'in_tok':>8}")
    for r in rows:
        perf = r.get("extra_perf") or {}
        answer = r.get("predicted_answer") or {}
        print(
            f"{r['question_id']:<6}{types.get(r['question_id'], '?'):<13}"
            f"{_fmt(r.get('exact_match'))} {_fmt(r.get('numerical_accuracy'))}  "
            f"{answer.get('answer_type', 'ERROR'):<22}"
            f"{str(perf.get('llm_calls', '-')):>5}{str(perf.get('input_tokens', '-')):>8}"
        )
        if r.get("error"):
            print(f"      error: {_short(r['error'], 110)}")
        else:
            print(f"      gold: {_short(r.get('gold'), 50):<52} pred: {_short(_value(answer), 50)}")


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def print_summary(rows: list[dict], types: dict[str, str], ran: list[dict], elapsed_s: float) -> None:
    answered = [r for r in rows if not r.get("error")]
    print()
    print(f"questions in report : {len(rows)}   answered: {len(answered)}   errors: {len(rows) - len(answered)}")
    print(f"exact match         : {_fmt(_mean([r.get('exact_match') for r in answered]))}  (answered only)")
    print(f"F1                  : {_fmt(_mean([r.get('f1') for r in answered]))}")
    print(f"numerical accuracy  : {_fmt(_mean([r.get('numerical_accuracy') for r in answered]))}  (numeric gold only)")
    print("by answer type      :")
    for t in TYPE_ORDER:
        of_type = [r for r in answered if types.get(r["question_id"]) == t]
        if of_type:
            em = _mean([r.get("exact_match") for r in of_type])
            num = _mean([r.get("numerical_accuracy") for r in of_type])
            print(f"   {t:<13} n={len(of_type):<3} EM {_fmt(em)}   num {_fmt(num)}")

    perfs = [r.get("extra_perf") or {} for r in answered]
    print(f"avg LLM calls       : {_fmt(_mean([p.get('llm_calls') for p in perfs]))}")
    print(f"avg input tokens    : {_mean([p.get('input_tokens') for p in perfs])}")
    print(f"avg output tokens   : {_mean([p.get('output_tokens') for p in perfs])}")
    costs = [p.get("cost_usd") for p in perfs if p.get("cost_usd") is not None]
    if costs:
        print(f"cost (answered)     : ${sum(costs):.4f} total, ${sum(costs) / len(costs):.5f} per question")
    print(f"avg latency         : {_mean([r.get('latency_ms') for r in answered]) or 0:.0f} ms")

    ran_input = sum((r.get("extra_perf") or {}).get("input_tokens") or 0 for r in ran)
    if elapsed_s > 0 and ran_input:
        print(f"this run            : {ran_input} input tokens in {elapsed_s / 60:.1f} min "
              f"= {ran_input / (elapsed_s / 60):.0f} input tokens/min")


def trailing_errors(results: list[dict]) -> int:
    """How many of the most recent results in a row are errors."""
    count = 0
    for row in reversed(results):
        if not row.get("error"):
            break
        count += 1
    return count


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        # Redirected output on Windows defaults to cp1252: one "≤" in a gold
        # answer raised UnicodeEncodeError and stopped a 100-question run.
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--sample", type=int, help="N questions, stratified across answer types")
    which.add_argument("--ids", help="comma-separated question ids, e.g. A001,A011")
    which.add_argument("--all", action="store_true", help="all 100 practice questions")
    which.add_argument("--retry-errors", action="store_true", help="re-run errored rows of the saved results")
    parser.add_argument("--resume", action="store_true", help="skip questions already answered in --out-dir")
    parser.add_argument("--dry-run", action="store_true", help="print what would run, call nothing")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--delay", type=float, default=0.0, help="seconds between questions")
    parser.add_argument("--timeout", type=float, default=300.0, help="seconds allowed per question")
    parser.add_argument("--eval-url", default="http://127.0.0.1:8005")
    # URLs as eval-service sees them from inside the compose network.
    parser.add_argument("--system-url", default="http://agent-service:8003/answer")
    parser.add_argument("--validator-url", default="http://answer-validator-api:8004/validate_answer")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--label", default="", help="name recorded with the run, e.g. an experiment id")
    parser.add_argument("--agent-url", default="http://127.0.0.1:8003", help="read for the config snapshot")
    parser.add_argument("--stop-after-errors", type=int, default=3,
                        help="stop after this many errors in a row; 0 never stops")
    args = parser.parse_args()

    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    types = {q["question_id"]: q["answer_type"] for q in questions}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    combined_path = args.out_dir / "combined.json"

    rows = load_saved_rows(args.out_dir)
    selected = select_questions(args, questions, rows)
    batches = [selected[i : i + args.batch_size] for i in range(0, len(selected), args.batch_size)]
    print(f"{len(rows)} results already saved; {len(selected)} questions to run in {len(batches)} batch(es); "
          f"delay {args.delay}s, timeout {args.timeout}s")
    if args.dry_run:
        print("would run:", ", ".join(q["question_id"] for q in selected) or "(nothing)")
        return
    # Fold any batch written before an interruption into the combined report now.
    write_combined(combined_path, rows, questions)

    # Record what actually ran, so a comparison between runs can't silently
    # mix configurations. A resumed run is appended, not overwritten.
    try:
        agent = httpx.get(f"{args.agent_url}/health", timeout=10).json()
    except httpx.HTTPError as exc:
        agent = {"unavailable": str(exc)}
    entry = {
        "label": args.label,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "agent": agent,
        "question_ids": [q["question_id"] for q in selected],
        "batch_size": args.batch_size,
        "delay_s": args.delay,
        "timeout_s": args.timeout,
    }
    info_path = args.out_dir / "run_info.json"
    if info_path.exists() and (args.resume or args.retry_errors):
        info = json.loads(info_path.read_text(encoding="utf-8"))
        info.setdefault("resumed", []).append(entry)
    else:
        info = entry
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")

    ran: list[dict] = []
    started = time.monotonic()
    for number, batch in enumerate(batches, start=1):
        print(f"\n--- batch {number}/{len(batches)}: {', '.join(q['question_id'] for q in batch)}")
        report = run_batch(args, batch)
        (args.out_dir / f"batch-{report['run_id']}.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        for r in report["results"]:
            rows[r["question_id"]] = {**r, "run_id": report["run_id"]}
        ran.extend(report["results"])
        # Saved before printing, so a display failure can never lose results.
        write_combined(combined_path, rows, questions)
        print_rows(report["results"], types)
        # A usage limit fails every question after it: stop instead of
        # recording the rest of the run as errors.
        streak = trailing_errors(ran)
        if args.stop_after_errors and streak >= args.stop_after_errors:
            print(f"\nstopping: the last {streak} questions failed ({_short(ran[-1].get('error'), 160)})")
            print("saved results are kept; fix the cause, then rerun with --retry-errors or --resume")
            break

    ordered = [rows[q["question_id"]] for q in questions if q["question_id"] in rows]
    print_summary(ordered, types, ran, time.monotonic() - started)
    print(f"\ncombined report     : {combined_path}")


if __name__ == "__main__":
    main()
