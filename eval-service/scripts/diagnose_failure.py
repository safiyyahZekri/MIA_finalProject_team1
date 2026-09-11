"""Locate the stage where an answer failed: index text, retrieval, or a later
agent step.

For each question id, from a saved answer run:

  * whether each gold fact is in the index text of the gold pages, both
    exactly and ignoring case, spacing and punctuation. A fact missing or
    garbled there points at OCR, table extraction or chunking;
  * the agent's steps, with retrieved gold pages marked;
  * where the passage holding the most gold facts ranks when each of the
    agent's queries is replayed against the endpoints the agent calls, at the
    agent's top 5 and within the top 50, with reranking on and off.

Replays need retrieval-api running on the same index as the run.

Usage, from eval-service/:
  python scripts/diagnose_failure.py path/to/chunks.jsonl \
      results/answer-eval/full-100-v1 A003 A068
"""
import argparse
import json
import re
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.compare_answer_runs import Judge  # noqa: E402

# The endpoints agent-service's retrieve step calls for each question type.
ENDPOINTS = {
    "numerical": ("/search/vector", "/search/tables"),
    "table": ("/search/tables", "/search/vector"),
    "text": ("/search/vector", "/search/bm25"),
}


def _loose(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _search(base_url: str, path: str, body: dict) -> list:
    request = urllib.request.Request(
        base_url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)["hits"]


def _rank(hits: list, chunk_ids: set):
    return next((i for i, hit in enumerate(hits, 1) if hit.get("chunk_id") in chunk_ids), None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("chunks", type=Path, help="chunks.jsonl exported from the index")
    parser.add_argument("run_dir", type=Path, help="answer run directory holding combined.json")
    parser.add_argument("ids", nargs="+", help="question ids, e.g. A003 A068")
    parser.add_argument("--retrieval-url", default="http://localhost:8002")
    parser.add_argument("--no-replay", action="store_true", help="skip replaying queries")
    args = parser.parse_args()
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")

    judge = Judge(args.chunks)
    rows = {
        row["question_id"]: row
        for row in json.loads((args.run_dir / "combined.json").read_text(encoding="utf-8"))["results"]
    }
    pages = defaultdict(list)
    with args.chunks.open(encoding="utf-8") as handle:
        for line in handle:
            chunk = json.loads(line)
            pages[judge._canon(chunk["document_id"])].append(chunk)

    for qid in args.ids:
        question, row = judge.questions[qid], rows[qid]
        print(f"\n=== {qid} [{question['answer_type']}] outcome {judge.outcome(row)}")
        print("Q:", question["question_text"])
        print("gold:", json.dumps(question.get("ground_truth_answer"), ensure_ascii=False))

        gold, facts = set(), []
        for evidence in question.get("gold_evidence") or []:
            gold.add(judge._canon(evidence["source_doc_uid"]))
            facts += evidence.get("gold_facts") or []
        facts = [fact for fact in dict.fromkeys(facts) if len(_loose(fact)) >= 2]

        print("-- gold facts in the index")
        fact_chunks = set()
        for page in sorted(gold):
            chunks = pages.get(page, [])
            print(f"  page {page}: {len(chunks)} chunks")
            for fact in facts:
                exact = sum(1 for c in chunks if fact in (c.get("text") or ""))
                loose = sum(1 for c in chunks if _loose(fact) in _loose(c.get("text") or ""))
                print(f"    {fact[:70]!r}: exact {exact}, ignoring case/spacing/punctuation {loose}")
            scored = [
                (sum(1 for fact in facts if _loose(fact) in _loose(c.get("text") or "")), c["chunk_id"])
                for c in chunks
            ]
            top = max((score for score, _ in scored), default=0)
            if top:
                fact_chunks |= {chunk_id for score, chunk_id in scored if score == top}

        print("-- agent steps")
        for step in row.get("system_trace") or []:
            step = {key: value for key, value in step.items() if key != "usage"}
            if step.get("step") == "retrieve":
                step["hits"] = [
                    ("GOLD " if judge._canon(hit.rsplit(":p", 1)[0]) in gold else "") + hit
                    for hit in step.get("hits") or []
                ]
            print("  ", json.dumps(step, ensure_ascii=False))
        print("-- answer:", json.dumps((row.get("predicted_answer") or {}).get("params"), ensure_ascii=False))

        if args.no_replay or not fact_chunks:
            continue
        print(f"-- replayed retrieval: rank of the passage holding the most gold facts ({len(fact_chunks)} chunk(s))")
        for step in row.get("system_trace") or []:
            if step.get("step") != "retrieve":
                continue
            cells = []
            for path in ENDPOINTS.get(step.get("question_type"), ("/search/vector",)):
                query = {"query": step["query"]}
                top5 = _rank(_search(args.retrieval_url, path, {**query, "top_k": 5}), fact_chunks)
                deep = {**query, "top_k": 50, "candidate_k": 250}
                top50 = _rank(_search(args.retrieval_url, path, deep), fact_chunks)
                plain = _rank(_search(args.retrieval_url, path, {**deep, "rerank": False}), fact_chunks)
                cells.append(f"{path.rsplit('/', 1)[-1]}: top5={top5} top50={top50} top50-no-rerank={plain}")
            print(f"   {step['query'][:80]!r}\n     " + " | ".join(cells))


if __name__ == "__main__":
    main()
