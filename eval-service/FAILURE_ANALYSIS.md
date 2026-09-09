# Failure analysis — how to do it without the index

`failure_cases.json` holds, for all 100 questions, what was asked, which
documents were gold, and what each of the four variants actually retrieved
(top 10) in baseline run `563bbceb`. That is enough to work through failures
on any machine: **you do not need the shared index or the tunnel.** You only
need the extracted corpus, because every uid is a filename stem — a case is
inspected by opening `<uid>.pdf`.

Regenerate after a new run with:

```bash
python scripts/export_failure_cases.py --comparison results/retrieval-ablation-<run-id>.json
```

## Pre-sorted buckets

`buckets` in the file groups the interesting cases:

| bucket | count | what it shows |
|---|---|---|
| `missed_by_all_variants` | 34 | no variant found the gold document at all |
| `bm25_hit_dense_miss` | 23 | lexical matching worked where embeddings failed |
| `dense_hit_bm25_miss` | 10 | the reverse — semantic matching earned its place |
| `reranker_dropped_from_top5` | 8 | hybrid had it in the top 5, the reranker pushed it out |

The deliverable asks for at least 5 examples. Take a couple from each bucket
rather than 5 of the same kind — the contrast is the finding.

## How to work a single case

**Step 1 —** find the question in `failure_cases.json` by `question_id`; read
its `question` and its `gold` uids.

**Step 2 —** open the gold PDF (`<gold-uid>.pdf`) and locate the passage that
answers it. Note whether the answer sits in prose, in a table, or has to be
computed from several figures.

**Step 3 —** open the top 2-3 documents the variant actually returned
(`variants.<name>.retrieved`). Ask what they had in common with the query that
the gold document did not.

**Step 4 —** write down the mechanism, not the symptom. "Retrieved the right
metric from the wrong company" is a finding; "recall was low" is not.

## What the baseline already suggests

Worth confirming or refuting against real cases rather than assuming:

- **Lexical beats dense here** (bm25 recall@5 0.4684 vs 0.3421). Financial
  questions carry exact tokens — company names, fiscal years, line-item
  labels — that embeddings blur together. The 23 `bm25_hit_dense_miss` cases
  are where to check this.
- **Hybrid scores below its own best component** (0.4211 vs bm25's 0.4684),
  so fusion is diluting a strong lexical signal with a weak dense one rather
  than combining them.
- **The reranker lowers recall@5** (0.4105) while raising recall@10 (0.4947).
  The 8 `reranker_dropped_from_top5` cases are the clearest evidence of it
  actively demoting correct documents.
- **34 questions no variant answers.** Check whether these are cross-document
  or derived-metric questions (`task_family`, `primary_scenario` in the
  question set) — if so the gap is question type, not ranking, and chunking
  or query decomposition is the lever.

## Feeding the bonus-feature choice

The analysis should end pointing somewhere specific. On these numbers the
weak component is dense retrieval and the expensive one is the reranker, so
the defensible options are improving or re-weighting the dense side, or
dropping the reranker for its latency. Let the cases decide which.
