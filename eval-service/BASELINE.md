# Retrieval baseline — full TAT-DQA corpus

Run `563bbceb-1bfd-4ce5-baf5-19561dde9bbe`, 2026-09-09, on commit `3e31430`.
Raw per-question rows stay under `results/` (gitignored); this file is the
record of the numbers themselves.

## Corpus

```
corpus files      2,758  ->  2,386 unique documents (the dataset ships the
                             same PDF under several uids)
index             2,446 documents · 14,386 chunks · 2,748 pages · 5,082 tables
gold coverage     101/101 documents
answerable        95/95 questions (5 of the 100 are deliberately unanswerable)
```

## Results

100 questions, `candidate_k=30`, `top_k=10`, document-level relevance.

| variant | recall@1 | recall@5 | recall@10 | MRR | mean | p95 |
|---|---|---|---|---|---|---|
| dense | 0.1368 | 0.3421 | 0.3895 | 0.2213 | 54 ms | 65 ms |
| **bm25** | **0.2526** | **0.4684** | **0.5211** | **0.3562** | 82 ms | 104 ms |
| hybrid | 0.2053 | 0.4211 | 0.4421 | 0.3044 | 105 ms | 134 ms |
| hybrid_reranker | 0.2421 | 0.4105 | 0.4947 | 0.3235 | 1,350 ms | 1,669 ms |

## What the numbers say

**BM25 wins on every metric**, and cheaply. It beats dense by 37% relative on
recall@5 (0.4684 vs 0.3421) and nearly doubles recall@1. Financial questions
are lexically precise -- company names, fiscal years, line-item labels -- and
exact term matching suits them better than embedding similarity does.

**Hybrid scores below its own best component.** Fusing dense and BM25 gives
0.4211 recall@5, under BM25's 0.4684, so the dense half is pulling the fusion
down rather than complementing it. The fusion weighting is miscalibrated for
this corpus.

**The reranker costs 16x the latency and hurts at k=5.** recall@5 drops to
0.4105 while recall@10 rises to 0.4947: it reorders within a candidate set it
cannot improve, and sometimes pushes correct documents out of the top 5.

The weak component is dense retrieval, and the most expensive component is not
paying for itself. That is where the effort belongs.

## Scoring note

Identity is canonicalised through `identity_aliases.json` before gold and hits
are compared. The corpus contains 333 groups of byte-identical documents under
different uids, and only one copy of each is indexed -- so without the mapping
a correct retrieval carrying the twin's name scores as a miss. With it,
coverage goes from 93/101 to 101/101 gold documents and bm25 from recall@5
0.4474 to 0.4684. Nothing is inflated; those retrievals were already correct.

## Reproducing

```bash
cd eval-service
python - <<'PY'
import json
from pathlib import Path
from app.retrieval_benchmark import run_ablation, format_comparison
qs = json.loads(Path('questions_setA_practice.json').read_text(encoding='utf-8'))
print(format_comparison(run_ablation('http://127.0.0.1:8002', qs,
                                     candidate_k=30, top_k=10,
                                     output_dir=Path('results'))))
PY
```

Takes about 160 s; the reranker variant is roughly 80% of that. Regenerate
`identity_aliases.json` with `scripts/build_identity_aliases.py` if the corpus
changes.
