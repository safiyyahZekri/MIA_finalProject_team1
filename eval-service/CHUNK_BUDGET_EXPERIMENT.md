# Table chunk budget: does truncating fewer table chunks change retrieval?

2026-09-10. Retrieval only: no model calls, deterministic results.

## The defect

The chunker budgets table chunks in characters (`LEDGER_CHUNK_MAX_CHARS`,
1,800), but `bge-small-en-v1.5` reads at most 512 tokens and silently cuts the
rest before embedding. Table rows ("Row 3: Inventories, net | 3,875 | 3,410")
tokenize far more densely than prose. Measured on the exact text the engine
embeds (`filename + metadata + section + text`):

| table budget | table chunks | over 512 tokens | tokens cut |
|---|---:|---:|---:|
| 1,800 (live) | 5,082 | 478 (9.4%) | 41,186 |
| 1,500 | 5,228 | 446 (8.5%) | 26,202 |
| 1,200 | 5,659 | 117 (2.1%) | 3,455 |
| 1,000 | 6,268 | 8 (0.1%) | 622 |

## Method

No OCR and no document processing. Every table chunk stores its whole table
(`parent_text`) and cells, so `retrieval-api/scripts/table_chunk_budget.py`
re-splits tables with the chunker's own `_table_parts()` under a new budget:

- `measure` first rebuilds at 1,800 and checks the result against the live
  index: **4,958 of 4,958 tables reproduce exactly** (same chunk ids and text).
- `build` writes a complete index to a new directory. Paragraph chunks and
  their vectors are reused unchanged; only table chunks are re-embedded
  (5,659 chunks at 1,200, 6,268 at 1,000; about 7 minutes each on CPU).

Each index was served by its own retrieval-api (the experimental ones on port
8012; the live index on 8002 was never written) and given the same four-variant
ablation as the retrieval baseline: all 100 practice questions (95 scored),
`candidate_k=30`, `top_k=10`, unscoped.

## Results

The control reproduces the retrieval baseline (run `563bbceb`) exactly.

| variant | metric | 1,800 (live) | 1,200 | 1,000 |
|---|---|---:|---:|---:|
| dense | recall@1 | 0.1368 | 0.1368 | 0.1368 |
| dense | recall@5 | 0.3421 | **0.3579** | 0.3368 |
| dense | recall@10 | 0.3895 | 0.4000 | 0.3789 |
| dense | MRR | 0.2213 | 0.2252 | 0.2238 |
| bm25 | recall@5 | 0.4684 | 0.4579 | 0.4684 |
| bm25 | MRR | 0.3562 | 0.3613 | 0.3678 |
| hybrid | recall@5 | 0.4211 | 0.4105 | 0.4211 |
| hybrid | MRR | 0.3044 | 0.2989 | 0.3134 |
| hybrid + reranker | recall@5 | 0.4105 | 0.4000 | **0.4211** |
| hybrid + reranker | recall@10 | 0.4947 | 0.4947 | 0.5158 |
| hybrid + reranker | MRR | 0.3235 | 0.3148 | 0.3306 |

Questions whose recall@5 changed against 1,800:

| variant | 1,200 | 1,000 |
|---|---|---|
| dense | gained A048, A062 | gained A048; lost A057 |
| hybrid | lost A054 | gained A055; lost A054 |

Latencies are not compared: the live retrieval-api sends traces to
eval-service and the experimental ones did not, so their lower times are not a
speed-up from the budget.

## Why the effect is so small

For each changed question, the gold facts were located in the chunks of their
gold documents and their position measured in embedder tokens:

| question | change | gold fact at 1,800 | what actually changed |
|---|---|---|---|
| A048 | gained (dense, both budgets) | token 283 of a 601-token table chunk, inside the limit | the chunk shrank to 431 / 382 tokens |
| A062 | gained (dense, 1,200 only) | tokens 235 and 244 of a 634-token chunk, inside the limit | shorter chunks; the gain is gone at 1,000 |
| A055 | gained (hybrid, 1,000) | token 381 of a 514-token chunk, inside the limit | fact now at token 185 of 318 |
| A054 | lost (hybrid, both budgets) | tokens 158-189; its chunks are identical under every budget | outranked by other documents' extra chunks |
| A057 | lost (dense, 1,000) | not found verbatim (OCR differs from the gold wording); its 7 chunks are identical under every budget | outranked by other documents' extra chunks |

**No gold fact in these questions sat past the 512-token cut.** The fix removes
truncation, but none of these questions depended on a truncated row. The gains
come from gold tables split into shorter, more focused chunks; the losses come
from the added chunks (14,386 → 14,963 → 15,572) crowding other documents out
of the top 5. They cancel: no variant moves by more than two of 95 questions.

## Conclusion

A real defect, fixed and measured, with a negligible effect on retrieval. No
budget improves every variant, so the live index keeps 1,800 characters. The
truncation share being 9.4% did not mean 9.4% of questions targeted a truncated
row, and on this question set none did.

## Reproducing

Inside the retrieval-api image, with the live index volume mounted read-only
at `/data`, the model cache mounted, and `retrieval-api/` mounted at `/work`:

```bash
python scripts/table_chunk_budget.py measure
python scripts/table_chunk_budget.py build --table-max-chars 1200 --output-dir /exp/t1200
```

Serve the new directory with a second retrieval-api
(`-e LEDGER_RETRIEVAL_DATA_DIR=/data -v <exp dir>:/data -p 8012:8002`), then
run the ablation against both from `eval-service/`:

```bash
python -m app.retrieval_benchmark --retrieval-url http://localhost:8012 \
  --questions questions_setA_practice.json --output-dir results/chunk-budget-experiment
```
