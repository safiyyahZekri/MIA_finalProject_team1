# No-preprocessing retrieval optimization experiment

Two independent switches are ready:

- `QUERY_DECOMPOSITION`: split multi-entity/operand questions and merge evidence.
- `HYBRID_RERANKING`: use retrieval-api's existing dense+BM25 fusion and cross-encoder.

Both use the existing corpus. No PDF processing, chunk rebuilding, embeddings,
or index writes occur. Their accuracy effects are unmeasured until this trial.

## Controlled four-way trial

Use the same 12 question IDs from `QUERY_DECOMPOSITION_EXPERIMENT.md`. Pause
ingestion and keep the real model, shared index, `TOP_K_FINAL=5`, retry and
grading settings fixed. Use a new output directory for every configuration.

Run agent-service with these four configurations, checking `/health` before
each evaluation:

| Run | QUERY_DECOMPOSITION | HYBRID_RERANKING |
|---|---|---|
| baseline | false | false |
| decomposition | true | false |
| hybrid-reranker | false | true |
| combined | true | true |

For each row:

```bash
QUERY_DECOMPOSITION=<true-or-false> \
HYBRID_RERANKING=<true-or-false> \
docker compose up -d --build --no-deps agent-service eval-service

curl --fail http://127.0.0.1:8003/health

python eval-service/scripts/run_answer_eval.py \
  --ids A001,A018,A020,A022,A024,A026,A048,A059,A017,A028,A009,A016 \
  --batch-size 2 --timeout 600 --label <run-name> \
  --out-dir eval-service/results/<run-name>
```

If retrieval-api returns `reranked=false` for non-empty searches, hybrid modes
fail deliberately. Confirm the shared retrieval service has a cross-encoder
loaded before evaluating those modes.

Produce isolated comparisons:

```bash
# Decomposition effect with legacy retrieval
python eval-service/scripts/compare_decomposition_runs.py \
  --optimization query_decomposition \
  --baseline eval-service/results/baseline \
  --candidate eval-service/results/decomposition \
  --output eval-service/results/delta-decomposition.json

# Hybrid-reranking effect without decomposition
python eval-service/scripts/compare_decomposition_runs.py \
  --optimization hybrid_reranking \
  --baseline eval-service/results/baseline \
  --candidate eval-service/results/hybrid-reranker \
  --output eval-service/results/delta-hybrid-reranker.json

# Decomposition effect after enabling hybrid reranking
python eval-service/scripts/compare_decomposition_runs.py \
  --optimization query_decomposition \
  --baseline eval-service/results/hybrid-reranker \
  --candidate eval-service/results/combined \
  --output eval-service/results/delta-decomposition-on-hybrid.json

# Hybrid-reranking effect after enabling decomposition
python eval-service/scripts/compare_decomposition_runs.py \
  --optimization hybrid_reranking \
  --baseline eval-service/results/decomposition \
  --candidate eval-service/results/combined \
  --output eval-service/results/delta-hybrid-on-decomposition.json
```

Adopt a switch only when it improves answer quality without introducing
unanswerable-question regressions or errors. Report latency and cost changes
beside EM/F1. Because the trial emphasizes cross-document questions, confirm
the selected configuration on all 100 questions or a separate held-out set.
