# Answer-shape guidance experiment

This is the third no-preprocessing optimization. It changes only the existing
answer-extraction prompt: no PDFs are processed, no embeddings/indexes are
rebuilt, and no extra LLM call is added.

The full-100 audit found repeated output-form failures: a support page was put
inside a multi-span answer, causal answers were split into independent spans,
currency scales were dropped, and a full company name was shortened. Enable:

```bash
ANSWER_SHAPE_GUIDANCE=true
```

The new rules are shared by Anthropic, Groq and Ollama. The flag is false by
default and is shown by `GET /health` so results cannot be mislabeled.

## Controlled off/on test

Keep the real LLM model, corpus, retrieval settings, `TOP_K_FINAL`, grading,
and retries identical. Do not enable query decomposition or hybrid reranking
during this isolated test.

Run the baseline:

```bash
QUERY_DECOMPOSITION=false \
HYBRID_RERANKING=false \
ANSWER_SHAPE_GUIDANCE=false \
docker compose up -d --build --no-deps agent-service eval-service

curl --fail http://127.0.0.1:8003/health

python eval-service/scripts/run_answer_eval.py \
  --ids A005,A013,A025,A043,A045,A049,A057,A073,A081,A086 \
  --batch-size 2 --timeout 600 --label answer-shape-off \
  --out-dir eval-service/results/answer-shape-off
```

Run the candidate against the same ten questions:

```bash
QUERY_DECOMPOSITION=false \
HYBRID_RERANKING=false \
ANSWER_SHAPE_GUIDANCE=true \
docker compose up -d --build --no-deps agent-service eval-service

curl --fail http://127.0.0.1:8003/health

python eval-service/scripts/run_answer_eval.py \
  --ids A005,A013,A025,A043,A045,A049,A057,A073,A081,A086 \
  --batch-size 2 --timeout 600 --label answer-shape-on \
  --out-dir eval-service/results/answer-shape-on
```

Compare the saved runs:

```bash
python eval-service/scripts/compare_decomposition_runs.py \
  --optimization answer_shape_guidance \
  --baseline eval-service/results/answer-shape-off \
  --candidate eval-service/results/answer-shape-on \
  --output eval-service/results/delta-answer-shape.json
```

The quick set is diagnostic, not a final benchmark. If EM/F1 improves without
new errors or wrong answers, repeat the off/on comparison on all 100 questions
before combining it with query decomposition or hybrid reranking. Do not claim
an improvement until those real-model results exist.
