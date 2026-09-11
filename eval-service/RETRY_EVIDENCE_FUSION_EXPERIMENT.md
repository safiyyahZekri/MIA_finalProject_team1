# Retry evidence fusion experiment

This optimization retains useful evidence across the agent's existing query
reformulation retries. It requires no OCR, chunking, embedding, re-indexing, or
additional service/model calls.

## Why this is next

The audited 100-question run had seven questions where a gold document was
found in some retrieval attempt but was absent from the final attempt:

```text
A001 A011 A032 A035 A067 A089 A094
```

Five of those (`A001`, `A032`, `A035`, `A089`, `A094`) ended as answerable
abstentions. The previous graph overwrote its entire evidence list after each
reformulation. With fusion enabled, rankings from all attempts remain local to
that request and are combined using reciprocal-rank fusion. No evidence is
written back to the corpus.

The default fused budget is 10 passages:

```bash
RETRY_EVIDENCE_FUSION=true
RETRY_FUSION_MAX_HITS=10
```

The first attempt remains unchanged. The larger evidence set is used only
after a retry.

## Controlled off/on test

Keep the model, corpus, retrieval configuration, grading settings and question
set fixed. Disable the other experimental switches to isolate retry fusion.
`A017` is included as an unanswerable safety control and `A009` as an
entity-attribution control.

Baseline:

```bash
QUERY_DECOMPOSITION=false \
HYBRID_RERANKING=false \
ANSWER_SHAPE_GUIDANCE=false \
RETRY_EVIDENCE_FUSION=false \
RETRY_FUSION_MAX_HITS=10 \
docker compose up -d --build --no-deps agent-service eval-service

curl --fail http://127.0.0.1:8003/health

python eval-service/scripts/run_answer_eval.py \
  --ids A001,A009,A011,A017,A032,A035,A067,A089,A094 \
  --batch-size 2 --timeout 600 --label retry-fusion-off \
  --out-dir eval-service/results/retry-fusion-off
```

Candidate:

```bash
QUERY_DECOMPOSITION=false \
HYBRID_RERANKING=false \
ANSWER_SHAPE_GUIDANCE=false \
RETRY_EVIDENCE_FUSION=true \
RETRY_FUSION_MAX_HITS=10 \
docker compose up -d --build --no-deps agent-service eval-service

curl --fail http://127.0.0.1:8003/health

python eval-service/scripts/run_answer_eval.py \
  --ids A001,A009,A011,A017,A032,A035,A067,A089,A094 \
  --batch-size 2 --timeout 600 --label retry-fusion-on \
  --out-dir eval-service/results/retry-fusion-on
```

Compare:

```bash
python eval-service/scripts/compare_decomposition_runs.py \
  --optimization retry_evidence_fusion \
  --baseline eval-service/results/retry-fusion-off \
  --candidate eval-service/results/retry-fusion-on \
  --output eval-service/results/delta-retry-fusion.json
```

The diagnostic set establishes whether the mechanism helps, but it is not a
final benchmark. If it improves EM/F1 or reduces valid abstentions without
answering `A017`, confirm it on all 100 questions. Compare input tokens and cost
as well: the optimization adds no calls, but retries can carry more passages.
