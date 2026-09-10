# Entity-first document routing experiment

This no-preprocessing optimization resolves company names before retrieving the
requested financial metric. It uses the existing classification call and
existing dense/BM25 index. It performs no OCR, chunking, embedding, or writes.

## Why this candidate is next

In the saved 100-question audit, 33 answerable questions were declined after
the gold document was never retrieved. Every one explicitly names its company.
The ordinary corpus-wide query can overweight common financial terms and return
the same line item from another report. Entity routing first finds a document
using only the stated company name, verifies the hit contains that entity, and
then scopes the metric search to it.

## Controlled off/on trial

Keep the model, shared corpus, retrieval settings, grading, and other experiment
flags fixed. The focused set contains single-company misses, cross-company
misses, one previously retrieved control, and one unanswerable safety control.

Baseline:

```bash
QUERY_DECOMPOSITION=false \
HYBRID_RERANKING=false \
ANSWER_SHAPE_GUIDANCE=false \
RETRY_EVIDENCE_FUSION=false \
ENTITY_DOCUMENT_ROUTING=false \
docker compose up -d --build --no-deps agent-service eval-service

curl --fail http://127.0.0.1:8003/health

python eval-service/scripts/run_answer_eval.py \
  --ids A003,A008,A015,A020,A024,A026,A036,A054,A064,A070,A017,A009 \
  --batch-size 2 --timeout 600 --label entity-routing-off \
  --out-dir eval-service/results/entity-routing-off
```

Candidate:

```bash
QUERY_DECOMPOSITION=false \
HYBRID_RERANKING=false \
ANSWER_SHAPE_GUIDANCE=false \
RETRY_EVIDENCE_FUSION=false \
ENTITY_DOCUMENT_ROUTING=true \
ENTITY_ROUTING_CANDIDATES=5 \
docker compose up -d --build --no-deps agent-service eval-service

curl --fail http://127.0.0.1:8003/health

python eval-service/scripts/run_answer_eval.py \
  --ids A003,A008,A015,A020,A024,A026,A036,A054,A064,A070,A017,A009 \
  --batch-size 2 --timeout 600 --label entity-routing-on \
  --out-dir eval-service/results/entity-routing-on
```

Compare:

```bash
python eval-service/scripts/compare_decomposition_runs.py \
  --optimization entity_document_routing \
  --baseline eval-service/results/entity-routing-off \
  --candidate eval-service/results/entity-routing-on \
  --output eval-service/results/delta-entity-routing.json
```

Adopt it only if retrieval Recall@5 or answer quality improves without new
errors or an incorrect answer to `A017`. Report latency because entity routing
adds retrieval calls even though it adds no model calls. Confirm a positive
focused result on all 100 questions before enabling it by default.
