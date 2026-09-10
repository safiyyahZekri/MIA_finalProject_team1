# Search-time optimization experiments

These changes use the existing index without OCR, corpus embeddings, identity
changes or index writes. Each runtime feature defaults to false. Implementation
and offline tests are complete; live EM/F1 improvements remain unmeasured.

## Implemented behavior

| Optimization | Switch / tool | Behavior and limits |
|---|---|---|
| Evidence calibration | `GRADE_CALIBRATION`, `calibrate_evidence_gate.py` | Sweep retrieval-score and model-confidence gates on reviewed development evidence. Never overrides a model's `sufficient=false`. |
| Adaptive Top-K | `ADAPTIVE_TOP_K` | Widen 5 → 10 → 15 after failed grading, capped by `ADAPTIVE_TOP_K_MAX` and API maximum 50. |
| Diversification | `EVIDENCE_DIVERSIFICATION` | Retrieve the candidate pool, then select evidence using relevance and token-overlap redundancy. Preserve distinct chunks on the same page and reserve routed-entity evidence. |
| Query expansion | `QUERY_EXPANSION` | Add original question plus controlled financial alternatives: revenue/net sales, operating income/profit, EPS, R&D, SG&A. These are search alternatives, not accounting equivalences. |
| Table rewriting | `TABLE_QUERY_REWRITING` | Add a table query retaining company wording, metric, periods and units for table/numerical questions. At most two variants across both query switches. |
| Fusion/reranker tuning | `tune_retrieval.py` | Sweep dense weight, RRF constant, candidate count and reranking off/on using identical corpus-wide queries. Save JSON/CSV and a Markdown comparison. |
| Adjacent evidence | `ADJACENT_EVIDENCE` | On retries, read current/adjacent pages for two strong anchors. At most two chunks per page, four added context chunks, 50 total hits. Context gets `score=0`, `context_only=true`; metadata matches cannot raise the gate. |
| Answer repair | `ANSWER_REPAIR` | One extra extraction call after local checks for singleton multi-span, missing/invalid citation indexes or presentation problems. Accept fact-preserving output with supporting citations. Reject changed numbers, units, qualifiers/order and honor a model's abstention. Does not infer missing units or guess arithmetic. |
| Normalization | `ANSWER_NORMALIZATION` | Normalize whitespace, Unicode numeric minus and percentage spacing. Retain currency, scale, precision, commas, ranges and value order. The evaluation scorer is unchanged. |
| Type settings | `QUESTION_TYPE_RETRIEVAL`, `RETRIEVAL_PROFILES_JSON` | Validated profiles for the existing `text`, `table`, `numerical` classifier outputs; never use gold answer-type or unanswerable labels at runtime. |

Extra searches preserve explicit document scopes. With the new search policies
enabled, metadata results cannot displace a full ranked evidence set or supply
metadata-match confidence to grading. Service errors remain visible.

## Trial order

1. Adaptive Top-K, then diversification, each separately.
2. Query expansion and table rewriting separately, then combined.
3. Adjacent context on weak retries.
4. Fusion/reranker sweep on development questions; confirm the winner on held-out
   questions before applying per-type profiles.
5. Answer repair. Normalization helps presentation but did not improve the saved benchmark.
6. Evidence calibration after obtaining reviewed development evidence labels.

`MIN_EVIDENCE_SCORE` previously controlled only the mock grader. Real models
judge sufficiency semantically. The new gate can reject weak approvals or relax
a previously configured gate, but cannot recover semantic model rejections.
The calibration report counts those separately. No unmeasured threshold was installed.

## First live off/on trial

Keep the model, corpus and questions fixed. Coordinate with the index owner to
avoid ingestion/corrections during the trial. Run from the repository root in
the evaluation Python environment. Configure real provider credentials and
`RETRIEVAL_API_URL` first; Docker defaults to `http://retrieval-api:8002`.
The document processor is not needed.

```bash
export QUERY_DECOMPOSITION=false HYBRID_RERANKING=false
export ENTITY_DOCUMENT_ROUTING=false ANSWER_SHAPE_GUIDANCE=false
export RETRY_EVIDENCE_FUSION=false ADAPTIVE_TOP_K=false
export EVIDENCE_DIVERSIFICATION=false QUERY_EXPANSION=false
export TABLE_QUERY_REWRITING=false ADJACENT_EVIDENCE=false
export ANSWER_REPAIR=false ANSWER_NORMALIZATION=false
export QUESTION_TYPE_RETRIEVAL=false GRADE_CALIBRATION=false

docker compose up -d --build --no-deps agent-service eval-service
curl --fail http://127.0.0.1:8003/health
python eval-service/scripts/run_answer_eval.py \
  --sample 20 --batch-size 2 --timeout 600 \
  --label adaptive-off --out-dir eval-service/results/adaptive-off

export ADAPTIVE_TOP_K=true
docker compose up -d --build --no-deps agent-service
python eval-service/scripts/run_answer_eval.py \
  --sample 20 --batch-size 2 --timeout 600 \
  --label adaptive-on --out-dir eval-service/results/adaptive-on

python eval-service/scripts/compare_decomposition_runs.py \
  --optimization adaptive_top_k \
  --baseline eval-service/results/adaptive-off \
  --candidate eval-service/results/adaptive-on \
  --output eval-service/results/adaptive-comparison.json
```

Use fresh output directories. The comparison rejects mock runs, mismatched
question sets and settings changes beyond the chosen switch. Keep auxiliary
settings identical on both sides (including profiles/thresholds) and toggle only
their enable switch. Compare EM/F1, retrieval recall, errors, abstentions,
latency and cost. Repeated tuning on practice questions is not test-set proof.

## Fusion/reranker sweep

Deploy the updated retrieval-api code first; existing indexes remain compatible.
The runner verifies applied settings and reranking confirmation so an older
service cannot silently ignore the experiment parameters.

```bash
python eval-service/scripts/tune_retrieval.py \
  --retrieval-url "$RETRIEVAL_API_URL" \
  --questions /path/to/development-questions.json \
  --weights 0.35 0.55 --rrf-constants 60 --candidates 30 60 \
  --output eval-service/results/retrieval-tuning
```

Default grid: eight configurations; hard cap: 48. Each reports Recall/Precision/
Hit Rate at 1/5/10, MRR, mean/median/p95 latency and errors. Failed configurations
cannot win and the command exits nonzero if any fail. No settings are applied
automatically. Select on development data and validate on held-out data.

A profile example (syntax demonstration, not a measured recommendation):

```bash
export RETRIEVAL_PROFILES_JSON='{"numerical":{"top_k":8,"candidate_k":60,"dense_weight":0.35,"rrf_k":60}}'
export QUESTION_TYPE_RETRIEVAL=true HYBRID_RERANKING=true
docker compose up -d --build --no-deps agent-service
```

Hybrid weights require `HYBRID_RERANKING=true` in the current agent; budgets
apply to ordinary retrieval too. Unknown fields/types and out-of-range values
are rejected. Disable `QUESTION_TYPE_RETRIEVAL` to restore global settings.

## Reviewed evidence calibration

Supply a JSON array of development evidence attempts. Schema example only:

```json
[
  {"question_id":"dev-a","attempt":0,"top_score":0.8,"grade_confidence":0.9,"model_sufficient":true,"evidence_sufficient":true},
  {"question_id":"dev-b","attempt":0,"top_score":0.5,"grade_confidence":0.7,"model_sufficient":true,"evidence_sufficient":false}
]
```

Get score/confidence/model decisions from new `grade` trace entries. A reviewer
must judge `evidence_sufficient` from the actual passages, named entity, period
and needed operands. A gold-document hit alone is not a sufficiency label.
Include positive and negative examples, without duplicate question/attempts.

```bash
python eval-service/scripts/calibrate_evidence_gate.py \
  --labels /path/to/reviewed-development-evidence.json \
  --max-false-accept-rate 0.05 \
  --output eval-service/results/evidence-calibration.json
```

Recommendations require nonempty acceptance within the development error
budget. Validate separately, then set `GRADE_MIN_SCORE`,
`GRADE_MIN_CONFIDENCE`, and `GRADE_CALIBRATION=true`. Defaults are zero.
Enabled type profiles can override these with `min_score` and
`min_grade_confidence`. Calibration does not claim a new EM/F1 result.

## Measured normalization replay

Replayed all 100 saved answers from `ledger-eval-full-100-v1/combined.json` using
the production normalization code and unchanged scorer:

| Metric | Original | Normalized | Delta |
|---|---:|---:|---:|
| Exact Match | 0.2500 | 0.2500 | 0.0000 |
| F1 | 0.34978347 | 0.34978347 | 0.0000 |

Only A027's presentation changed. This was not a fresh agent/retrieval run.
Saved result: `results/no-preprocessing-verification/normalization-replay.json`.

```bash
python eval-service/scripts/replay_answer_normalization.py \
  --report /path/to/ledger-eval-full-100-v1/combined.json \
  --output eval-service/results/no-preprocessing-verification/normalization-replay.json
```

Remaining measurements need a running real-model agent, the shared retrieval
corpus, and reviewed development labels for calibration. Offline fixture tests
prove implementation behavior, not quality gains. No live EM/F1 gain is claimed.
