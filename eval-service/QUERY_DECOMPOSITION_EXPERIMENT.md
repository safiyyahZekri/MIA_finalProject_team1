# Query decomposition: no-preprocessing accuracy experiment

The candidate adds focused searches and a merge that reserves evidence slots
for each subquery. It also retains those results through retries. This tests
decomposition and merging together, not planning in isolation. It uses the
existing index; no OCR, embeddings, or contextual preprocessing is required.

## Focused off/on trial

Use the team's populated retrieval service and configured real model/provider
credentials. Keep the shared index, model, `TOP_K_FINAL=5`, and grading settings
identical for both runs. Pause ingestion during the comparison. Agent health
snapshots record model settings but do not fingerprint the remote corpus.

Run from the repository root with the evaluation Python environment active.
These commands restart agent/eval only; retrieval and validator should already
be running with their existing URLs and volumes.

If the populated index is hosted by a teammate, set `RETRIEVAL_API_URL` to
their current shared URL in your shell or the repository-root `.env` before
both runs. Docker Compose now passes this value to the agent; otherwise it
defaults to `http://retrieval-api:8002`. Check the shared `/health` endpoint
and confirm it reports indexed chunks before starting. Do not start an empty
local index as a substitute for the evaluation corpus.

The repository-root `.env` also needs the existing real `LLM_PROVIDER` and
its provider credentials. Keep credentials in that local ignored file;
do not include them in reports or commits. The agent must not report
`llm_provider: mock` for the real comparison.

```bash
QUERY_DECOMPOSITION=false docker compose up -d --build --no-deps agent-service eval-service
curl --fail http://127.0.0.1:8003/health

python eval-service/scripts/run_answer_eval.py \
  --ids A001,A018,A020,A022,A024,A026,A048,A059,A017,A028,A009,A016 \
  --batch-size 2 --timeout 600 --label decomposition-off \
  --out-dir eval-service/results/decomposition-off

QUERY_DECOMPOSITION=true docker compose up -d --no-deps agent-service
curl --fail http://127.0.0.1:8003/health

python eval-service/scripts/run_answer_eval.py \
  --ids A001,A018,A020,A022,A024,A026,A048,A059,A017,A028,A009,A016 \
  --batch-size 2 --timeout 600 --label decomposition-on \
  --out-dir eval-service/results/decomposition-on

python eval-service/scripts/compare_decomposition_runs.py \
  --baseline eval-service/results/decomposition-off \
  --candidate eval-service/results/decomposition-on \
  --output eval-service/results/decomposition-comparison.json
```

Wait for each health check and verify `query_decomposition` is false/true as
intended before running its variant. The subset includes the eight
cross-document questions plus two unanswerable and two simpler controls.
Use fresh output directories; resume only with identical settings and corpus.

The comparison reports EM, F1, strict numeric accuracy, abstentions, errors,
final retrieval Recall@5/MRR@5, latency, cost, and candidate-minus-baseline
deltas. It checks question-set equality and matching saved model settings,
rejects mock-model runs, and counts errors as zero for EM/F1. Missing metrics
are reported as unavailable. Per-question audits are included in its JSON.

For all 100 questions, replace `--ids ...` with `--all` in both commands and
choose new output directories. Inspect wrong answers and unanswerable controls;
a positive trial result needs confirmation on held-out questions.

Return to the baseline with:

```bash
QUERY_DECOMPOSITION=false docker compose up -d --no-deps agent-service
```

## Verification status

Deterministic graph tests exercise two-operand retrieval through grading,
calculation and citations, disabled mode, retry reuse, planning failure,
scope preservation, deduplication and bounded results. Provider tests verify
structured planning and usage accounting. These do not establish corpus accuracy.

No live EM/F1 rerun was performed locally: no model credentials or running
service stack were configured. The existing 25% EM / 35% F1 remains the latest
measured answer-quality baseline until the off/on experiment runs.
