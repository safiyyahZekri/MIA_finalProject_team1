# eval-service

Service #5 in the LEDGER architecture — Evaluation & Observability. Two jobs:

1. **Centralized Langfuse tracing.** Other services POST step events here
   instead of each holding Langfuse credentials. If `LANGFUSE_PUBLIC_KEY` /
   `LANGFUSE_SECRET_KEY` aren't set, tracing degrades to structured local
   logging + in-memory storage so nothing blocks on Langfuse being
   provisioned yet.
2. **Automated benchmark harness.** Runs the held-out practice question set
   end-to-end against the live system (question -> agent-service ->
   predicted answer -> compare to gold), and reports Exact Match, F1,
   numerical accuracy, Recall@K / Precision@K / Hit Rate / MRR, average
   latency, and (if the system reports them) LLM calls / token usage / cost.
   It also calls `answer-validator-api` for every predicted answer so
   "schema validity rate" comes for free, and opens its own Langfuse trace
   per question so every result is diagnosable regardless of whether
   agent-service's own tracing is wired up.

No-preprocessing accuracy experiments are documented in
[`QUERY_DECOMPOSITION_EXPERIMENT.md`](QUERY_DECOMPOSITION_EXPERIMENT.md),
[`RETRIEVAL_OPTIMIZATION_EXPERIMENT.md`](RETRIEVAL_OPTIMIZATION_EXPERIMENT.md),
[`ANSWER_SHAPE_EXPERIMENT.md`](ANSWER_SHAPE_EXPERIMENT.md), and
[`RETRY_EVIDENCE_FUSION_EXPERIMENT.md`](RETRY_EVIDENCE_FUSION_EXPERIMENT.md).
Each feature is off by default and recorded in the agent health snapshot for
controlled runs.

## Run it

```bash
docker build -t ledger-eval-service .
docker run -p 8005:8005 \
  -e LANGFUSE_PUBLIC_KEY=... -e LANGFUSE_SECRET_KEY=... \
  ledger-eval-service
```

Or without Docker / Langfuse (fine for local dev):

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8005
```

## Tracing endpoints

- `POST /trace/start {"name": "...", "metadata": {...}}` -> `{"trace_id": "..."}`
- `POST /trace/step {"trace_id", "name", "input", "output", "latency_ms", "tokens", "metadata"}`
- `POST /trace/end {"trace_id", "output"}`
- `GET /trace/{trace_id}` — inspect a trace; includes `trace_url`, the
  actual Langfuse dashboard link, once Langfuse is enabled and
  authenticated (`null` on the local-fallback path).

agent-service / orchestrator-api are free to call these too if they want a
shared trace with eval-service's own steps, but it isn't required — the
benchmark harness always opens its own trace per question independent of
whatever tracing agent-service does internally (see "Failure analysis"
below).

### Langfuse SDK note

The installed SDK (`langfuse==4.15.1`) is the OpenTelemetry-based client
generation, which has a completely different API from the older
`Langfuse.trace()` / `.span()` methods — those methods do not exist on this
version and calling them raises `AttributeError` immediately. `tracing.py`
is written against the actual installed API (`create_trace_id`,
`start_observation`, span `.update()` / `.end()`, `.flush()`,
`get_trace_url`), confirmed by introspecting the installed package rather
than assumed. `tests/test_tracing.py` pins this down with a stand-in that
mirrors the real method signatures, so a future accidental revert to the
old API shape fails a test instead of silently only working on the
local-fallback path.

**What has and hasn't been verified end-to-end:**

- ✅ With no credentials set: local-fallback path fully exercised by
  `test_tracing.py` and the benchmark tests — every trace_id resolves via
  `GET /trace/{id}`.
- ✅ With syntactically-valid but fake credentials against the real
  `https://cloud.langfuse.com` host: the SDK call executes with no
  `AttributeError`/crash, and produces a real HTTP round trip that comes
  back `403 Forbidden` — i.e., the request is correctly shaped and
  reaches Langfuse's real ingestion endpoint; it is only rejected because
  the credentials aren't real. This is the strongest check possible
  without an actual account.
- ❌ **Not verified**: an actual trace appearing on a real Langfuse
  dashboard. This needs a real `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY`
  pair. Once available, run the service with them set and hit
  `POST /trace/start` -> `/trace/step` -> `/trace/end`, then open the
  `trace_url` from `GET /trace/{trace_id}` to confirm it renders — this is
  the one remaining item from the Final Demo Checklist and needs a real
  account to close out.

## Benchmark endpoints

### Read answer citations and retrieval separately

New answer benchmark reports include `citation` and `agent_retrieval` in
each result and the summary. `agent_retrieval.first_attempt` and
`agent_retrieval.final_attempt` score the ranked hits actually passed to
grading. Retry-union coverage is reported as `any_attempt_hit_rate` and
`any_attempt_recall`; it is not a single-search Recall@K. A declined answer
can have zero citation coverage and successful retrieval.

The old per-result `retrieval` and summary `retrieval_*` fields remain
**deprecated citation metrics** for compatibility. New consumers should use
the explicit fields and `metric_definitions`. New Precision@K uses K as its
denominator, whereas legacy citation precision used the returned list length.
Ranks are chunk positions, with document-level relevance; repeated document
hits do not increase recall. Missing/malformed traces yield unavailable
metrics, not fabricated zeros. Unresolved `sha256-...` identities are listed
in `agent_retrieval.unresolved_document_ids`.

Audit an existing run without OCR, re-indexing, or paid model calls (from
`eval-service/`):

```bash
python scripts/audit_answer_eval.py \
  --report /path/to/ledger-eval-full-100-v1/combined.json \
  --questions questions_setA_practice.json \
  --identity-aliases identity_aliases.json \
  --k 5 \
  --output-dir results/full-100-v1-audit
```

This writes `audit.json` (per-question diagnostics and input hashes) and
`audit.md`. The original run is preserved. The alias file is required by
this audit command so a missing deployment asset cannot silently invalidate
the result. New agent traces retain source identity metadata without passage
text; the evaluator can resolve newly indexed hash-only citations from those
traces even when the static alias map predates the document.

See [EVALUATION_AUDIT.md](EVALUATION_AUDIT.md) for the measured full-100-v1
findings. Use the existing direct retrieval benchmark for standalone engine
evaluation; these agent traces include query rewriting and retries.

- `POST /benchmark/run` — inline question list in the body:

  ```json
  {
    "system_url": "http://agent-service:8003/answer",
    "validator_url": "http://answer-validator-api:8004/validate_answer",
    "questions": [
      {
        "question_id": "A001",
        "question_text": "...",
        "ground_truth_answer": 304811,
        "scale": "thousand",
        "source_doc_uid": null,
        "gold_evidence": [{"source_doc_uid": "..."}, {"source_doc_uid": "..."}]
      }
    ]
  }
  ```

- `POST /benchmark/run_from_file` — multipart upload of
  `questions_setA_practice.json` (or a `.jsonl` variant) directly, plus the
  same fields as form fields.
- `GET /benchmark/results/{run_id}` — full per-question results + summary,
  also written to `results/{run_id}.json` on disk (gitignored — see
  below).

### Practice-set record schema (confirmed)

Field names are matched exactly against `questions_setA_practice.json` +
its companion `record_schema_guide_ar.md`, not a generic guess:

| Concept | Field | Notes |
|---|---|---|
| Question text | `question_text` | sent to the system as `{"question": ...}` |
| Gold answer | `ground_truth_answer` | number / string / list / `null` |
| Unanswerable | `answer_type == "unanswerable"` | `ground_truth_answer` is `null` in this case — scored as "did the system correctly abstain with `insufficient_evidence`", not as a text/numeric miss |
| Answer scale | `scale` | `""`, `"thousand"`, `"million"`, `"billion"`, `"percent"` — normalized against in `metrics.numerical_accuracy` before comparing |
| Single-doc scoping id | `source_doc_uid` (top level) | retained for identity; gold scoping is disabled by default and only available as an explicit diagnostic mode |
| Relevant docs for retrieval metrics | `gold_evidence[*].source_doc_uid` | there is no flat `relevant_document_ids` field in the real data |

### Confirmed agent-service contract

`system_url` should point at agent-service's `POST /answer` endpoint (built
specifically for eval-service's use, distinct from the
`POST /agent/query` endpoint orchestrator-api calls). It takes
`{"question": ..., "document_id": ...}` and returns:

```json
{
  "answer": {"answer_type": "...", "evidence": [...], "params": {...}},
  "question_type": "...",
  "retries_used": 2,
  "trace": [ "..." ]
}
```

This was confirmed end-to-end against the real `feature/agent-service`
branch: 100 practice questions run through
`eval-service -> agent-service (mock LLM) -> answer-validator-api`,
0 errors, 100% schema validity. `answer_key="answer"` is therefore the
**default** here, matching the real system — override it only for a
system that returns some other shape. `trace` is confirmed to be a list
(the agent's internal step history); its per-item shape isn't part of the
confirmed contract, so it's still treated as opaque — read verbatim if
present (stored on the result as `system_trace`, clearly separate from
eval-service's own `trace_id`), never required, never validated, and
**never forwarded to answer-validator-api**, since that endpoint's schema
rejects unknown top-level keys — `benchmark.py` strips everything except
`answer_type`/`evidence`/`params` before validation. `retries_used` is
captured as a system-performance metric alongside `llm_calls` /
`tokens_used` / `cost_usd` (`avg_retries_used` in the summary).

If a system instead returns the bare answer object with `trace`/
`question_type` as top-level siblings (no `"answer"` wrapper), pass
`answer_key=None` to fall back to that shape — both are supported and
tested (`test_split_core_answer_confirmed_nested_shape` and
`test_split_core_answer_flat_shape_still_supported`).

### Failure analysis

Every `QuestionResult` (and therefore every row in
`results/{run_id}.json`) carries:

- `trace_id` — opened by eval-service itself for that question, logging
  `call_system`, `validate_schema`, and `score` as steps. This exists
  regardless of whether agent-service's own tracing is wired up.
- `system_trace` — whatever `trace` payload agent-service returned, if any,
  stored verbatim and labeled separately.
- `question_type` — from the confirmed contract, if present.

The run summary's `failed_examples` (up to 5) lists `question_id`,
`trace_id`, and `trace_url` (populated once Langfuse is authenticated) for
anything that errored, failed schema validation, or missed on exact match —
this is the direct link the spec's "diagnosable using Langfuse traces"
failure-analysis requirement needs.

### Config knobs

- `answer_key`: dotted path into the system's JSON response where the
  answer object lives. Defaults to `"answer"`, matching the confirmed
  agent-service contract above. Pass `None` for a system that returns the
  bare answer object with `trace`/`question_type` as top-level siblings
  instead.
- `question_field` / `gold_field` / `scale_field`: override if a future
  practice set renames these; defaults match the confirmed schema above.

## Tests

### Direct retrieval evaluation

The retrieval benchmark calls retrieval directly; it never infers quality from
the agent's final citations. Normal runs are corpus-wide and never send the gold
`source_doc_uid` as a filter. It reports Recall, Precision, and Hit Rate at
1/5/10, MRR, and mean/median/p95 request latency to JSON and CSV.

Run all four comparable configurations against one indexed corpus:

```bash
cd eval-service
python -m app.retrieval_benchmark \
  --retrieval-url http://localhost:8002 \
  --questions questions_setA_practice.json \
  --output-dir results
```

This executes dense-only, BM25-only, hybrid without reranking, and hybrid with
reranking, then writes a comparison including reranker-minus-hybrid deltas.
Pass `--variant dense`, `bm25`, `hybrid`, or `hybrid_reranker` for a single run.
The same operations are exposed as `POST /benchmark/retrieval/run` and
`POST /benchmark/retrieval/ablation`.

For the real OCR path, start document processor on 8001 and retrieval on 8002,
then run:

```bash
python scripts/tatdqa_retrieval_pipeline.py \
  --pdf-dir /path/to/TAT-DQA/pdfs \
  --questions /path/to/TAT-DQA/questions.json \
  --identity-manifest /path/to/filename_to_source_uid.json \
  --batch-size 32 \
  --output-dir results
```

The official TAT-DQA archives name PDFs `<source_doc_uid>.pdf`, which the runner
recognizes automatically. A manifest is only needed for renamed PDFs and takes
precedence over filename inference. The runner preserves identity, skips already
indexed sources unless `--reindex` is supplied, reports OCR/index failures,
batch-indexes, then runs the four-way unscoped ablation. It uses actual PDFs and
the document processor—not synthetic processor JSON.

If an older run indexed this corpus with `sha256-...` document IDs and null
`source_doc_uid` values, reset that dedicated corpus before rerunning. Otherwise
the old and corrected identities coexist as duplicate searchable content and
retrieval metrics cannot be trusted. Corpus deletion is intentionally not done
automatically by this script.

### Service tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -q
```

The suite covers:

- `test_metrics.py` — every metric function, including scale normalization
  (word units and digit-adjacent abbreviations like `"$142.5M"`).
- `test_benchmark.py` — mocked runs, the confirmed-contract answer
  unwrapping (nested `"answer"` shape, `retries_used`, `trace` as a list,
  plus the flat-shape fallback), per-question trace resolution over a real
  10-question subset, and a full run over all 100 real questions with 0
  field-mapping errors.
- `test_api.py` — the actual HTTP endpoints via `TestClient` (not just the
  internal `run_benchmark` function), including a full 100-question run
  through the multipart upload endpoint. Added specifically after finding
  that `main.py` had fallen out of sync with `BenchmarkConfig`'s field
  names in a way the function-level tests alone couldn't catch.
- `test_tracing.py` — regression coverage for the Langfuse SDK API-shape
  fix described above.

All 100 real practice questions have also been run against the actual
`answer-validator-api` service (not mocked) and a stand-in agent-service
implementing the exact confirmed `/answer` contract (nested `"answer"`,
`retries_used`, `trace` as a list), over real HTTP: 100/100, 0 errors,
`schema_validity_rate: 1.0`, every result carrying a resolvable `trace_id`
and a correctly-captured `avg_retries_used`.

## Known gaps / next steps

- **Real Langfuse dashboard verification** (see "Langfuse SDK note" above)
  — needs a real account's credentials.
- **Meaningful failure analysis** — the confirmed integration run against
  `feature/agent-service` used `LLM_PROVIDER=mock` +
  `RETRIEVAL_FALLBACK_TO_MOCK=true` (an offline corpus), so accuracy was
  low (~5%) by design, not a bug — a failed trace correctly showed
  `insufficient_evidence` when the mock corpus had no relevant evidence.
  Once retrieval-api is wired in, re-run the same benchmark against
  `agent-service + retrieval-api` for accuracy numbers, and against
  real failures, that are actually diagnostic (retrieval vs. reasoning vs.
  numerical).
- Langfuse experiment/dataset comparisons (chunking strategy on/off, etc.)
  are not yet wired up — once agent-service exposes a way to run with
  different configs, add a `/benchmark/compare` endpoint that runs the same
  question set twice and diffs the summaries.
- Token usage / cost are only populated if the system response includes
  `llm_calls` / `tokens_used` / `cost_usd` as top-level fields; confirm
  agent-service will surface these under those exact names (`retries_used`
  is already confirmed and captured).
