# eval-service

Service #5 in the LEDGER architecture — Evaluation & Observability. Two jobs:

1. **Centralized Langfuse tracing.** Other services POST step events here
   instead of each holding Langfuse credentials. If `LANGFUSE_PUBLIC_KEY` /
   `LANGFUSE_SECRET_KEY` aren't set, tracing degrades to structured local
   logging + in-memory storage so nothing blocks on Langfuse being
   provisioned yet.
2. **Automated benchmark harness.** Runs a held-out TAT-DQA-style question
   set end-to-end against the live system (question -> orchestrator/agent ->
   predicted answer -> compare to gold), and reports Exact Match, F1,
   numerical accuracy, Recall@K / Precision@K / Hit Rate / MRR, average
   latency, and (if the system reports them) LLM calls / token usage / cost.
   It also calls `answer-validator-api` for every predicted answer so
   "schema validity rate" comes for free.

## Run it

```bash
docker build -t ledger-eval-service .
docker run -p 8003:8003 \
  -e LANGFUSE_PUBLIC_KEY=... -e LANGFUSE_SECRET_KEY=... \
  ledger-eval-service
```

Or without Docker / Langfuse (fine for local dev):

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8003
```

## Tracing endpoints

- `POST /trace/start {"name": "...", "metadata": {...}}` -> `{"trace_id": "..."}`
- `POST /trace/step {"trace_id", "name", "input", "output", "latency_ms", "tokens", "metadata"}`
- `POST /trace/end {"trace_id", "output"}`
- `GET /trace/{trace_id}` — inspect a trace (local fallback view; full detail lives in Langfuse once enabled)

agent-service / orchestrator-api should call `/trace/start` once per
question, `/trace/step` after each stage (query classification, retrieval,
reranking, tool calls, generation, verification), and `/trace/end` when the
answer is returned.

## Benchmark endpoints

- `POST /benchmark/run` — inline question list in the body:

  ```json
  {
    "system_url": "http://orchestrator-api:8000/ask",
    "validator_url": "http://answer-validator-api:8002/validate_answer",
    "questions": [
      {"question_id": "q1", "question": "...", "document_id": "doc_017", "answer": "$142.5M", "relevant_document_ids": ["doc_017"]}
    ]
  }
  ```

- `POST /benchmark/run_from_file` — multipart upload of the 100-question
  practice set (JSON array or `.jsonl`), plus the same fields as form
  fields. Use this once `questions_setA_practice.json`/`.jsonl` is available
  — **the exact field names in that file should be checked against its
  companion MD file and reconciled with `gold_field` /
  `relevant_docs_field` / `answer_key` below** if they differ from the
  defaults.
- `GET /benchmark/results/{run_id}` — full per-question results + summary,
  also written to `results/{run_id}.json` on disk.

### Config knobs that matter once integration starts

- `answer_key`: dotted path into the system's JSON response where the
  Strict-Answer-Schema object lives, if it's nested (e.g. `"answer"` if the
  orchestrator wraps it as `{"answer": {...}}`). Leave unset if the response
  body *is* the answer object.
- `gold_field` / `relevant_docs_field`: field names in the question records
  for the gold answer and the gold relevant document ids, in case the
  practice-set format differs from the default guess documented in
  `app/benchmark.py`.

## Tests

```bash
pip install pytest
python -m pytest tests/ -q
```

12 tests cover every metric function plus a full mocked run of the
benchmark harness (fake system + fake validator, no real network calls).

## Known gaps / next steps

- Langfuse experiment/dataset comparisons (chunking strategy on/off, etc.)
  are not yet wired up — once `agent-service` exposes a way to run with
  different configs, add a `/benchmark/compare` endpoint that runs the same
  question set twice and diffs the summaries.
- Token usage / cost are only populated if the system response includes
  `llm_calls` / `tokens_used` / `cost_usd`; confirm with Karim (agent) that
  `agent-service` will surface these, and adjust `benchmark.py` if the field
  names differ.
