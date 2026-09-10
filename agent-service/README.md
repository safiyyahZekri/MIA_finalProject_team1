# agent-service — LEDGER's Reasoning "Brain"

LangGraph-based reasoning service for **Project LEDGER**. It routes a
question, retrieves evidence via `retrieval-api`, grades that evidence,
retries on weak evidence, reasons over it, and returns a
schema-compliant, cited answer — never guessing or hallucinating a number.

## Query decomposition (optional, no preprocessing)

Set `QUERY_DECOMPOSITION=true` and restart the agent. A planning call asks the
configured model for up to three focused searches for separate entities or
operands. Simple questions can keep the single-query path. The planner sees
only the question, not gold answers, source IDs, or dataset labels.

The agent searches the original query and subqueries with its existing tools.
It reserves a unique hit per subquery, then fills remaining slots by rank
fusion. `TOP_K_FINAL`, normalized confidence, grading, calculation and citation
validation remain in effect. Distinct chunks on the same page survive the
merge. Subquery results persist within this request's retries, without any
shared cache or index writes.

The feature is off by default for baseline comparison. Enabling it adds one
planning call per question and, for a split question, up to six additional
retrieval calls. Plans are bounded to three subqueries and rejected if the
evidence budget cannot accommodate them. Planning failures log a fallback
and use the ordinary query; retrieval errors still surface normally.

Anthropic, Groq and Ollama use model-generated, validated plans. Mock mode
returns no split and cannot establish accuracy gains. `/health` records
`query_decomposition`; traces record planning usage, subquery latencies,
document IDs, and the merged ranking without adding full document text.

Follow the [off/on experiment guide](../eval-service/QUERY_DECOMPOSITION_EXPERIMENT.md)
to measure EM/F1 changes using the existing corpus.

## Run it right now (zero config)

```bash
cd agent-service
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8003
```

That's it. No API key, no retrieval-api, no database required. By default:

- `LLM_PROVIDER=mock` → a deterministic, offline heuristic stand-in for the
  LLM (classification, query reformulation, evidence grading, and answer
  extraction all work without any network call or API key).
- If `retrieval-api` isn't reachable at `RETRIEVAL_API_URL`, the retrieval
  client automatically falls back to a small built-in mock corpus, so the
  whole graph is runnable and demoable in total isolation from the rest of
  the team's services.

Try it:

```bash
curl -X POST http://localhost:8003/answer \
  -H "Content-Type: application/json" \
  -d '{"question": "What was the percentage change in operating expenses from 2020 to 2021?"}'
```

Run the test suite (11 tests, fully offline, ~1s):

```bash
pytest tests/ -v --asyncio-mode=auto
```

## Running against a local LLM (Ollama + qwen2.5) — recommended

No API key, no cloud cost, everything stays on your machine.

1. Install [Ollama](https://ollama.com) and pull a qwen2.5 tag:

   ```bash
   ollama pull qwen2.5:7b
   ```

   Smaller machines can use `qwen2.5:3b` or `qwen2.5:1.5b` instead — just
   point `OLLAMA_MODEL` at whatever you pulled (`ollama list` to check exact
   tag names).

2. Make sure the server is up (it's usually auto-started as a background
   service after install; if not: `ollama serve`).

3. Configure the agent-service:

   ```bash
   cp .env.example .env
   # .env already defaults to:
   #   LLM_PROVIDER=ollama
   #   OLLAMA_BASE_URL=http://localhost:11434
   #   OLLAMA_MODEL=qwen2.5:7b
   # edit OLLAMA_MODEL if you pulled a different tag
   ```

4. Run it:

   ```bash
   export $(grep -v '^#' .env | xargs)   # or use python-dotenv / your shell's own .env loading
   uvicorn app.main:app --host 0.0.0.0 --port 8003
   ```

5. Sanity-check the model is actually reachable:

   ```bash
   curl http://localhost:8003/health
   # {"status": "ok", "llm_provider": "ollama",
   #  "ollama": {"reachable": true, "base_url": "...", "model": "qwen2.5:7b"}}
   ```

   If `reachable` is `false` or `status` is `"degraded"`, the response
   tells you exactly what to run (`ollama serve` / `ollama pull ...`).

6. Ask it something:

   ```bash
   curl -X POST http://localhost:8003/answer \
     -H "Content-Type: application/json" \
     -d '{"question": "What was the percentage change in operating expenses from 2020 to 2021?"}'
   ```

**How the Ollama provider stays reliable (`OllamaLLM` in `app/llm.py`):**
every call to the model asks for `format: "json"` and a specific schema in
the prompt, then validates the reply with the same pydantic models used
everywhere else in this service. If qwen2.5 replies with malformed JSON,
it gets one corrective retry ("your previous reply wasn't valid JSON...");
if that still fails — or Ollama isn't reachable at all — the call degrades
to the offline `MockLLM` heuristic for that single step instead of
crashing the request. You'll see this in the logs as a `WARNING`, and the
graph still returns a schema-valid answer either way. Token usage
(`prompt_eval_count` / `eval_count` from Ollama's response) is attached to
each `classify` / `grade` / `reformulate` / `reason` step in the response
`trace`, so you can see what the local model is actually costing per
question.

## Switching providers

- **`LLM_PROVIDER=mock`** — zero-dependency regex heuristics, no model or
  network needed at all. Good for quick iteration on the graph shape
  itself, but too limited for real TAT-DQA coverage (fixed at 2 numeric
  operands, no sorting/composed arithmetic) — don't demo or grade against
  this mode.
- **`LLM_PROVIDER=ollama`** — real local LLM via `ollama serve`. Good for
  solo development, but **only works on the machine Ollama is installed
  on** — don't rely on this for the team's actual demo/grading unless
  everyone installs Ollama, or the run always happens on the same laptop.
- **`LLM_PROVIDER=groq`** — hosted LLM via a shared `GROQ_API_KEY` (get one
  free at https://console.groq.com). Fast, and doesn't depend on any one
  teammate's machine being on — **recommended for the team demo**, since
  it works identically no matter whose computer runs `docker compose up`.
  Uses `GROQ_MODEL` (defaults to `llama-3.3-70b-versatile`).
- **`LLM_PROVIDER=anthropic`** — set `ANTHROPIC_API_KEY` and optionally
  `ANTHROPIC_MODEL` (defaults to `claude-haiku-4-5-20251001`) to use Claude
  instead, via structured output.
- **Real retrieval-api**: set `RETRIEVAL_API_URL` to your teammate's
  running service. The contract this client expects is documented at the
  top of `app/retrieval_client.py` (`/search/vector`, `/search/bm25`,
  `/search/tables`, `/filter`, all returning `{"hits":[...]}`).

All three real providers (`ollama`, `groq`, `anthropic`) share the same
reliability contract: every LLM call is requested in JSON mode, validated
against the same pydantic schemas, retried once on malformed output, and
degraded to the offline mock heuristic as a last resort for that single
call — a bad or slow model response never crashes a request.

## Architecture — how the graph satisfies the spec

```
START -> classify -> retrieve -> grade --sufficient--> reason -> END
                          ^           |
                          |     retry (< MAX_RETRIES)
                          |           v
                          +----- reformulate
                                      |
                              insufficient (retries exhausted)
                                      v
                             build_insufficient -> END
```

- **`classify`** — real conditional branch #1: routes retrieval strategy by
  **text vs. table vs. numerical** question type (not a fixed chain).
  - `numerical` → `search_documents` (vector) **and** `search_tables`
  - `table` → `search_tables` **and** `search_documents`
  - `text` → `search_documents` (vector) **and** `search_bm25`
    (the required non-vector retrieval path)
  All hits are deduplicated by `(document_id, page, section)` and ranked —
  reranking itself is `retrieval-api`'s responsibility per the spec; this
  service treats returned hits as already reranked.
- **`grade`** — real conditional branch #2: **sufficient vs. insufficient
  evidence**, with a bounded retry loop (`MAX_RETRIES`, default 2) that
  reformulates the query and re-retrieves before giving up.
- **`reason`** — real conditional branch #3, by extracted answer shape:
  - `calculated` → operands are extracted from evidence, then the
    **formula is executed by `app/calculator.py`'s safe AST evaluator**,
    never by the LLM's own arithmetic. An unsafe/invalid formula from the
    LLM is rejected and the graph falls back to `insufficient_evidence`
    rather than trust a bad calculation.
  - `direct` → single cited value.
  - `multi_span` → 2+ cited values.
  - anything the LLM can't ground → `insufficient_evidence`, never a guess.
- Every answer is validated locally against `app/schemas.py` (mirrors the
  spec's Strict Answer Schema) **before** it's returned — a schema
  violation is caught in-process and downgraded to `insufficient_evidence`
  rather than ever leaving this service malformed.
- Every step appends to `state["trace"]` (question classification,
  retrieval, reranked-hit counts, grading confidence, reformulations,
  reasoning) for `eval-service`/Langfuse-style observability.

## Tools exposed (`app/tools.py`)

| Tool | Purpose | Deterministic? |
|---|---|---|
| `search_documents(query)` | corpus-wide semantic (vector) retrieval | via retrieval-api |
| `search_bm25(query)` | non-vector lexical retrieval | via retrieval-api |
| `search_tables(query)` | table-aware retrieval | via retrieval-api |
| `filter_documents(metadata)` | direct metadata lookup | via retrieval-api |
| `calculate(expression)` | pure-arithmetic AST evaluator | **fully local, safe, deterministic** |

Also exposed as LangChain `@tool` objects (`TOOLS` list in `app/tools.py`)
if you want to wire a tool-calling agent executor in front of / alongside
the deterministic graph.

## Integration notes for the orchestrator owner

- Call `POST /answer {"question": "...", "document_id": null}` on this
  service.
- Per the architecture, **you** (orchestrator) are responsible for then
  POSTing `response.answer` to `answer-validator-api /validate_answer`
  before it reaches the user. This service already self-validates the same
  schema, but the validator is the system's single source of truth.
- If you want agent-service to *also* call the validator directly while
  you're testing it standalone, set `ANSWER_VALIDATOR_URL` — its verdict
  will be appended to the response `trace`.

## File map

```
agent-service/
├── app/
│   ├── main.py             FastAPI app: POST /answer, GET /health
│   ├── graph.py            LangGraph StateGraph (the actual "brain")
│   ├── llm.py              MockLLM (offline) + OllamaLLM (local qwen2.5) + AnthropicLLM (cloud) providers
│   ├── calculator.py       Safe AST arithmetic evaluator
│   ├── tools.py            search_documents / search_tables / calculate / filter_documents
│   ├── retrieval_client.py HTTP client for retrieval-api + offline mock corpus
│   ├── schemas.py          Pydantic models for the Strict Answer Schema
│   └── config.py           All env-driven settings
├── tests/test_graph.py     11 offline tests covering every answer type + edge cases
├── requirements.txt
├── .env.example
└── README.md
```

## Gemini provider

`LLM_PROVIDER=gemini` runs the agent on Google Gemini through the official
`google-genai` SDK, with the same prompts and failure handling as the
Anthropic provider. Set `GEMINI_API_KEY` (a Google AI Studio key) and
optionally `GEMINI_MODEL`, `GEMINI_TEMPERATURE`, `GEMINI_MAX_OUTPUT_TOKENS`
and `GEMINI_MAX_ATTEMPTS`. Rate limits are retried with backoff inside the
SDK; truncated or blocked output is reported as an error, not an answer.

## Enhancement switches

Each is off by default and reported by `/health`, so a saved run records
which were on. None needs re-indexing.

| setting | effect |
|---|---|
| `QUERY_DECOMPOSITION` | split multi-company or multi-item questions into separate searches |
| `ANSWER_FORMAT_FIXES` | keep scale words, keep page numbers out of answers and grading, prefer table figures, count only the year-on-year changes inside a period |
| `GRADE_COMPANY_CONTEXT` | accept a passage that names no company when nothing ties it to a different company |
| `GRADE_REQUIRE_NAMED_TABLE` | require evidence from a table or section the question names |
| `RANK_FUSION_MERGE` | merge a retrieve step's searches by rank instead of raw score |
