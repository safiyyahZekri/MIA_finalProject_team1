# LEDGER — Financial Document Intelligence Agent

LEDGER answers natural-language questions across a collection of financial
report PDFs (the TAT-DQA dataset). It reads the raw PDFs with a deep-learning
OCR model, retrieves evidence across the whole corpus, computes any arithmetic
with a calculator tool, checks every answer against a strict schema, and shows
the source document and page, highlighted, next to the answer. When the
evidence is not there it says so instead of guessing.

## Quick start

For someone who already has Docker running and the two index files from the
[team's shared drive](https://drive.google.com/drive/folders/1_mP7CP3orWL5QyUMOhaAOol08TDqFyYy) (`ledger-index-company.tgz`, `ledger-source-pdfs.tgz`).
Run these in Git Bash (Windows) or a terminal (macOS/Linux):

```bash
git clone https://github.com/safiyyahZekri/MIA_finalProject_team1.git
cd MIA_finalProject_team1
git checkout feature/retrieval-integration        # or stay on main once PR #1 is merged
cp .env.example .env                              # then open .env and add your Gemini keys
docker compose build                              # first time only, 10–20 minutes
bash scripts/import_index.sh ~/Downloads/ledger-index-company.tgz ~/Downloads/ledger-source-pdfs.tgz
docker compose -f docker-compose.yml -f docker-compose.best.yml up -d
```

Wait until `docker compose ps` shows all seven services as `healthy`, then open
<http://localhost:7860>. Each step is explained below.

## Services

| Service | Port | What it does |
|---|---|---|
| `ui-service` | 7860 | Gradio UI: chat, upload, documents, OCR correction, dashboard |
| `orchestrator-api` | 8000 | Routes documents and questions, sends answers to the validator, answer cache |
| `doc-processor-api` | 8001 | docTR OCR and img2table table extraction from raw PDFs |
| `retrieval-api` | 8002 | FAISS dense search, BM25, table search, metadata filter, reranker |
| `agent-service` | 8003 | LangGraph agent: classify, retrieve, grade, retry, calculate, answer |
| `answer-validator-api` | 8004 | Strict answer schema; logs `[ANSWER-VALIDATOR-SUCCESS]` / `[ANSWER-VALIDATOR-ERROR]` |
| `eval-service` | 8005 | Langfuse tracing and the automated benchmark |

All seven run in Docker. When Docker is closed, LEDGER and its UI are not
running.

## 1. Requirements

- **Docker Desktop** (Windows, macOS) or Docker Engine with Compose v2 (Linux)
- At least **8 GB of memory for Docker** (Docker Desktop → Settings → Resources)
  and about **12 GB of free disk** for the images and models
- **Internet** for the first build and the first start (Python packages, and the
  OCR, embedding and reranker models)
- Free ports **7860** and **8000–8005**
- Python 3.12 only if you run the evaluation (step 8)

## 2. Get the code

```bash
git clone https://github.com/safiyyahZekri/MIA_finalProject_team1.git
cd MIA_finalProject_team1
git checkout feature/retrieval-integration   # or stay on main once PR #1 is merged
```

## 3. Create `.env`

```bash
cp .env.example .env
```

Open `.env` in a text editor and fill in your own keys. `.env` is git-ignored;
never commit it.

| Variable | Needed? | Value |
|---|---|---|
| `LLM_PROVIDER` | yes | `gemini` |
| `GEMINI_MODEL` | yes | `gemini-3.5-flash-lite` |
| `GEMINI_API_KEYS` | yes | one or more Google AI Studio keys, separated by commas |
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL` | only for tracing | your Langfuse project keys |

- Without a model key the agent runs in **mock mode** and its answers are canned.
- Gemini keys have a **daily quota**. One question uses about 5 model calls. Keys
  from different Google projects add up; keys from the same project share one quota.

## 4. Build the images

```bash
docker compose build
```

Needed once, and again after pulling code changes. The first build takes 10–20
minutes.

## 5. Load the document index

A fresh clone has an **empty index**: the UI shows 0 documents and every
question returns "not enough evidence". Load an index before starting.

### Option A — the prepared index (recommended, a few seconds)

1. Download these files from the [team's shared drive](https://drive.google.com/drive/folders/1_mP7CP3orWL5QyUMOhaAOol08TDqFyYy) into one folder:

   | File | Contents |
   |---|---|
   | `ledger-index-company.tgz` (50 MB) | 2,446 TAT-DQA documents, 14,386 chunks, with company metadata |
   | `ledger-source-pdfs.tgz` (3 MB) | original PDFs of the demo documents, used to highlight evidence on the page |
   | `SHA256SUMS.txt` | checksums |

2. Optionally check the download, from that folder: `sha256sum -c SHA256SUMS.txt`
   (both lines should say `OK`).
3. From the repository root, load both files (use your own paths):

   ```bash
   bash scripts/import_index.sh ~/Downloads/ledger-index-company.tgz ~/Downloads/ledger-source-pdfs.tgz
   ```

   It prints the index manifest (`"chunks": 14386`) and `Source PDFs loaded: 7`.
   Running it again replaces what was loaded before.

### Option B — build the index from the PDFs (slow)

This runs OCR on every PDF: about 20 seconds per page on a CPU, roughly 15
hours for the whole corpus.

1. Download `tatdqa_docs_train.zip`, `tatdqa_docs_dev.zip` and
   `tatdqa_docs_test.zip` from <https://huggingface.co/datasets/next-tat/TAT-DQA>
   and extract them.
2. In `.env`, set `LEDGER_INDEX_FORMAT=contextual_text,company_metadata` so the
   new index gets company metadata.
3. Start LEDGER (step 6), then from the repository root:

   ```bash
   pip install -r eval-service/requirements.txt
   python eval-service/scripts/tatdqa_retrieval_pipeline.py \
     --pdf-dir path/to/extracted/pdfs \
     --questions eval-service/questions_setA_practice.json \
     --doc-processor-url http://localhost:8001 \
     --retrieval-url http://localhost:8002
   ```

   It skips documents that are already indexed, so it can be stopped and run
   again.

An index built earlier without company metadata can be upgraded from its
stored chunks, with no OCR, using `retrieval-api/scripts/rebuild_index.py`; see
`eval-service/REINDEX_EXPERIMENT.md`.

## 6. Start LEDGER

With the **best measured configuration** (30/100 exact match):

```bash
docker compose -f docker-compose.yml -f docker-compose.best.yml up -d
```

Or with every enhancement switch off (the measured baseline):

```bash
docker compose up -d
```

The first start downloads models, and `doc-processor-api` can take up to 5
minutes to become healthy. Check that everything is up:

| Command | Expected |
|---|---|
| `docker compose ps` | all seven services `healthy` |
| `curl http://localhost:8002/health` | `"indexed_chunks": 14386`, `"index_format": "contextual_text,company_metadata"` |
| `curl http://localhost:8003/health` | `"llm_provider": "gemini"` |

### Stopping and restarting

- `docker compose down` stops LEDGER. Closing Docker Desktop stops it too.
- The index, uploaded PDFs, corrections, reviews and downloaded models are kept
  in Docker volumes and survive both.
- The answer cache and the dashboard's recent questions are in memory and start
  empty after a restart.
- Nothing restarts by itself: after reopening Docker Desktop, run the same `up`
  command from step 6 again.

## 7. Use the UI

Open <http://localhost:7860>.

| Tab | What to do |
|---|---|
| **Chat** | Ask about any company in the corpus without naming a document. The answer shows its citation (document and page), the formula for calculations, and the cited page with the evidence highlighted when that document's PDF is available. It also says whether the answer came from the agent (about 15 s) or from the cache (milliseconds, for a repeated or reworded question). **Review or correct this answer** records human feedback. |
| **Upload** | Upload a financial-report PDF: it is OCR'd, chunked and indexed, and the PDF is kept for highlighting. |
| **Documents** | Every indexed document with its pages and detected tables. |
| **Correct OCR** | Paste a document ID, load its extracted text and table chunks, fix a wrong value and re-index it. Each correction is audited. |
| **Dashboard** | Indexed documents, detected tables, recent questions with latency and cache hits, human reviews. |

Questions that answered correctly in every evaluation run, and whose PDFs are
in the prepared archive (so the evidence is highlighted):

| Question | Answer |
|---|---|
| By how much did Microchip Technology Inc.'s net sales change between 2016 and 2017? | 1234.5 (`3407.8-2173.3`) |
| What was Activision Blizzard, Inc.'s amount of cash and cash equivalents in 2018? | 4,225 |
| What is iSelect Limited's payroll tax for 2019 and 2018 respectively? | 2,657 and 3,035 |

If an answer comes back as "not enough evidence", ask again: the model varies
between runs, and declines are not cached.

## 8. Run the evaluation

With LEDGER running, from the `eval-service/` folder:

```bash
pip install -r requirements.txt

# Answers: exact match, F1, numerical accuracy, latency, model calls, tokens
python scripts/run_answer_eval.py --ids A034,A039 --label my-check --out-dir results/answer-eval/my-check
python scripts/run_answer_eval.py --all --label my-run --out-dir results/answer-eval/my-run
# After an interruption or a used-up quota, run the same command with --resume

# Retrieval: recall@k, precision@k and MRR for dense, BM25, hybrid, hybrid + reranker
python -m app.retrieval_benchmark --retrieval-url http://localhost:8002 \
  --questions questions_setA_practice.json --output-dir results/retrieval-check
```

A full answer run uses about 500 model calls. Every benchmarked question is
traced to Langfuse when its keys are set.

Measured results on the 100 practice questions:

| Configuration | Exact match | F1 |
|---|---:|---:|
| Gemini 3.5 Flash Lite, all switches off | 19 | 0.22 |
| + 16 enhancement switches | 26 | 0.35 |
| + company-metadata index and evidence company label (`docker-compose.best.yml`) | **30** | **0.40** |

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| <http://localhost:7860> does not load | Docker is closed or LEDGER is stopped: open Docker Desktop, run the `up` command from step 6 |
| UI shows 0 documents; every answer is "not enough evidence" | No index loaded: step 5 |
| Answers look canned; `curl http://localhost:8003/health` shows `"llm_provider": "mock"` | `.env` missing or `LLM_PROVIDER` not set: step 3, then the `up` command again |
| Agent log shows `429` or "used its daily quota" | The Gemini keys are used up for today: add keys from another Google project to `GEMINI_API_KEYS`, or wait until tomorrow |
| "The citation is valid, but its source PDF could not be rendered" | That document's PDF is not stored: upload it in **Upload**, or load `ledger-source-pdfs.tgz` |
| retrieval-api fails with "format switches this retrieval-api does not know" | The image is older than the index: `docker compose build retrieval-api` |
| Answers time out while something heavy runs | OCR or index rebuilds use all CPU cores: do not run them during an evaluation or a demo |

## Documentation

| File | Contents |
|---|---|
| `eval-service/FAILURE_ANALYSIS_E2E.md` | Ten failed answers, each traced to the pipeline stage that caused it |
| `eval-service/REINDEX_EXPERIMENT.md` | Company metadata, contextual text and table titles: method and results |
| `eval-service/EXPERIMENTS.md` | Answer-level experiments and the first full evaluation |
| `eval-service/BASELINE.md` | Retrieval baseline on the full corpus |
| `eval-service/CHUNK_BUDGET_EXPERIMENT.md` | Table chunk size experiment |
| `BONUS_DEMO.md` | Demonstrating the bonus features |
| `retrieval-api/README.md`, `eval-service/README.md` | Service contracts and benchmark details |
