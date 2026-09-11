# LEDGER retrieval integration

This branch assembles the latest service branches needed for the retrieval
path. The integrated ports are:

- orchestrator: 8000
- document processor: 8001
- retrieval: 8002
- agent: 8003
- answer validator: 8004
- evaluation/observability: 8005

Start all seven services with one command:

```bash
docker compose up --build
```

Compose waits for services to become healthy before starting their dependants.
Retrieval indexes, Hugging Face models, and uploaded source PDFs use named
volumes. The service URLs are:

| Service | URL |
| --- | --- |
| Orchestrator | http://localhost:8000 |
| Document processor | http://localhost:8001 |
| Retrieval | http://localhost:8002 |
| Agent | http://localhost:8003 |
| Answer validator | http://localhost:8004 |
| Evaluation / tracing | http://localhost:8005 |
| UI | http://localhost:7860 |

Upload a PDF through `POST http://localhost:8000/documents/ingest`, then query
the agent/orchestrator. When OCR coordinates are available, the UI highlights
the cited region on the original PDF page. Reviewers can also use **Correct
OCR** to edit an indexed text/table extraction: the original value is audited,
the corrected chunk is re-embedded, and retrieval is rebuilt without rerunning
PDF preprocessing. Answer-level feedback remains separately available in Chat.

## Best measured configuration

Every enhancement switch is off by default, so `docker compose up` runs the
measured baseline. The best configuration (30/100 exact match, F1 0.402 on the
100 practice questions) turns on 17 agent switches, listed in
`docker-compose.best.yml`:

```bash
docker compose -f docker-compose.yml -f docker-compose.best.yml up --build -d
```

It needs a retrieval index built with company metadata; the same switches
scored 26/100 on the legacy index. Rebuild the index from the stored chunks,
with no OCR, inside the retrieval-api image:

```bash
python scripts/rebuild_index.py --data-dir /data build \
    --format contextual_text,company_metadata --output-dir /exp/idx-company
```

then serve the new directory as the `retrieval-data` volume. Results and
method: `eval-service/REINDEX_EXPERIMENT.md`.

A root `.env` (not in git) supplies the model and keys, for example
`LLM_PROVIDER=gemini`, `GEMINI_MODEL=gemini-3.5-flash-lite`, `GEMINI_API_KEYS`,
and `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_BASE_URL` for
tracing. Without it the agent runs in mock mode.

Service-specific retrieval contracts, calibration, benchmark commands, and
the TAT-DQA OCR pipeline are documented in `retrieval-api/README.md` and
`eval-service/README.md`.
