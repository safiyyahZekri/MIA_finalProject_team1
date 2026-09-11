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

Service-specific retrieval contracts, calibration, benchmark commands, and
the TAT-DQA OCR pipeline are documented in `retrieval-api/README.md` and
`eval-service/README.md`.
