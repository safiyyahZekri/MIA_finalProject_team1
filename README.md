# LEDGER retrieval integration

This branch assembles the latest service branches needed for the retrieval
path. The integrated ports are:

- orchestrator: 8000
- document processor: 8001
- retrieval: 8002
- agent: 8003
- answer validator: 8004
- evaluation/observability: 8005

Start the stack with `docker compose up --build`. Upload a PDF through
`POST http://localhost:8000/documents/ingest`, then query the agent/orchestrator.
Retrieval data and the Hugging Face model cache use named Docker volumes.

Service-specific retrieval contracts, calibration, benchmark commands, and
the TAT-DQA OCR pipeline are documented in `retrieval-api/README.md` and
`eval-service/README.md`.
