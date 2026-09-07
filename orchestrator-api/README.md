# orchestrator-api

Central router: UI -> agent-service -> answer-validator-api -> UI.
Ships in MOCK_MODE=true by default so it runs standalone before the other
services exist.

## Run

```
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## Switch to real services

Set these env vars and MOCK_MODE=false:

```
AGENT_SERVICE_URL=http://localhost:8003
VALIDATOR_SERVICE_URL=http://localhost:8004
DOC_PROCESSOR_URL=http://localhost:8001
RETRIEVAL_SERVICE_URL=http://localhost:8002
MOCK_MODE=false
```

## Endpoints

- `POST /ask` `{question, document_id?}` -> strict answer schema + validator result
- `POST /documents/ingest` multipart PDF -> document processor -> retrieval index
- `GET /documents` -> indexed document list (proxied from retrieval-api)
- `GET /recent_queries` -> last 20 queries with latency, for the UI dashboard
- `GET /health`

The ingestion endpoint accepts multipart fields `file`, optional
`source_doc_uid`, and optional JSON-object string `metadata_json`. Dataset UIDs
become canonical IDs. Generic uploads use `sha256-<pdf-content-hash>`, so retries
replace the same retrieval document instead of creating random duplicates.
Processor and index errors are returned as structured HTTP 502 responses with
the failing stage. Non-PDF uploads return 415.

```bash
curl -X POST http://localhost:8000/documents/ingest \
  -F file=@annual-report.pdf \
  -F source_doc_uid=tatdqa-source-id \
  -F 'metadata_json={"company":"Example"}'
```
