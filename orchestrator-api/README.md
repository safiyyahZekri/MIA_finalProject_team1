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
DOCUMENT_STORE_DIR=data/documents
# Optional; defaults to DOCUMENT_STORE_DIR/reviews.jsonl
REVIEW_STORE_PATH=data/documents/reviews.jsonl
MOCK_MODE=false
```

## Endpoints

- `POST /ask` `{question, document_id?}` -> strict answer schema + validator result
- `POST /documents/ingest` multipart PDF -> document processor -> retrieval index
- `GET /documents` -> indexed document list (proxied from retrieval-api)
- `GET /documents/{document_id}/chunks` -> extracted text/table chunks for review
- `POST /documents/{document_id}/corrections` -> audit, apply, and re-index a correction
- `GET /documents/{document_id}/corrections` -> correction history
- `GET /recent_queries` -> last 20 queries with latency, for the UI dashboard
- `POST /reviews` -> persist a correct/needs-correction human review
- `GET /reviews?limit=20` -> newest human reviews for the UI dashboard
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

Human reviews are append-only annotations containing the question, original
validated answer, verdict, optional correction/comment, and timestamp. A
`needs_correction` review requires `corrected_answer`. Recording feedback does
not change the agent answer or retrieval index.

Extracted-field corrections are different: orchestrator proxies them to
retrieval-api, which preserves the original/current values in an audit record,
re-embeds the corrected chunk, rebuilds its indexes, and persists both changes.

## Evidence boxes

`/ask` returns the validated answer unchanged, with citations holding only
`document_id`, `page` and `section` as the Strict Answer Schema requires.
Highlight coordinates come beside it in `evidence_boxes`: every indexed
chunk on a cited page, narrowed to the cited section when there is one,
looked up from retrieval-api's `/documents/{document_id}/chunks`. A failed
lookup returns no boxes and never fails the answer.
