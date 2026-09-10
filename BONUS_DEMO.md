# Bonus demo: deployment, cited-PDF highlighting, and human correction

These features only count as bonus work when their effect is demonstrated.
The checks below exercise the implemented behavior rather than only showing
source code.

## 1. Seven-service Docker Compose deployment

Build and start the complete system:

```bash
docker compose up --build -d
docker compose ps
```

`docker compose ps` should show these seven services as running/healthy:

```text
orchestrator-api       8000
doc-processor-api      8001
retrieval-api          8002
agent-service          8003
answer-validator-api   8004
eval-service           8005
ui-service             7860
```

Verify every API health endpoint:

```bash
curl --fail http://localhost:8000/health
curl --fail http://localhost:8001/
curl --fail http://localhost:8002/health
curl --fail http://localhost:8003/health
curl --fail http://localhost:8004/health
curl --fail http://localhost:8005/health
curl --fail http://localhost:7860/
```

The `retrieval-data`, `model-cache`, and `source-documents` named volumes keep
the index, model downloads, and source PDFs across container restarts.

## 2. Bounding-box highlighting on the source PDF

1. Open `http://localhost:7860`.
2. In **Upload**, select a financial-report PDF and click **Process and index**.
3. Copy the returned `document_id`.
4. In **Chat**, paste the ID into the optional document scope.
5. Ask a question whose evidence appears in the uploaded report.
6. Confirm the answer includes a document/page citation and the cited region is
   shown with a yellow fill and red border on the original PDF page.

For an API-level check, the answer evidence now includes OCR coordinates:

```json
{
  "document_id": "sha256-...",
  "page": 2,
  "section": "Revenue",
  "bbox": [106, 412, 1098, 678]
}
```

The coordinates originate in `doc-processor-api`, are retained in retrieval
chunks, survive agent citation selection and validation, and are finally drawn
by `ui-service`. Older documents already in a remote retrieval index remain
searchable, but their original PDFs must also exist in this orchestrator's
`source-documents` volume before the UI can render their pages.

## 3. Human-in-the-loop correction of extracted fields

This flow corrects an OCR/table extraction already in the retrieval index. It
does **not** run document processing again.

1. Open **Correct OCR** and paste an indexed `document_id`.
2. Click **Load extracted fields** to see the indexed text and table chunks.
3. Choose a chunk, correct the bad value in **Corrected extracted text**, enter
   the reviewer name, and click **Save correction and re-index**.
4. Confirm the correction ID appears and the audit history records who made the
   change and when.
5. Search or ask the original question again and verify the corrected value is
   now retrievable.

API-level demonstration:

```bash
curl --fail http://localhost:8000/documents/DOCUMENT_ID/chunks

curl --fail -X POST http://localhost:8000/documents/DOCUMENT_ID/corrections \
  -H 'Content-Type: application/json' \
  -d '{
    "chunk_id": "CHUNK_ID_FROM_THE_FIRST_COMMAND",
    "corrected_text": "Corrected extracted table or paragraph text",
    "corrected_by": "TEAM_MEMBER_NAME",
    "comment": "Verified against the source PDF"
  }'

curl --fail http://localhost:8000/documents/DOCUMENT_ID/corrections
```

The retrieval service stores `original_text`, `corrected_text`, reviewer,
comment, timestamp, and correction ID in `corrections.jsonl`. It embeds the
corrected chunk, rebuilds BM25/FAISS, and persists the updated index. The
original value remains in the audit record rather than being lost.

## 4. Optional answer feedback

1. Ask any question in the **Chat** tab and wait for its validated answer.
2. Open **Review or correct this answer** beneath the evidence display.
3. Select **Correct** and save, or select **Needs correction**, enter the human
   answer, optionally add a note, and save.
4. Confirm the UI displays the generated review ID.
5. Open **Dashboard**, click **Refresh**, and confirm the review appears with
   its verdict, question, correction, and timestamp.

Verify persistence through the API:

```bash
curl --fail http://localhost:8000/reviews
```

Or submit a correction directly:

```bash
curl --fail -X POST http://localhost:8000/reviews \
  -H 'Content-Type: application/json' \
  -d '{
    "question": "What was operating income in 2022?",
    "verdict": "needs_correction",
    "original_answer": {
      "answer_type": "direct",
      "params": {"value": "$40 million"},
      "evidence": [{"document_id": "sha256-example", "page": 4}],
      "valid": true
    },
    "corrected_answer": "$42 million",
    "comment": "The cited row shows the corrected amount.",
    "document_id": "sha256-example"
  }'
```

Restart the UI and orchestrator containers, then call `GET /reviews` again to
show that the annotation survived. Answer reviews are append-only feedback and
do not bypass validation. They are separate from extracted-field corrections,
which explicitly update retrieval through the audited correction endpoint.
