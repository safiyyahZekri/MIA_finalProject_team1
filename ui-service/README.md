# ui-service

Gradio front door: Chat, Upload, Documents, Correct OCR, and Dashboard tabs.
Ships in MOCK_MODE=true by default so it runs standalone before
orchestrator-api exists.

## Run

```
pip install -r requirements.txt
python app.py
```

## Switch to real orchestrator

```
ORCHESTRATOR_URL=http://localhost:8000
MOCK_MODE=false
```

The Upload tab sends a PDF through the processor and retrieval indexing flow.
Ask a question in Chat (optionally scoped to the returned `document_id`) to see
the cited retrieval bounding boxes highlighted on the original PDF page.

After an answer, open **Review or correct this answer**. A reviewer can mark it
correct or select **Needs correction**, enter the corrected answer, and add an
optional note. The exact original answer, question, optional document scope,
verdict, correction, timestamp, and review ID are stored by orchestrator-api.
Recent reviews are visible in the Dashboard tab.

Corrections are annotations for evaluation and later improvement. They do not
replace the validated answer, bypass `answer-validator-api`, or mutate the
retrieval index.

The separate **Correct OCR** tab implements human correction of extracted
fields. Enter a document ID, load its indexed text/table chunks, select one,
edit the extracted text, and identify the reviewer. Saving creates an immutable
audit entry and asks retrieval-api to re-embed the corrected chunk and rebuild
its BM25/FAISS indexes. It does not rerun PDF/OCR preprocessing.

Source PDFs are stored by orchestrator-api in the `source-documents` Docker
volume. Documents indexed before this feature remain searchable, but must be
uploaded through the orchestrator once before their pages can be rendered. In
Docker Compose, human reviews are stored in that persistent volume at
`/data/documents/reviews.jsonl`.
