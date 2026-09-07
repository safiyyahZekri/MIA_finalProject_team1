# LEDGER Retrieval API

Hybrid, corpus-wide retrieval for the structured JSON emitted by LEDGER's
`doc-processor-api`.

## What this service implements

- Section-aware paragraph chunks grouped on detected headings and block boundaries.
- Table-aware chunks split only between rows, with the header repeated when a table
  is too large for one chunk.
- Parent context plus citation metadata on every chunk: `document_id`, source
  filename, page, section, content type, bounding box, and source block UUIDs.
- Dense retrieval with FAISS and `BAAI/bge-small-en-v1.5` embeddings.
- BM25 lexical retrieval, including number normalization for values such as `9,447`.
- Reciprocal-rank fusion followed by optional
  `cross-encoder/ms-marco-MiniLM-L-6-v2` reranking.
- Direct table-only retrieval and metadata filters.
- On-disk FAISS/chunk persistence. Re-indexing a `document_id` replaces its prior chunks.

The models are loaded lazily on the first indexing/search request. `/health`,
`/stats`, and `/documents` do not download model weights.

## Run locally

```bash
cd retrieval-api
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --host 0.0.0.0 --port 8001
```

Configuration is through environment variables:

| Variable | Default |
| --- | --- |
| `LEDGER_RETRIEVAL_DATA_DIR` | `./data` |
| `LEDGER_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` |
| `LEDGER_RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| `LEDGER_MODEL_DEVICE` | `cpu` |
| `LEDGER_CHUNK_MAX_CHARS` | `1800` |
| `LEDGER_CHUNK_OVERLAP_BLOCKS` | `1` |
| `LEDGER_ENABLE_RERANKER` | `true` |

Use `LEDGER_MODEL_DEVICE=cuda` when a CUDA-enabled PyTorch environment is
available.

## API contract

Index the exact document object returned by `doc-processor-api`, while attaching
the original filename and any useful filters:

```bash
curl -X POST http://localhost:8001/documents \
  -H 'content-type: application/json' \
  -d '{
    "document": {
      "document_id": "doc-041",
      "pages": [{"bbox":[0,0,1000,1400],"page_number":1,"blocks":[]}]
    },
    "source_filename": "cts-corporation_2019.pdf",
    "metadata": {"company":"CTS Corporation","year":2019}
  }'
```

Corpus-wide hybrid search over 30 candidates, reranked to 5:

```bash
curl -X POST http://localhost:8001/search \
  -H 'content-type: application/json' \
  -d '{
    "query":"What was CTS finished-goods balance in 2019?",
    "candidate_k":30,
    "top_k":5,
    "mode":"hybrid",
    "rerank":true
  }'
```

`POST /search/tables` accepts the same body and forces a table-content filter.
`mode` can be `hybrid`, `dense`, or `lexical`; the latter two enable controlled
retrieval experiments. Optional filters are `document_id`, `source_filename`,
`page`, `section`, `content_type`, and exact-match custom `metadata`.

Each hit returns all evidence needed downstream, including:

```json
{
  "document_id": "doc-041",
  "source_filename": "cts-corporation_2019.pdf",
  "page": 1,
  "section": "Inventory",
  "content_type": "table",
  "bbox": [10, 120, 900, 400],
  "source_block_ids": ["6849600d-a2dd-45be-b923-2b314f7ccbdc"],
  "scores": {
    "dense": 0.81,
    "lexical": 5.42,
    "fusion": 0.016,
    "reranker": 7.13
  }
}
```

Other routes: `GET /health`, `GET /stats`, `GET /documents`, and
`DELETE /documents/{document_id}`.

## Test

```bash
pytest -q
```

Tests use deterministic model doubles and do not download pretrained weights.

Run a single API worker against a data directory. The in-process write lock protects
concurrent requests within that worker; shared multi-process writers are outside this
service's current scope.

