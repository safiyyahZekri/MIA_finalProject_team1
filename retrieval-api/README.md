# LEDGER Retrieval API

Persistent retrieval for the structured JSON produced by `doc-processor-api`.
The service combines BGE dense search, BM25, reciprocal-rank fusion, and an
optional cross-encoder reranker. It listens on port **8002**; document
processing remains on 8001.

## Run

```bash
cd retrieval-api
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
python -m app
```

The embedding and reranker models load lazily. Lightweight tests use
deterministic model doubles and do not download model weights.

Important environment variables:

| Variable | Default |
| --- | --- |
| `LEDGER_RETRIEVAL_PORT` | `8002` |
| `LEDGER_RETRIEVAL_DATA_DIR` | `./data` |
| `LEDGER_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` |
| `LEDGER_RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| `LEDGER_MODEL_DEVICE` | `cpu` |
| `LEDGER_ENABLE_RERANKER` | `true` |
| `EVAL_SERVICE_URL` | empty (tracing disabled) |

## Indexing

`POST /documents` indexes or atomically replaces one document. Stable
identity precedence is `source_doc_uid`, explicit `document_id`, then the
processor-provided ID. The original filename and dataset UID are retained on
chunks and returned by search.

```json
{
  "document_id": "tatdqa-uid-1",
  "source_doc_uid": "tatdqa-uid-1",
  "source_filename": "annual-report.pdf",
  "metadata": {"company": "Example", "year": 2022},
  "document": {
    "document_id": "tatdqa-uid-1",
    "source_doc_uid": "tatdqa-uid-1",
    "original_filename": "annual-report.pdf",
    "pages": [{"bbox": [0, 0, 1000, 1400], "page_number": 1, "blocks": []}]
  }
}
```

`POST /documents/batch` accepts `{"documents": [<the envelopes above>]}`.
It chunks and embeds the whole batch, then rebuilds BM25/FAISS and persists
once. Reindexing the same canonical ID replaces its old chunks.

Other corpus routes are `GET /documents`, `DELETE /documents/{document_id}`,
and `GET /stats`.

## Search API

The following endpoints intentionally preserve the existing agent contract:

| Endpoint | Semantics |
| --- | --- |
| `POST /search` | Dense + BM25 + RRF; optional reranking |
| `POST /search/vector` | Dense only; optional reranking |
| `POST /search/bm25` | BM25 only; optional reranking |
| `POST /search/tables` | Table-only retrieval using the requested mode |
| `POST /filter` | Deterministic metadata/document lookup, without query scoring |

Both legacy flat filters and canonical nested filters are accepted:

```json
{"query": "operating income 2022", "top_k": 5, "document_id": "tatdqa-uid-1"}
```

```json
{
  "query": "operating income 2022",
  "top_k": 5,
  "candidate_k": 30,
  "rerank": true,
  "filters": {"document_id": "tatdqa-uid-1", "content_type": "table"}
}
```

`document_id` matches either the canonical ID or retained `source_doc_uid`.
Other filters are `source_doc_uid`, `filename`/`source_filename`, `page`,
`section`, public `content_type` (`text` or `table`), and exact-match custom
metadata. Conflicting flat and nested values produce HTTP 422 instead of being
silently ignored.

Every hit has the agent-facing contract:

```json
{
  "chunk_id": "chunk-123",
  "document_id": "tatdqa-uid-1",
  "source_doc_uid": "tatdqa-uid-1",
  "filename": "annual-report.pdf",
  "content": "Operating income | 2022 | 125 million",
  "text": "Operating income | 2022 | 125 million",
  "content_type": "table",
  "page": 4,
  "score": 0.82,
  "scores": {"dense": 0.71, "lexical": 4.23, "fusion": 0.026, "reranker": 7.1},
  "metadata": {}
}
```

`score` is always bounded to `[0, 1]`. It is a deterministic confidence-like
ranking score, not a probability: dense cosine uses a centered logistic
transform; BM25 combines within-query relative score with query-token coverage;
hybrid combines dense, lexical, and normalized RRF signals; reranked results
blend that base score with a shifted sigmoid of the cross-encoder logit. Raw
stage values remain in `scores` for diagnosis and evaluation.

Financial tables retain their cells, spans, inferred multi-row headers, title,
nearby heading, units/currency/scale text, period labels, and row labels. Their
deterministic serialization is shared by dense and lexical indexing.

## Observability and persistence

When `EVAL_SERVICE_URL` is set, each search sends best-effort trace events for
filtering, dense search, BM25, fusion, reranking, and final results, including
candidate counts and per-stage latency. No full chunk/document bodies are sent.
Tracing timeouts or outages are logged and never fail retrieval.

The chunk store, vectors, FAISS index, and embedding-model manifest are written
under `LEDGER_RETRIEVAL_DATA_DIR` and reloaded on restart. Use one API worker per
data directory; the lock coordinates threads within that worker, not multiple
process writers.

## Test

```bash
pip install -r requirements-ci.txt
ruff check app tests
pytest -q
```

See `eval-service/README.md` for direct retrieval metrics, ablations, and the
real TAT-DQA PDF -> OCR -> batch index -> benchmark command.
