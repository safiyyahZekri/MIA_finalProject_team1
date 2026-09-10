from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request, status

from .chunking import ChunkingConfig
from .config import Settings
from .engine import RetrievalEngine
from .models import (
    BatchIndexRequest,
    BatchIndexResponse,
    CorpusStats,
    DocumentSummary,
    ExtractedField,
    ExtractionCorrectionRecord,
    ExtractionCorrectionRequest,
    FilterRequest,
    IndexDocumentRequest,
    IndexResponse,
    SearchMode,
    SearchRequest,
    SearchResponse,
)
from .models_runtime import CrossEncoderReranker, SentenceTransformerEmbedder
from .tracing import RetrievalTracer


def build_default_engine(settings: Settings) -> RetrievalEngine:
    reranker = (
        CrossEncoderReranker(settings.reranker_model, settings.device)
        if settings.enable_reranker
        else None
    )
    return RetrievalEngine(
        data_dir=settings.data_dir,
        embedder=SentenceTransformerEmbedder(settings.embedding_model, settings.device),
        reranker=reranker,
        chunking=ChunkingConfig(
            max_chars=settings.chunk_max_chars,
            overlap_blocks=settings.chunk_overlap_blocks,
        ),
    )


def _service_error(exc: Exception) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={"code": "retrieval_unavailable", "message": str(exc)},
    )


def create_app(
    engine: RetrievalEngine | None = None, tracer: RetrievalTracer | None = None
) -> FastAPI:
    application = FastAPI(
        title="LEDGER Retrieval API",
        version="1.0.0",
        description="Persistent hybrid and reranked financial-document retrieval.",
    )
    settings = Settings()
    application.state.engine = engine or build_default_engine(settings)
    application.state.tracer = tracer or RetrievalTracer(
        settings.eval_service_url, settings.tracing_timeout_seconds
    )

    @application.on_event("startup")
    def _warmup_models() -> None:
        # Blocks Uvicorn from accepting connections until the embedder/
        # reranker are actually loaded, so a healthy container is truly
        # ready to serve a fast search rather than reporting healthy
        # before ever having loaded its models.
        application.state.engine.warmup()

    @application.get("/health")
    def health(request: Request) -> dict[str, str | int | bool]:
        corpus = request.app.state.engine.stats()
        return {
            "status": "running",
            "port": settings.port,
            "indexed_chunks": corpus.chunks,
            "tracing_enabled": request.app.state.tracer.enabled,
        }

    @application.get("/stats", response_model=CorpusStats)
    def stats(request: Request) -> CorpusStats:
        return request.app.state.engine.stats()

    @application.get("/documents", response_model=list[DocumentSummary])
    def documents(request: Request) -> list[DocumentSummary]:
        return request.app.state.engine.documents()

    @application.post(
        "/documents", response_model=IndexResponse, status_code=status.HTTP_201_CREATED
    )
    def index_document(
        payload: IndexDocumentRequest, request: Request
    ) -> IndexResponse:
        try:
            return request.app.state.engine.index_document(payload)
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail={"code": "invalid_document", "message": str(exc)},
            ) from exc
        except (RuntimeError, OSError) as exc:
            raise _service_error(exc) from exc

    @application.post(
        "/documents/batch",
        response_model=BatchIndexResponse,
        status_code=status.HTTP_201_CREATED,
    )
    def index_documents(
        payload: BatchIndexRequest, request: Request
    ) -> BatchIndexResponse:
        try:
            return request.app.state.engine.index_documents(payload.documents)
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail={"code": "invalid_document_batch", "message": str(exc)},
            ) from exc
        except (RuntimeError, OSError) as exc:
            raise _service_error(exc) from exc

    @application.delete("/documents/{document_id}")
    def delete_document(document_id: str, request: Request) -> dict[str, int | str]:
        removed = request.app.state.engine.delete_document(document_id)
        if not removed:
            raise HTTPException(
                status_code=404,
                detail={"code": "document_not_found", "message": "document not found"},
            )
        return {"document_id": document_id, "chunks_removed": removed}

    @application.get(
        "/documents/{document_id}/chunks", response_model=list[ExtractedField]
    )
    def document_chunks(document_id: str, request: Request) -> list[ExtractedField]:
        try:
            return request.app.state.engine.document_chunks(document_id)
        except KeyError as exc:
            raise HTTPException(
                status_code=404,
                detail={"code": "document_not_found", "message": str(exc.args[0])},
            ) from exc

    @application.post(
        "/documents/{document_id}/corrections",
        response_model=ExtractionCorrectionRecord,
    )
    def correct_extracted_field(
        document_id: str,
        payload: ExtractionCorrectionRequest,
        request: Request,
    ) -> ExtractionCorrectionRecord:
        try:
            return request.app.state.engine.correct_extracted_field(document_id, payload)
        except KeyError as exc:
            raise HTTPException(
                status_code=404,
                detail={"code": "extracted_field_not_found", "message": str(exc.args[0])},
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail={"code": "invalid_correction", "message": str(exc)},
            ) from exc
        except (RuntimeError, OSError) as exc:
            raise _service_error(exc) from exc

    @application.get(
        "/documents/{document_id}/corrections",
        response_model=list[ExtractionCorrectionRecord],
    )
    def extraction_corrections(
        document_id: str, request: Request
    ) -> list[ExtractionCorrectionRecord]:
        try:
            return request.app.state.engine.corrections(document_id)
        except KeyError as exc:
            raise HTTPException(
                status_code=404,
                detail={"code": "document_not_found", "message": str(exc.args[0])},
            ) from exc
        except (RuntimeError, OSError) as exc:
            raise _service_error(exc) from exc

    def execute_search(payload: SearchRequest, request: Request) -> SearchResponse:
        try:
            response = request.app.state.engine.search(payload)
            response.trace_id = request.app.state.tracer.record(payload, response)
            return response
        except (RuntimeError, OSError) as exc:
            raise _service_error(exc) from exc

    @application.post("/search", response_model=SearchResponse)
    def search(payload: SearchRequest, request: Request) -> SearchResponse:
        payload.mode = SearchMode.hybrid
        return execute_search(payload, request)

    @application.post("/search/vector", response_model=SearchResponse)
    def search_vector(payload: SearchRequest, request: Request) -> SearchResponse:
        payload.mode = SearchMode.dense
        return execute_search(payload, request)

    @application.post("/search/bm25", response_model=SearchResponse)
    def search_bm25(payload: SearchRequest, request: Request) -> SearchResponse:
        payload.mode = SearchMode.lexical
        return execute_search(payload, request)

    @application.post("/search/tables", response_model=SearchResponse)
    def search_tables(payload: SearchRequest, request: Request) -> SearchResponse:
        payload.filters.content_type = "table"
        return execute_search(payload, request)

    @application.post("/filter", response_model=SearchResponse)
    def filter_documents(payload: FilterRequest, request: Request) -> SearchResponse:
        try:
            return request.app.state.engine.filter(payload)
        except (RuntimeError, OSError) as exc:
            raise _service_error(exc) from exc

    return application


app = create_app()
