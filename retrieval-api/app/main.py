from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request, status

from .chunking import ChunkingConfig
from .config import Settings
from .engine import RetrievalEngine
from .models import (
    CorpusStats,
    DocumentSummary,
    IndexDocumentRequest,
    IndexResponse,
    SearchRequest,
    SearchResponse,
)
from .models_runtime import CrossEncoderReranker, SentenceTransformerEmbedder


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


def create_app(engine: RetrievalEngine | None = None) -> FastAPI:
    application = FastAPI(
        title="LEDGER Retrieval API",
        version="0.1.0",
        description="Hybrid, reranked retrieval over structured financial documents.",
    )
    settings = Settings()
    application.state.engine = engine or build_default_engine(settings)

    @application.get("/health")
    def health() -> dict[str, str]:
        return {"status": "running"}

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
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @application.delete("/documents/{document_id}")
    def delete_document(document_id: str, request: Request) -> dict[str, int | str]:
        removed = request.app.state.engine.delete_document(document_id)
        if not removed:
            raise HTTPException(status_code=404, detail="document not found")
        return {"document_id": document_id, "chunks_removed": removed}

    @application.post("/search", response_model=SearchResponse)
    def search(payload: SearchRequest, request: Request) -> SearchResponse:
        try:
            return request.app.state.engine.search(payload)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @application.post("/search/tables", response_model=SearchResponse)
    def search_tables(payload: SearchRequest, request: Request) -> SearchResponse:
        table_payload = payload.model_copy(deep=True)
        table_payload.filters.content_type = "table"
        try:
            return request.app.state.engine.search(table_payload)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return application


app = create_app()
