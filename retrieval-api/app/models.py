from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
BBox = tuple[int, int, int, int]
MetadataValue = str | int | float | bool | None
# The section of every chunk that comes before the document's first heading.
DEFAULT_SECTION = "Document"


class WordData(BaseModel):
    model_config = ConfigDict(extra="ignore")
    word_list: list[str] = Field(default_factory=list)
    bbox_list: list[list[int]] = Field(default_factory=list)


class Cell(BaseModel):
    model_config = ConfigDict(extra="ignore")
    bbox: list[int] = Field(min_length=4, max_length=4)
    text: str = ""
    row_span: list[int] = Field(min_length=2, max_length=2)
    col_span: list[int] = Field(min_length=2, max_length=2)


class DocumentBlock(BaseModel):
    """Structured block accepted from doc-processor-api."""

    model_config = ConfigDict(extra="ignore")
    bbox: list[int] = Field(min_length=4, max_length=4)
    uuid: NonEmptyString
    text: str = ""
    order: int = Field(ge=1)
    content_type: Literal["paragraph", "text", "heading", "table"]
    words: WordData | None = None
    cells: list[Cell] = Field(default_factory=list)


class Page(BaseModel):
    model_config = ConfigDict(extra="ignore")
    bbox: list[int] = Field(min_length=4, max_length=4)
    blocks: list[DocumentBlock] = Field(default_factory=list)
    page_number: int = Field(ge=0)


class ProcessedDocument(BaseModel):
    model_config = ConfigDict(extra="ignore")
    pages: list[Page]
    document_id: NonEmptyString
    source_doc_uid: str | None = None
    original_filename: str | None = None


class IndexDocumentRequest(BaseModel):
    """Indexing envelope with stable external-identity normalization."""

    model_config = ConfigDict(extra="ignore")
    document: ProcessedDocument
    document_id: NonEmptyString | None = None
    source_doc_uid: NonEmptyString | None = None
    source_filename: NonEmptyString | None = None
    filename: NonEmptyString | None = None
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def normalize_identity(self) -> IndexDocumentRequest:
        metadata_uid = self.metadata.get("source_doc_uid")
        uid = self.source_doc_uid or self.document.source_doc_uid
        if uid is None and isinstance(metadata_uid, str) and metadata_uid.strip():
            uid = metadata_uid.strip()
        canonical_id = uid or self.document_id or self.document.document_id
        original_filename = (
            self.source_filename
            or self.filename
            or self.document.original_filename
            or canonical_id
        )
        self.source_doc_uid = uid
        self.document_id = canonical_id
        self.document.document_id = canonical_id
        self.source_filename = original_filename
        self.filename = original_filename
        self.document.source_doc_uid = uid
        self.document.original_filename = original_filename
        if uid:
            self.metadata["source_doc_uid"] = uid
        self.metadata["original_filename"] = original_filename
        return self


class BatchIndexRequest(BaseModel):
    documents: list[IndexDocumentRequest] = Field(min_length=1, max_length=1000)


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    source_doc_uid: str | None = None
    source_filename: str
    page: int
    section: str
    content_type: Literal["paragraph", "table"]
    text: str
    parent_text: str
    bbox: BBox
    source_block_ids: list[str]
    table_title: str | None = None
    table_context: str | None = None
    table_cells: list[Cell] = Field(default_factory=list)
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)


class SearchMode(str, Enum):
    hybrid = "hybrid"
    dense = "dense"
    lexical = "lexical"


class SearchFilters(BaseModel):
    document_id: str | None = None
    source_doc_uid: str | None = None
    source_filename: str | None = None
    filename: str | None = None
    page: int | None = Field(default=None, ge=0)
    section: str | None = None
    content_type: Literal["paragraph", "text", "table"] | None = None
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def normalize_aliases(self) -> SearchFilters:
        if self.filename and self.source_filename and self.filename != self.source_filename:
            raise ValueError("filename conflicts with source_filename")
        self.source_filename = self.source_filename or self.filename
        if self.content_type == "text":
            self.content_type = "paragraph"
        return self


class SearchRequest(BaseModel):
    """Canonical request plus the agent's legacy flat-filter aliases."""

    model_config = ConfigDict(extra="ignore")
    query: NonEmptyString
    top_k: int = Field(default=5, ge=1, le=50)
    candidate_k: int = Field(default=30, ge=1, le=250)
    mode: SearchMode = SearchMode.hybrid
    dense_weight: float = Field(default=0.55, ge=0, le=1, allow_inf_nan=False)
    rrf_k: int = Field(default=60, ge=1, le=200)
    rerank: bool = True
    include_parent_context: bool = True
    filters: SearchFilters = Field(default_factory=SearchFilters)
    document_id: str | None = None
    source_doc_uid: str | None = None
    filename: str | None = None
    trace_id: str | None = None

    @model_validator(mode="after")
    def normalize_legacy_filters(self) -> SearchRequest:
        if self.candidate_k < self.top_k:
            raise ValueError("candidate_k must be greater than or equal to top_k")
        aliases = {
            "document_id": self.document_id,
            "source_doc_uid": self.source_doc_uid,
            "source_filename": self.filename,
        }
        for field_name, value in aliases.items():
            if value is None:
                continue
            current = getattr(self.filters, field_name)
            if current is not None and current != value:
                raise ValueError(f"{field_name} conflicts with filters.{field_name}")
            setattr(self.filters, field_name, value)
        return self


class FilterRequest(BaseModel):
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)
    top_k: int = Field(default=5, ge=1, le=250)
    include_parent_context: bool = True
    document_id: str | None = None
    source_doc_uid: str | None = None
    filename: str | None = None
    page: int | None = Field(default=None, ge=0)
    section: str | None = None
    content_type: Literal["paragraph", "text", "table"] | None = None

    def as_filters(self) -> SearchFilters:
        metadata = dict(self.metadata)
        values: dict[str, Any] = {
            "document_id": self.document_id,
            "source_doc_uid": self.source_doc_uid,
            "filename": self.filename,
            "page": self.page,
            "section": self.section,
            "content_type": self.content_type,
        }
        aliases = {"source_filename": "filename"}
        for key in tuple(metadata):
            target = aliases.get(key, key)
            if target in values and values[target] is None:
                values[target] = metadata.pop(key)
        return SearchFilters(metadata=metadata, **values)


class RetrievalScores(BaseModel):
    dense: float | None = None
    lexical: float | None = None
    fusion: float
    reranker: float | None = None


class SearchHit(BaseModel):
    rank: int
    chunk_id: str
    document_id: str
    source_doc_uid: str | None = None
    filename: str
    source_filename: str
    page: int
    section: str
    content_type: Literal["text", "table"]
    content: str
    text: str
    score: float = Field(ge=0.0, le=1.0)
    parent_text: str | None = None
    bbox: BBox
    source_block_ids: list[str]
    table_title: str | None = None
    table_context: str | None = None
    table_cells: list[Cell] = Field(default_factory=list)
    metadata: dict[str, MetadataValue]
    scores: RetrievalScores


class SearchDiagnostics(BaseModel):
    search_settings: dict[str, float | int] = Field(default_factory=dict)
    eligible_count: int = 0
    dense_candidate_count: int = 0
    lexical_candidate_count: int = 0
    fused_candidate_count: int = 0
    reranker_candidate_count: int = 0
    stage_latency_ms: dict[str, float] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    query: str
    mode: SearchMode
    candidate_count: int
    reranked: bool
    latency_ms: float
    hits: list[SearchHit]
    diagnostics: SearchDiagnostics = Field(default_factory=SearchDiagnostics)
    trace_id: str | None = None


class IndexResponse(BaseModel):
    document_id: str
    source_doc_uid: str | None = None
    source_filename: str
    chunks_indexed: int
    chunks_replaced: int


class BatchIndexResponse(BaseModel):
    documents_indexed: int
    chunks_indexed: int
    chunks_replaced: int
    documents: list[IndexResponse]


class DocumentSummary(BaseModel):
    document_id: str
    source_doc_uid: str | None = None
    filename: str
    source_filename: str
    pages: int
    chunks: int
    tables: int
    metadata: dict[str, MetadataValue]


class ExtractedField(BaseModel):
    """Editable retrieval representation of one extracted text/table chunk."""

    chunk_id: str
    document_id: str
    page: int
    section: str
    content_type: Literal["text", "table"]
    content: str
    bbox: BBox
    source_block_ids: list[str]


class ExtractionCorrectionRequest(BaseModel):
    chunk_id: NonEmptyString
    corrected_text: NonEmptyString = Field(max_length=50000)
    corrected_by: NonEmptyString = Field(max_length=200)
    comment: str | None = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def normalize_comment(self) -> ExtractionCorrectionRequest:
        self.comment = self.comment.strip() if self.comment else None
        return self


class ExtractionCorrectionRecord(BaseModel):
    correction_id: str
    document_id: str
    chunk_id: str
    original_text: str
    corrected_text: str
    corrected_by: str
    comment: str | None = None
    created_at: datetime
    status: Literal["applied"] = "applied"


class CorpusStats(BaseModel):
    documents: int
    chunks: int
    tables: int
    embedding_model: str
    reranker_model: str | None
    # Switches the index was built with (app/index_format.py); empty is legacy.
    index_format: list[str] = Field(default_factory=list)
