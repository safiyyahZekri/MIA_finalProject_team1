from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
BBox = tuple[int, int, int, int]


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
    """The common subset emitted by doc-processor-api for text and tables."""

    model_config = ConfigDict(extra="ignore")

    bbox: list[int] = Field(min_length=4, max_length=4)
    uuid: NonEmptyString
    text: str = ""
    order: int = Field(ge=1)
    content_type: Literal["paragraph", "table"]
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


class IndexDocumentRequest(BaseModel):
    document: ProcessedDocument
    source_filename: NonEmptyString | None = None
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    source_filename: str
    page: int
    section: str
    content_type: Literal["paragraph", "table"]
    text: str
    parent_text: str
    bbox: BBox
    source_block_ids: list[str]
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class SearchMode(str, Enum):
    hybrid = "hybrid"
    dense = "dense"
    lexical = "lexical"


class SearchFilters(BaseModel):
    document_id: str | None = None
    source_filename: str | None = None
    page: int | None = Field(default=None, ge=0)
    section: str | None = None
    content_type: Literal["paragraph", "table"] | None = None
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class SearchRequest(BaseModel):
    query: NonEmptyString
    top_k: int = Field(default=5, ge=1, le=50)
    candidate_k: int = Field(default=30, ge=1, le=250)
    mode: SearchMode = SearchMode.hybrid
    rerank: bool = True
    include_parent_context: bool = True
    filters: SearchFilters = Field(default_factory=SearchFilters)

    @model_validator(mode="after")
    def candidate_pool_must_cover_results(self) -> "SearchRequest":
        if self.candidate_k < self.top_k:
            raise ValueError("candidate_k must be greater than or equal to top_k")
        return self


class RetrievalScores(BaseModel):
    dense: float | None = None
    lexical: float | None = None
    fusion: float
    reranker: float | None = None


class SearchHit(BaseModel):
    rank: int
    chunk_id: str
    document_id: str
    source_filename: str
    page: int
    section: str
    content_type: Literal["paragraph", "table"]
    text: str
    parent_text: str | None = None
    bbox: BBox
    source_block_ids: list[str]
    metadata: dict[str, str | int | float | bool | None]
    scores: RetrievalScores


class SearchResponse(BaseModel):
    query: str
    mode: SearchMode
    candidate_count: int
    reranked: bool
    latency_ms: float
    hits: list[SearchHit]


class IndexResponse(BaseModel):
    document_id: str
    source_filename: str
    chunks_indexed: int
    chunks_replaced: int


class DocumentSummary(BaseModel):
    document_id: str
    source_filename: str
    pages: int
    chunks: int
    tables: int
    metadata: dict[str, str | int | float | bool | None]


class CorpusStats(BaseModel):
    documents: int
    chunks: int
    tables: int
    embedding_model: str
    reranker_model: str | None
