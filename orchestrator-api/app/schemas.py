from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class Evidence(BaseModel):
    document_id: str
    page: int
    section: Optional[str] = None


class EvidenceBox(BaseModel):
    """Where a cited page's evidence sits, for highlighting it on the PDF."""

    document_id: str
    page: int
    section: Optional[str] = None
    bbox: tuple[int, int, int, int]


class AskRequest(BaseModel):
    question: str
    document_id: Optional[str] = None


class AskResponse(BaseModel):
    answer_type: Literal["direct", "calculated", "multi_span", "insufficient_evidence"]
    evidence: list[Evidence]
    params: dict
    valid: bool
    validator_message: Optional[str] = None
    # Beside the answer, not inside it: the Strict Answer Schema allows only
    # document_id, page and section per citation.
    evidence_boxes: list[EvidenceBox] = Field(default_factory=list)


class IngestResponse(BaseModel):
    document_id: str
    source_doc_uid: Optional[str] = None
    filename: str
    pages_processed: int
    chunks_indexed: int
    chunks_replaced: int
    status: Literal["indexed", "reindexed"]


class ReviewRequest(BaseModel):
    """Human review of one answer; annotations never bypass validation."""

    question: str = Field(min_length=1, max_length=4000)
    verdict: Literal["correct", "needs_correction"]
    original_answer: dict[str, Any] = Field(min_length=1)
    corrected_answer: Optional[str] = Field(default=None, max_length=10000)
    comment: Optional[str] = Field(default=None, max_length=4000)
    document_id: Optional[str] = Field(default=None, max_length=512)

    @model_validator(mode="after")
    def require_correction_when_marked_wrong(self):
        self.question = self.question.strip()
        self.corrected_answer = (
            self.corrected_answer.strip() if self.corrected_answer else None
        )
        self.comment = self.comment.strip() if self.comment else None
        self.document_id = self.document_id.strip() if self.document_id else None
        if not self.question:
            raise ValueError("question must not be blank")
        if self.verdict == "needs_correction" and not self.corrected_answer:
            raise ValueError("corrected_answer is required when the answer needs correction")
        return self


class ReviewRecord(ReviewRequest):
    review_id: str
    created_at: datetime
    status: Literal["recorded"] = "recorded"


class ExtractedField(BaseModel):
    chunk_id: str
    document_id: str
    page: int
    section: str
    content_type: Literal["text", "table"]
    content: str
    bbox: tuple[int, int, int, int]
    source_block_ids: list[str]


class ExtractionCorrectionRequest(BaseModel):
    chunk_id: str = Field(min_length=1, max_length=512)
    corrected_text: str = Field(min_length=1, max_length=50000)
    corrected_by: str = Field(min_length=1, max_length=200)
    comment: Optional[str] = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def normalize_fields(self):
        self.chunk_id = self.chunk_id.strip()
        self.corrected_text = self.corrected_text.strip()
        self.corrected_by = self.corrected_by.strip()
        self.comment = self.comment.strip() if self.comment else None
        if not self.chunk_id or not self.corrected_text or not self.corrected_by:
            raise ValueError("correction fields must not be blank")
        return self


class ExtractionCorrectionRecord(ExtractionCorrectionRequest):
    correction_id: str
    document_id: str
    original_text: str
    created_at: datetime
    status: Literal["applied"] = "applied"
