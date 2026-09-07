from typing import Literal, Optional

from pydantic import BaseModel


class Evidence(BaseModel):
    document_id: str
    page: int
    section: Optional[str] = None


class AskRequest(BaseModel):
    question: str
    document_id: Optional[str] = None


class AskResponse(BaseModel):
    answer_type: Literal["direct", "calculated", "multi_span", "insufficient_evidence"]
    evidence: list[Evidence]
    params: dict
    valid: bool
    validator_message: Optional[str] = None


class IngestResponse(BaseModel):
    document_id: str
    source_doc_uid: Optional[str] = None
    filename: str
    pages_processed: int
    chunks_indexed: int
    chunks_replaced: int
    status: Literal["indexed", "reindexed"]
