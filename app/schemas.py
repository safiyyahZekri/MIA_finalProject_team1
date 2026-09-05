from typing import Literal, Optional, Union

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
