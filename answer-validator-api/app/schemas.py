"""
Strict Answer Schema definitions for LEDGER.

Mirrors the schema in the Final Project spec exactly:

    {
      "answer_type": "<type_name>",
      "evidence": [ { "document_id": "...", "page": 0, "section": "..." } ],
      "params": { ... }
    }

Four allowed answer_type values: direct, calculated, multi_span, insufficient_evidence.
"""
from __future__ import annotations

from typing import Any, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

ANSWER_TYPES = ("direct", "calculated", "multi_span", "insufficient_evidence")


class EvidenceItem(BaseModel):
    """A single citation. document_id + page are the hard requirement from the
    spec ("At least 1 citation with document_id and page"). `section` is
    included in every example but is not called out as mandatory, so it is
    optional here — teams that want to force it can flip it to required.
    """

    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(min_length=1)
    page: int = Field(ge=0)
    section: Optional[str] = None

    @field_validator("document_id")
    @classmethod
    def _non_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("document_id must not be blank")
        return v


Number = Union[int, float]
Value = Union[str, Number]


class DirectParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: Value


class CalculatedParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: Number
    formula: str = Field(min_length=1)


class MultiSpanParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    values: List[Value] = Field(min_length=1)


class InsufficientEvidenceParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reason: str = Field(min_length=1)


PARAMS_MODEL_BY_TYPE = {
    "direct": DirectParams,
    "calculated": CalculatedParams,
    "multi_span": MultiSpanParams,
    "insufficient_evidence": InsufficientEvidenceParams,
}

# Whether an empty evidence list is acceptable for the given answer_type.
EVIDENCE_MAY_BE_EMPTY = {
    "direct": False,
    "calculated": False,
    "multi_span": False,
    "insufficient_evidence": True,
}


class RawAnswer(BaseModel):
    """Loose top-level shape used just to walk the payload safely before we
    dispatch to the type-specific, strict, extra='forbid' models above."""

    model_config = ConfigDict(extra="forbid")

    answer_type: str
    evidence: List[dict] = Field(default_factory=list)
    params: dict = Field(default_factory=dict)


class ValidateAnswerResponse(BaseModel):
    """Response shape the orchestrator expects back, verbatim from Safiyyah's
    integration message: {"valid": true, "reason": null}"""

    valid: bool
    reason: Optional[str] = None
