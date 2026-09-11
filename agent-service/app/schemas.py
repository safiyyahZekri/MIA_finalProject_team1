"""
Pydantic models mirroring the "Strict Answer Schema" from the project spec.

answer-validator-api is the single source of truth for validation in the
running system, but the agent performs the same check locally before ever
returning an answer -- catching schema drift immediately, in-process,
instead of only discovering it downstream.
"""
from __future__ import annotations

from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator


class Evidence(BaseModel):
    document_id: str
    page: int
    section: Optional[str] = None


class DirectParams(BaseModel):
    value: Union[str, float, int]


class CalculatedParams(BaseModel):
    value: float
    formula: str


class MultiSpanParams(BaseModel):
    values: List[Union[str, float, int]] = Field(min_length=2)


class InsufficientEvidenceParams(BaseModel):
    reason: str


class DirectAnswer(BaseModel):
    answer_type: Literal["direct"] = "direct"
    evidence: List[Evidence] = Field(min_length=1)
    params: DirectParams


class CalculatedAnswer(BaseModel):
    answer_type: Literal["calculated"] = "calculated"
    evidence: List[Evidence] = Field(min_length=1)
    params: CalculatedParams


class MultiSpanAnswer(BaseModel):
    answer_type: Literal["multi_span"] = "multi_span"
    evidence: List[Evidence] = Field(min_length=1)
    params: MultiSpanParams

    @model_validator(mode="after")
    def _check_counts(self):
        # Spec: at least one citation per value (a citation may cover >1 value
        # if they share a cell/passage), so evidence must never be empty and
        # should not wildly exceed the number of values.
        if len(self.evidence) < 1:
            raise ValueError("multi_span requires at least one evidence citation")
        return self


class InsufficientEvidenceAnswer(BaseModel):
    answer_type: Literal["insufficient_evidence"] = "insufficient_evidence"
    evidence: List[Evidence] = Field(default_factory=list)
    params: InsufficientEvidenceParams


AnswerUnion = Union[
    DirectAnswer, CalculatedAnswer, MultiSpanAnswer, InsufficientEvidenceAnswer
]

_TYPE_MAP = {
    "direct": DirectAnswer,
    "calculated": CalculatedAnswer,
    "multi_span": MultiSpanAnswer,
    "insufficient_evidence": InsufficientEvidenceAnswer,
}


def validate_answer_dict(answer: dict) -> AnswerUnion:
    """Raise pydantic.ValidationError if `answer` doesn't conform to its
    declared answer_type's schema. Returns the parsed model on success."""
    answer_type = answer.get("answer_type")
    model = _TYPE_MAP.get(answer_type)
    if model is None:
        raise ValueError(
            f"Unknown answer_type '{answer_type}'. Must be one of {list(_TYPE_MAP)}"
        )
    return model.model_validate(answer)


class AnswerRequest(BaseModel):
    question: str
    document_id: Optional[str] = None  # optional scoping; corpus-wide by default


class AnswerResponse(BaseModel):
    answer: dict
    question_type: str
    retries_used: int
    trace: list
    # System-performance figures eval-service reads from the top level.
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    tokens_used: int = 0
    cost_usd: Optional[float] = None
