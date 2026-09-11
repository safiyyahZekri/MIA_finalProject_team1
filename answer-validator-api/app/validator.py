"""
Validation engine for the answer-validator-api.

Responsibilities (from the Final Project spec, service #7):
  1. Strictly check the incoming JSON conforms to the Strict Answer Schema —
     answer_type, presence of required evidence citations, and data types of
     every value.
  2. Log a success or a detailed error line to the console in an exact
     format:
       [ANSWER-VALIDATOR-SUCCESS] Received and validated answer of type
       'calculated' with evidence {'document_id': 'doc_041', 'page': 2}.

       [ANSWER-VALIDATOR-ERROR] Invalid answer. Reason: Missing required
       evidence citation.

       [ANSWER-VALIDATOR-ERROR] Invalid answer for 'calculated': Missing
       required key 'formula'.

This module is pure (no FastAPI/logging side effects other than returning
the exact strings) so it is trivial to unit test.

Decision record — `calculated` evidence count: this validator enforces only
that `evidence` is non-empty for `calculated` answers, with no minimum tied
to the formula's contents. See README.md for the dataset analysis that
justifies this (evidence count tracks distinct document/page sources, not
formula literal count, and a validator has no gold data to check the former
against).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from pydantic import ValidationError

from .schemas import (
    ANSWER_TYPES,
    EVIDENCE_MAY_BE_EMPTY,
    PARAMS_MODEL_BY_TYPE,
    EvidenceItem,
    RawAnswer,
)


@dataclass
class ValidationOutcome:
    valid: bool
    reason: Optional[str]
    log_line: str


def _generic_error(reason: str) -> ValidationOutcome:
    return ValidationOutcome(
        valid=False,
        reason=reason,
        log_line=f"[ANSWER-VALIDATOR-ERROR] Invalid answer. Reason: {reason}",
    )


def _typed_error(answer_type: str, message: str) -> ValidationOutcome:
    return ValidationOutcome(
        valid=False,
        reason=message,
        log_line=f"[ANSWER-VALIDATOR-ERROR] Invalid answer for '{answer_type}': {message}",
    )


def _missing_key_message(exc: ValidationError) -> Optional[str]:
    """Turn a pydantic 'field required' error into
    "Missing required key '<field>'." — matching the spec's wording. Returns
    None if the first error isn't a plain missing-field error, so the caller
    can fall back to a more generic message."""
    for err in exc.errors():
        if err["type"] == "missing":
            field = err["loc"][-1]
            return f"Missing required key '{field}'."
    return None


def validate_answer(payload: dict) -> ValidationOutcome:
    # --- Step 0: payload must be a JSON object at all ---
    if not isinstance(payload, dict):
        return _generic_error("Payload must be a JSON object.")

    # --- Step 1: top-level shape (answer_type / evidence / params, no extras) ---
    try:
        raw = RawAnswer.model_validate(payload)
    except ValidationError as exc:
        # Distinguish "unknown top-level key" from "wrong type" from "missing".
        first = exc.errors()[0]
        if first["type"] == "extra_forbidden":
            bad_key = first["loc"][-1]
            return _generic_error(f"Unexpected field '{bad_key}' at top level.")
        if first["type"] == "missing":
            missing_field = first["loc"][-1]
            return _generic_error(f"Missing required key '{missing_field}'.")
        return _generic_error(f"Malformed answer object ({first['msg']}).")

    answer_type = raw.answer_type
    if answer_type not in ANSWER_TYPES:
        return _generic_error(
            f"Unknown answer_type '{answer_type}'. Must be one of {list(ANSWER_TYPES)}."
        )

    # --- Step 2: evidence array — validate each item strictly ---
    evidence_items = []
    for idx, item in enumerate(raw.evidence):
        try:
            evidence_items.append(EvidenceItem.model_validate(item))
        except ValidationError as exc:
            first = exc.errors()[0]
            loc = ".".join(str(p) for p in first["loc"])
            return _generic_error(
                f"Malformed evidence entry at index {idx} ({loc}: {first['msg']})."
            )

    if not evidence_items and not EVIDENCE_MAY_BE_EMPTY[answer_type]:
        return _generic_error("Missing required evidence citation.")

    # --- Step 3: params — dispatch to the type-specific strict model ---
    params_model = PARAMS_MODEL_BY_TYPE[answer_type]
    try:
        params_model.model_validate(raw.params)
    except ValidationError as exc:
        missing = _missing_key_message(exc)
        if missing:
            return _typed_error(answer_type, missing)
        first = exc.errors()[0]
        if first["type"] == "extra_forbidden":
            bad_key = first["loc"][-1]
            return _typed_error(answer_type, f"Unexpected key '{bad_key}'.")
        loc = ".".join(str(p) for p in first["loc"]) or "params"
        return _typed_error(answer_type, f"Invalid value for '{loc}' ({first['msg']}).")

    # --- Step 4: type-specific evidence-quantity rules beyond "non-empty" ---
    if answer_type == "multi_span":
        values = raw.params.get("values", [])
        if len(evidence_items) < 1:
            return _typed_error(answer_type, "At least one citation is required for 'values'.")
        # A single citation may cover multiple values, so we only require
        # evidence <= values is NOT enforced; we just require non-empty
        # (already checked above) — documented limitation, see README.
        _ = values  # kept for clarity / future stricter checks

    # `calculated` intentionally has no evidence-count rule beyond
    # "non-empty" (enforced above). See the "Resolved: what 'one citation
    # per operand' actually means" section in README.md for the dataset
    # analysis this is based on: evidence-entry count tracks the number of
    # distinct (document, page) sources behind the formula's operands, not
    # the number of numeric literals in the formula string. A validator
    # only sees the evidence array the agent declares — it has no gold data
    # to check that count against — so the only thing it can safely enforce
    # is the non-empty floor; the real per-operand attribution is the
    # calculator tool's responsibility at generation time.

    # --- Success ---
    first_ev = evidence_items[0] if evidence_items else None
    ev_repr = (
        {"document_id": first_ev.document_id, "page": first_ev.page} if first_ev else {}
    )
    return ValidationOutcome(
        valid=True,
        reason=None,
        log_line=(
            f"[ANSWER-VALIDATOR-SUCCESS] Received and validated answer of type "
            f"'{answer_type}' with evidence {ev_repr}."
        ),
    )
