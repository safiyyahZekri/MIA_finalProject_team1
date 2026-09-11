"""One constrained extraction repair after local shape/citation validation."""

from app.llm import ExtractionDirect, ExtractionInsufficient, ExtractionMultiSpan
from app.search_policy import material_sequence, material_tokens, normalize_value


def values(extraction) -> list[str] | None:
    if isinstance(extraction, ExtractionDirect):
        return [str(extraction.value)]
    if isinstance(extraction, ExtractionMultiSpan):
        return [str(value) for value in extraction.values]
    return None


def repair_extraction(
    llm, question: str, question_type: str, evidence: list[dict], extraction
):
    original = values(extraction)
    if not original or not evidence:
        return extraction, "not_needed"
    indexes = extraction.evidence_indexes
    valid_indexes = lambda items: (
        bool(items)
        and all(
            isinstance(i, int) and not isinstance(i, bool) and 1 <= i <= len(evidence)
            for i in items
        )
    )
    problems = []
    if isinstance(extraction, ExtractionMultiSpan) and len(original) < 2:
        problems.append(
            "multi_span needs at least two values; a singleton must be direct"
        )
    if not valid_indexes(indexes):
        problems.append("cite valid 1-based evidence_indexes for the existing answer")
    if any(value != normalize_value(value) for value in original):
        problems.append("normalize whitespace and percentage spacing")
    if not problems:
        return extraction, "not_needed"
    prompt = (
        f"{question}\n\nRepair the previous extraction: {extraction.model_dump_json()}\n"
        f"Validation issues: {'; '.join(problems)}. Preserve all facts, numeric values, "
        "units, signs, qualifiers and order. Only repair presentation, shape and citations. "
        "Return insufficient if the existing answer cannot be supported by this evidence."
    )
    candidate = llm.extract(prompt, question_type, evidence)
    if isinstance(candidate, ExtractionInsufficient):
        return candidate, "declined"
    repaired = values(candidate)
    if not repaired or not valid_indexes(candidate.evidence_indexes):
        return extraction, "rejected"
    if isinstance(candidate, ExtractionMultiSpan) and len(repaired) < 2:
        return extraction, "rejected"
    if material_sequence(original) != material_sequence(repaired):
        return extraction, "rejected_changed_facts"
    if isinstance(extraction, ExtractionDirect) and isinstance(
        candidate, ExtractionMultiSpan
    ):
        return extraction, "rejected_changed_shape"
    # Preserve the order of independent answer values.
    if len(original) > 1 and [material_tokens([v]) for v in original] != [
        material_tokens([v]) for v in repaired
    ]:
        return extraction, "rejected_changed_order"
    # Repair cannot introduce a citation whose passage lacks the answer tokens.
    support = [
        str(evidence[i - 1].get("text") or evidence[i - 1].get("content") or "")
        for i in candidate.evidence_indexes
    ]
    if not set(material_tokens(repaired)) <= set(material_tokens(support)):
        return extraction, "rejected_unsupported"
    return candidate, "accepted"
