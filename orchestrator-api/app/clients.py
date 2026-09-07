import os
from typing import Any

import httpx

AGENT_SERVICE_URL = os.getenv("AGENT_SERVICE_URL", "http://localhost:8003")
VALIDATOR_SERVICE_URL = os.getenv("VALIDATOR_SERVICE_URL", "http://localhost:8004")
DOC_PROCESSOR_URL = os.getenv("DOC_PROCESSOR_URL", "http://localhost:8001")
RETRIEVAL_SERVICE_URL = os.getenv("RETRIEVAL_SERVICE_URL", "http://localhost:8002")

# MOCK_MODE lets this service run and be demoed before agent-service /
# answer-validator-api / retrieval-api exist. Flip to "false" once real
# services are up (see README).
MOCK_MODE = os.getenv("MOCK_MODE", "true").lower() == "true"

TIMEOUT = float(os.getenv("SERVICE_TIMEOUT_SECONDS", "15"))
DOC_PROCESSOR_TIMEOUT = float(os.getenv("DOC_PROCESSOR_TIMEOUT_SECONDS", "300"))
RETRIEVAL_INDEX_TIMEOUT = float(os.getenv("RETRIEVAL_INDEX_TIMEOUT_SECONDS", "300"))


class ServiceIntegrationError(RuntimeError):
    def __init__(self, stage: str, message: str, status_code: int | None = None):
        super().__init__(message)
        self.stage = stage
        self.status_code = status_code


def _error_message(response: httpx.Response) -> str:
    try:
        body: Any = response.json()
    except ValueError:
        return response.text or f"HTTP {response.status_code}"
    if isinstance(body, dict):
        detail = body.get("detail", body)
        if isinstance(detail, dict):
            return str(detail.get("message", detail))
        return str(detail)
    return str(body)


async def process_pdf(
    content: bytes,
    filename: str,
    document_id: str,
    source_doc_uid: str | None = None,
) -> dict:
    """Call the existing OCR service while explicitly propagating identity."""
    data = {"document_id": document_id}
    if source_doc_uid:
        data["source_doc_uid"] = source_doc_uid
    try:
        async with httpx.AsyncClient(timeout=DOC_PROCESSOR_TIMEOUT) as client:
            response = await client.post(
                f"{DOC_PROCESSOR_URL}/document_processing",
                data=data,
                files={"file": (filename, content, "application/pdf")},
            )
    except httpx.HTTPError as exc:
        raise ServiceIntegrationError("document_processor", str(exc)) from exc
    if response.is_error:
        raise ServiceIntegrationError(
            "document_processor", _error_message(response), response.status_code
        )
    try:
        processed = response.json()
    except ValueError as exc:
        raise ServiceIntegrationError(
            "document_processor", "processor returned malformed JSON"
        ) from exc
    if not isinstance(processed, dict) or not isinstance(processed.get("pages"), list):
        raise ServiceIntegrationError(
            "document_processor", "processor response is missing a pages array"
        )
    processed["document_id"] = document_id
    processed["source_doc_uid"] = source_doc_uid
    processed["original_filename"] = filename
    return processed


async def index_processed_document(
    processed: dict,
    filename: str,
    document_id: str,
    source_doc_uid: str | None,
    metadata: dict[str, str | int | float | bool | None],
) -> dict:
    payload = {
        "document": processed,
        "document_id": document_id,
        "source_doc_uid": source_doc_uid,
        "source_filename": filename,
        "metadata": metadata,
    }
    try:
        async with httpx.AsyncClient(timeout=RETRIEVAL_INDEX_TIMEOUT) as client:
            response = await client.post(
                f"{RETRIEVAL_SERVICE_URL}/documents", json=payload
            )
    except httpx.HTTPError as exc:
        raise ServiceIntegrationError("retrieval_index", str(exc)) from exc
    if response.is_error:
        raise ServiceIntegrationError(
            "retrieval_index", _error_message(response), response.status_code
        )
    try:
        result = response.json()
    except ValueError as exc:
        raise ServiceIntegrationError(
            "retrieval_index", "retrieval returned malformed JSON"
        ) from exc
    if not isinstance(result, dict) or result.get("document_id") != document_id:
        raise ServiceIntegrationError(
            "retrieval_index", "retrieval response has an unexpected document identity"
        )
    return result


async def ask_agent(question: str, document_id: str | None = None) -> dict:
    if MOCK_MODE:
        return _mock_agent_answer(question)
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.post(
            f"{AGENT_SERVICE_URL}/agent/query",
            json={"question": question, "document_id": document_id},
        )
        resp.raise_for_status()
        return resp.json()


async def validate_answer(answer: dict) -> dict:
    if MOCK_MODE:
        return _mock_validate(answer)
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.post(f"{VALIDATOR_SERVICE_URL}/validate_answer", json=answer)
        resp.raise_for_status()
        return resp.json()


async def list_documents() -> list[dict]:
    if MOCK_MODE:
        return _mock_documents()
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.get(f"{RETRIEVAL_SERVICE_URL}/documents")
        resp.raise_for_status()
        return resp.json()


def _mock_agent_answer(question: str) -> dict:
    q = question.lower()
    if "how many" in q or "count" in q or "which" in q:
        return {
            "answer_type": "multi_span",
            "evidence": [{"document_id": "doc_022", "page": 3, "section": "Operating Expenses"}],
            "params": {"values": ["Marketing", "R&D", "Logistics"]},
        }
    if "%" in q or "change" in q or "increase" in q or "difference" in q:
        return {
            "answer_type": "calculated",
            "evidence": [
                {"document_id": "doc_041", "page": 2, "section": "Operating Expenses"},
                {"document_id": "doc_041", "page": 2, "section": "Operating Expenses"},
            ],
            "params": {"value": 13.4, "formula": "(3875-3410)/3410*100"},
        }
    if "restructuring" in q:
        return {
            "answer_type": "insufficient_evidence",
            "evidence": [],
            "params": {"reason": "No document in the indexed corpus reports restructuring expenses."},
        }
    return {
        "answer_type": "direct",
        "evidence": [{"document_id": "doc_017", "page": 1, "section": "Income Statement"}],
        "params": {"value": "$142.5M"},
    }


def _mock_validate(answer: dict) -> dict:
    answer_type = answer.get("answer_type")
    evidence = answer.get("evidence", [])
    params = answer.get("params", {})

    if answer_type not in {"direct", "calculated", "multi_span", "insufficient_evidence"}:
        return {"valid": False, "reason": f"Unknown answer_type '{answer_type}'"}

    if answer_type != "insufficient_evidence" and not evidence:
        return {"valid": False, "reason": "Missing required evidence citation."}

    required_keys = {
        "direct": ["value"],
        "calculated": ["value", "formula"],
        "multi_span": ["values"],
        "insufficient_evidence": ["reason"],
    }[answer_type]
    for key in required_keys:
        if key not in params:
            return {"valid": False, "reason": f"Missing required key '{key}'"}

    return {"valid": True, "reason": None}


def _mock_documents() -> list[dict]:
    return [
        {"document_id": "doc_017", "title": "CTS Corp 10-K excerpt", "pages": 3, "tables": 2},
        {"document_id": "doc_041", "title": "Jabil Inc Annual Report excerpt", "pages": 2, "tables": 1},
        {"document_id": "doc_022", "title": "AMCON Distributing 2019", "pages": 4, "tables": 3},
    ]
