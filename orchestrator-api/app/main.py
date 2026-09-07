import logging
import hashlib
import json
import time
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app import clients
from app.schemas import AskRequest, AskResponse, IngestResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("orchestrator")

app = FastAPI(title="LEDGER Orchestrator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

recent_queries: list[dict] = []


@app.get("/health")
async def health():
    return {"status": "ok", "mock_mode": clients.MOCK_MODE}


@app.post("/documents/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    source_doc_uid: Annotated[str | None, Form()] = None,
    metadata_json: Annotated[str | None, Form()] = None,
):
    """Process a raw PDF and immediately index its structured representation."""
    filename = file.filename or "document.pdf"
    source_doc_uid = (source_doc_uid.strip() or None) if source_doc_uid else None
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=415,
            detail={"code": "unsupported_file", "message": "only PDF files are supported"},
        )
    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=422,
            detail={"code": "empty_file", "message": "uploaded PDF is empty"},
        )
    try:
        metadata = json.loads(metadata_json) if metadata_json else {}
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_metadata", "message": "metadata_json must be valid JSON"},
        ) from exc
    if not isinstance(metadata, dict):
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_metadata", "message": "metadata_json must be an object"},
        )

    document_id = source_doc_uid or f"sha256-{hashlib.sha256(content).hexdigest()}"
    metadata = {
        **metadata,
        "source_doc_uid": source_doc_uid,
        "original_filename": filename,
    }
    try:
        processed = await clients.process_pdf(
            content, filename, document_id, source_doc_uid
        )
        indexed = await clients.index_processed_document(
            processed,
            filename,
            document_id,
            source_doc_uid,
            metadata,
        )
    except clients.ServiceIntegrationError as exc:
        logger.error("retrieval ingestion failed at %s: %s", exc.stage, exc)
        raise HTTPException(
            status_code=502,
            detail={
                "code": f"{exc.stage}_failed",
                "message": str(exc),
                "upstream_status": exc.status_code,
            },
        ) from exc

    replaced = int(indexed.get("chunks_replaced", 0))
    return IngestResponse(
        document_id=document_id,
        source_doc_uid=source_doc_uid,
        filename=filename,
        pages_processed=len(processed["pages"]),
        chunks_indexed=int(indexed["chunks_indexed"]),
        chunks_replaced=replaced,
        status="reindexed" if replaced else "indexed",
    )


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    start = time.perf_counter()

    answer = await clients.ask_agent(request.question, request.document_id)
    validation = await clients.validate_answer(answer)

    latency_ms = round((time.perf_counter() - start) * 1000, 1)
    recent_queries.append(
        {"question": request.question, "latency_ms": latency_ms, "valid": validation["valid"]}
    )

    if validation["valid"]:
        logger.info(
            "[ANSWER-VALIDATOR-SUCCESS] Received and validated answer of type '%s' with evidence %s",
            answer["answer_type"],
            answer["evidence"][0] if answer["evidence"] else {},
        )
    else:
        logger.error("[ANSWER-VALIDATOR-ERROR] %s", validation["reason"])
        raise HTTPException(status_code=422, detail=validation["reason"])

    return AskResponse(
        answer_type=answer["answer_type"],
        evidence=answer["evidence"],
        params=answer["params"],
        valid=validation["valid"],
        validator_message=validation["reason"],
    )


@app.get("/documents")
async def documents():
    return await clients.list_documents()


@app.get("/recent_queries")
async def get_recent_queries():
    return recent_queries[-20:]
