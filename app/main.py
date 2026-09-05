import logging
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app import clients
from app.schemas import AskRequest, AskResponse

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
