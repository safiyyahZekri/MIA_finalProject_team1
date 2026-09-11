import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .schemas import ValidateAnswerResponse
from .validator import validate_answer

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("answer-validator-api")

app = FastAPI(
    title="LEDGER Answer Validator Service",
    description="Single source of truth for what constitutes a valid, grounded LEDGER answer.",
    version="1.0.0",
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/validate_answer", response_model=ValidateAnswerResponse)
async def validate_answer_endpoint(request: Request):
    try:
        payload = await request.json()
    except Exception:
        logger.info("[ANSWER-VALIDATOR-ERROR] Invalid answer. Reason: Body is not valid JSON.")
        return JSONResponse(
            status_code=400,
            content={"valid": False, "reason": "Body is not valid JSON."},
        )

    outcome = validate_answer(payload)
    logger.info(outcome.log_line)

    return ValidateAnswerResponse(valid=outcome.valid, reason=outcome.reason)
