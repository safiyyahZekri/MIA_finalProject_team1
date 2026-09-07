import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("LLM_PROVIDER", "mock")
os.environ.setdefault("RETRIEVAL_API_URL", "http://localhost:19999")  # unreachable on purpose
os.environ.setdefault("RETRIEVAL_FALLBACK_TO_MOCK", "true")

from app.calculator import CalculatorError, calculate  # noqa: E402
from app.graph import run_agent  # noqa: E402
from app.main import app  # noqa: E402
from app.schemas import validate_answer_dict  # noqa: E402

client = TestClient(app)


# --------------------------------------------------------------- calculator
def test_calculate_basic_arithmetic():
    assert calculate("(3875-3410)/3410*100") == pytest.approx(13.6363636, rel=1e-4)
    assert calculate("2+2") == 4.0
    assert calculate("10/2") == 5.0


def test_calculate_rejects_unsafe_input():
    with pytest.raises(CalculatorError):
        calculate("__import__('os').system('echo hi')")
    with pytest.raises(CalculatorError):
        calculate("1/0")
    with pytest.raises(CalculatorError):
        calculate("")


# ------------------------------------------------------------------ health
def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


# -------------------------------------------------------------- graph runs
@pytest.mark.asyncio
async def test_numerical_question_produces_calculated_answer():
    state = await run_agent(
        "What was the percentage change in operating expenses from 2020 to 2021?"
    )
    answer = state["answer"]
    validate_answer_dict(answer)  # must not raise
    assert answer["answer_type"] in ("calculated", "insufficient_evidence")
    if answer["answer_type"] == "calculated":
        assert "formula" in answer["params"]
        assert isinstance(answer["params"]["value"], (int, float))
        assert len(answer["evidence"]) >= 1


@pytest.mark.asyncio
async def test_direct_question_produces_direct_or_insufficient_answer():
    state = await run_agent("What was the operating income reported in 2020?")
    answer = state["answer"]
    validate_answer_dict(answer)
    assert answer["answer_type"] in ("direct", "insufficient_evidence")


@pytest.mark.asyncio
async def test_multi_span_question():
    state = await run_agent("Which expense categories increased in 2020?")
    answer = state["answer"]
    validate_answer_dict(answer)
    assert answer["answer_type"] in ("multi_span", "direct", "insufficient_evidence")


@pytest.mark.asyncio
async def test_unanswerable_question_returns_insufficient_evidence():
    state = await run_agent("What is the CEO's favorite color?")
    answer = state["answer"]
    validate_answer_dict(answer)
    assert answer["answer_type"] == "insufficient_evidence"
    assert "reason" in answer["params"]
    assert state["retry_count"] == 2  # MAX_RETRIES exhausted before giving up


@pytest.mark.asyncio
async def test_retry_loop_bounded_by_max_retries():
    state = await run_agent("What is the flux capacitor efficiency ratio?")
    assert state["retry_count"] <= 2
    assert state["answer"]["answer_type"] == "insufficient_evidence"


# ------------------------------------------------------------------- HTTP
def test_agent_query_endpoint_returns_raw_answer_only():
    """Orchestrator's contract: POST /agent/query -> ONLY the answer object
    at the top level, no wrapper (no question_type/retries_used/trace)."""
    resp = client.post(
        "/agent/query",
        json={"question": "What was the operating income reported in 2020?", "document_id": None},
    )
    assert resp.status_code == 200
    body = resp.json()
    validate_answer_dict(body)  # raises if it's not a bare, schema-valid answer
    assert "answer" not in body  # must NOT be wrapped like /answer's response
    assert "trace" not in body
    assert "answer_type" in body and "evidence" in body and "params" in body


def test_answer_endpoint_end_to_end():
    resp = client.post("/answer", json={"question": "What was the operating income reported in 2020?"})
    assert resp.status_code == 200
    body = resp.json()
    assert "answer" in body
    validate_answer_dict(body["answer"])
    assert body["question_type"] in ("numerical", "table", "text")
    assert isinstance(body["trace"], list) and len(body["trace"]) > 0


# -------------------------------------------------------------- ollama --
def test_ollama_falls_back_to_mock_when_unreachable():
    """OllamaLLM must never raise out of classify/grade/reformulate/extract
    just because `ollama serve` isn't running -- it should degrade to the
    same mock heuristic used by the offline provider."""
    from app.llm import OllamaLLM

    llm = OllamaLLM()
    llm.base_url = "http://localhost:1"  # nothing listens here
    llm._client = __import__("httpx").Client(timeout=1)

    classification = llm.classify("What was the operating income reported in 2020?")
    assert classification.question_type in ("numerical", "table", "text")

    grade = llm.grade("some question", [])
    assert grade.sufficient is False

    query = llm.reformulate("q", "prev query", 1)
    assert isinstance(query, str) and query

    extraction = llm.extract("What was the operating income?", "text", [])
    assert extraction is not None


def test_ollama_ping_false_when_unreachable():
    from app.llm import OllamaLLM

    llm = OllamaLLM()
    llm.base_url = "http://localhost:1"
    llm._client = __import__("httpx").Client(timeout=1)
    assert llm.ping() is False


# ---------------------------------------------------------------- groq --
def test_groq_falls_back_to_mock_when_unreachable():
    """GroqLLM must never raise out of classify/grade/reformulate/extract
    just because the API is unreachable or the key is bad -- it should
    degrade to the same mock heuristic used by the offline provider."""
    import os

    from app.llm import GroqLLM

    os.environ.setdefault("GROQ_API_KEY", "test-key-not-real")
    from app.config import settings as _settings

    _settings.GROQ_API_KEY = _settings.GROQ_API_KEY or "test-key-not-real"

    llm = GroqLLM()
    llm.base_url = "http://localhost:1"  # nothing listens here
    llm._client = __import__("httpx").Client(timeout=1)

    classification = llm.classify("What was the operating income reported in 2020?")
    assert classification.question_type in ("numerical", "table", "text")

    grade = llm.grade("some question", [])
    assert grade.sufficient is False

    query = llm.reformulate("q", "prev query", 1)
    assert isinstance(query, str) and query

    extraction = llm.extract("What was the operating income?", "text", [])
    assert extraction is not None


def test_groq_ping_false_when_unreachable():
    import os

    from app.llm import GroqLLM

    os.environ.setdefault("GROQ_API_KEY", "test-key-not-real")
    from app.config import settings as _settings

    _settings.GROQ_API_KEY = _settings.GROQ_API_KEY or "test-key-not-real"

    llm = GroqLLM()
    llm.base_url = "http://localhost:1"
    llm._client = __import__("httpx").Client(timeout=1)
    assert llm.ping() is False


def test_groq_requires_api_key():
    from app.config import settings as _settings
    from app.llm import GroqError, GroqLLM

    original = _settings.GROQ_API_KEY
    _settings.GROQ_API_KEY = ""
    try:
        with pytest.raises(GroqError):
            GroqLLM()
    finally:
        _settings.GROQ_API_KEY = original


def test_answer_endpoint_rejects_empty_question():
    resp = client.post("/answer", json={"question": "   "})
    assert resp.status_code == 422


def test_never_returns_malformed_schema_across_many_questions():
    questions = [
        "What was the total revenue in 2021?",
        "By how much did revenue grow between 2019 and 2020?",
        "List the expense categories that increased.",
        "What is the meaning of life?",
        "How many times larger is the 2021 expense than 2020?",
    ]
    for q in questions:
        resp = client.post("/answer", json={"question": q})
        assert resp.status_code == 200
        validate_answer_dict(resp.json()["answer"])  # raises on any schema drift
