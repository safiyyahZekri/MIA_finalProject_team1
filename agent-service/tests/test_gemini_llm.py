"""GeminiLLM must report failures the way AnthropicLLM does, and build
requests the google-genai SDK accepts.

A rate limit that outlasts the retries, a blocked prompt, or output cut off
at max_output_tokens has to reach the caller as an error. Scored as
"insufficient evidence" it would count a question the model never answered as
a wrong answer.
"""

from types import SimpleNamespace

import pytest

from app import llm as llm_module
from app.llm import (
    Classification,
    ExtractionDirect,
    ExtractionInsufficient,
    GeminiLLM,
    ModelOutputError,
    StructuredOutputError,
)


class FakeQuotaExhausted(Exception):
    """Stands in for google.genai.errors.ClientError (429) once retries run out."""


class FakeModels:
    def __init__(self, outcome):
        self.outcome = outcome
        self.requests = []

    def generate_content(self, **request):
        self.requests.append(request)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def _response(text="", finish="STOP", prompt_tokens=100, output_tokens=20, thoughts=5, block_reason=None):
    return SimpleNamespace(
        text=text,
        candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name=finish))] if finish else [],
        prompt_feedback=SimpleNamespace(block_reason=SimpleNamespace(name=block_reason)) if block_reason else None,
        usage_metadata=SimpleNamespace(
            prompt_token_count=prompt_tokens,
            candidates_token_count=output_tokens,
            thoughts_token_count=thoughts,
        ),
    )


def _llm(outcome):
    llm = GeminiLLM.__new__(GeminiLLM)  # skip the real client: no key, no network
    models = FakeModels(outcome)
    llm._client = SimpleNamespace(models=models)
    llm.last_usage = {}
    return llm, models


@pytest.fixture
def gemini(monkeypatch):
    for name, value in {
        "GEMINI_MODEL": "gemini-test",
        "GEMINI_MAX_OUTPUT_TOKENS": 16000,
        "GEMINI_TEMPERATURE": None,
    }.items():
        monkeypatch.setattr(llm_module.settings, name, value)


EVIDENCE = [{"document_id": "d", "page": 1, "text": "Revenue was 3,875."}]
DIRECT = '{"shape": "direct", "value": "3,875"}'


def test_structured_request_sends_the_json_schema(gemini):
    llm, models = _llm(_response(text='{"question_type": "table", "search_query": "revenue"}'))

    assert isinstance(llm.classify("q"), Classification)

    request = models.requests[0]
    assert request["model"] == "gemini-test"
    assert "Classify this financial-document question" in request["contents"]
    config = request["config"]
    assert config["response_mime_type"] == "application/json"
    assert config["response_json_schema"] == Classification.model_json_schema()
    assert config["max_output_tokens"] == 16000
    assert "temperature" not in config


def test_temperature_is_sent_only_when_configured(gemini, monkeypatch):
    monkeypatch.setattr(llm_module.settings, "GEMINI_TEMPERATURE", 0.0)
    llm, models = _llm(_response(text='{"question_type": "text", "search_query": "x"}'))

    llm.classify("q")

    assert models.requests[0]["config"]["temperature"] == 0.0


def test_reformulate_sends_no_schema(gemini):
    llm, models = _llm(_response(text="  broader query \n"))

    assert llm.reformulate("q", "previous", 1) == "broader query"
    assert "response_json_schema" not in models.requests[0]["config"]


def test_usage_counts_thinking_as_output(gemini):
    llm, _ = _llm(_response(text=DIRECT, prompt_tokens=400, output_tokens=25, thoughts=75))

    result = llm.extract("q", "text", EVIDENCE)

    assert isinstance(result, ExtractionDirect) and result.value == "3,875"
    assert llm.last_usage == {"prompt_tokens": 400, "completion_tokens": 100}


@pytest.mark.parametrize("question_type", ["numerical", "text"])
def test_truncation_is_an_error_not_an_answer(gemini, question_type):
    llm, _ = _llm(_response(text='{"shape": "dir', finish="MAX_TOKENS"))
    with pytest.raises(ModelOutputError, match="max_output_tokens"):
        llm.extract("q", question_type, EVIDENCE)


def test_safety_stop_is_an_error_not_an_answer(gemini):
    llm, _ = _llm(_response(text="", finish="SAFETY"))
    with pytest.raises(ModelOutputError, match="SAFETY"):
        llm.extract("q", "text", EVIDENCE)


def test_blocked_prompt_is_an_error(gemini):
    llm, _ = _llm(_response(finish=None, block_reason="PROHIBITED_CONTENT"))
    with pytest.raises(ModelOutputError, match="PROHIBITED_CONTENT"):
        llm.classify("q")


@pytest.mark.parametrize("question_type", ["numerical", "text"])
def test_api_failure_propagates_instead_of_scoring_as_insufficient(gemini, question_type):
    llm, _ = _llm(FakeQuotaExhausted("429 RESOURCE_EXHAUSTED"))
    with pytest.raises(FakeQuotaExhausted):
        llm.extract("q", question_type, EVIDENCE)


@pytest.mark.parametrize("question_type", ["numerical", "text"])
def test_complete_response_off_schema_degrades_to_insufficient(gemini, question_type):
    llm, _ = _llm(_response(text='{"unexpected": true}'))
    assert isinstance(llm.extract("q", question_type, EVIDENCE), ExtractionInsufficient)


def test_complete_response_off_schema_in_classify_raises(gemini):
    llm, _ = _llm(_response(text='{"unexpected": true}'))
    with pytest.raises(StructuredOutputError):
        llm.classify("q")


def test_uses_the_same_prompts_as_anthropic(gemini, monkeypatch):
    monkeypatch.setattr(llm_module.settings, "GRADE_REQUIRE_ENTITY_MATCH", True)
    llm, models = _llm(_response(text='{"sufficient": false, "confidence": 0.9, "reason": "r"}'))

    llm.grade("q", EVIDENCE)

    assert "If the question names a company" in models.requests[0]["contents"]


def test_client_retries_rate_limits_inside_the_sdk(monkeypatch):
    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("google.genai.Client", FakeClient)
    monkeypatch.setattr(llm_module.settings, "GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(llm_module.settings, "GEMINI_MAX_ATTEMPTS", 8)

    GeminiLLM()

    retry = captured["http_options"]["retry_options"]
    assert captured["api_key"] == "test-key"
    assert retry["attempts"] == 8
    assert 429 in retry["http_status_codes"]


def test_get_llm_selects_gemini(monkeypatch):
    monkeypatch.setattr("google.genai.Client", lambda **kwargs: SimpleNamespace())
    monkeypatch.setattr(llm_module.settings, "LLM_PROVIDER", "gemini")

    assert isinstance(llm_module.get_llm(), GeminiLLM)


def test_health_reports_gemini_settings_and_a_missing_key(monkeypatch):
    from fastapi.testclient import TestClient

    from app import main

    monkeypatch.setattr(main.settings, "LLM_PROVIDER", "gemini")
    monkeypatch.setattr(main.settings, "GEMINI_API_KEY", "")
    monkeypatch.setattr(main.settings, "GEMINI_MODEL", "gemini-test")

    body = TestClient(main.app).get("/health").json()

    assert body["config"]["model"] == "gemini-test"
    assert body["status"] == "degraded" and "GEMINI_API_KEY" in body["hint"]


def test_cost_uses_gemini_rates_only_when_set(monkeypatch):
    from app import main

    trace = [{"step": "grade", "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}}]
    monkeypatch.setattr(main.settings, "LLM_PROVIDER", "gemini")
    monkeypatch.setattr(main.settings, "GEMINI_INPUT_USD_PER_MTOK", 0.0)
    monkeypatch.setattr(main.settings, "GEMINI_OUTPUT_USD_PER_MTOK", 0.0)
    assert main._usage_summary(trace)["cost_usd"] is None

    monkeypatch.setattr(main.settings, "GEMINI_INPUT_USD_PER_MTOK", 0.3)
    monkeypatch.setattr(main.settings, "GEMINI_OUTPUT_USD_PER_MTOK", 2.5)
    assert main._usage_summary(trace)["cost_usd"] == 2.8
