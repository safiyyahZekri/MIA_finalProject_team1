"""GeminiLLM must report failures the way AnthropicLLM does, build requests
the google-genai SDK accepts, and move between API keys on quota errors.

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
    GeminiQuotaExhausted,
    ModelOutputError,
    StructuredOutputError,
)


class FakeQuotaExhausted(Exception):
    """Stands in for google.genai.errors.ClientError on a non-429 failure."""


class Fake429(Exception):
    """Stands in for google.genai.errors.ClientError with a 429 status."""

    code = 429


DAILY = Fake429(
    "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'details': [{'violations': [{'quotaId': "
    "'GenerateRequestsPerDayPerProjectPerModel-FreeTier', 'quotaValue': '20'}]}, {'retryDelay': '13s'}]}}"
)
PER_MINUTE = Fake429(
    "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'details': [{'violations': [{'quotaId': "
    "'GenerateRequestsPerMinutePerProjectPerModel-FreeTier', 'quotaValue': '5'}]}, {'retryDelay': '7s'}]}}"
)


class FakeModels:
    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.requests = []

    def generate_content(self, **request):
        self.requests.append(request)
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


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


def _llm(*outcomes, key="key-1"):
    """A GeminiLLM whose single configured key answers with `outcomes`."""
    llm = GeminiLLM.__new__(GeminiLLM)  # skip the real SDK: no key, no network
    models = FakeModels(*outcomes)
    llm._clients = {key: SimpleNamespace(models=models)}
    llm.last_usage = {}
    return llm, models


@pytest.fixture(autouse=True)
def gemini(monkeypatch):
    for name, value in {
        "GEMINI_MODEL": "gemini-test",
        "GEMINI_MAX_OUTPUT_TOKENS": 16000,
        "GEMINI_TEMPERATURE": None,
        "GEMINI_API_KEY": "key-1",
        "GEMINI_API_KEYS": "",
        "GEMINI_MAX_ATTEMPTS": 8,
        "GEMINI_EXHAUSTED_KEY_RETRY_S": 3600.0,
    }.items():
        monkeypatch.setattr(llm_module.settings, name, value)
    monkeypatch.setattr(llm_module, "_GEMINI_EXHAUSTED_UNTIL", {})
    sleeps = []
    monkeypatch.setattr(llm_module.time, "sleep", sleeps.append)
    return sleeps


EVIDENCE = [{"document_id": "d", "page": 1, "text": "Revenue was 3,875."}]
DIRECT = '{"shape": "direct", "value": "3,875"}'
CLASSIFICATION = '{"question_type": "table", "search_query": "revenue"}'


def test_structured_request_sends_the_json_schema():
    llm, models = _llm(_response(text=CLASSIFICATION))

    assert isinstance(llm.classify("q"), Classification)

    request = models.requests[0]
    assert request["model"] == "gemini-test"
    assert "Classify this financial-document question" in request["contents"]
    config = request["config"]
    assert config["response_mime_type"] == "application/json"
    assert config["response_json_schema"] == Classification.model_json_schema()
    assert config["max_output_tokens"] == 16000
    assert "temperature" not in config


def test_temperature_is_sent_only_when_configured(monkeypatch):
    monkeypatch.setattr(llm_module.settings, "GEMINI_TEMPERATURE", 0.0)
    llm, models = _llm(_response(text=CLASSIFICATION))

    llm.classify("q")

    assert models.requests[0]["config"]["temperature"] == 0.0


def test_reformulate_sends_no_schema():
    llm, models = _llm(_response(text="  broader query \n"))

    assert llm.reformulate("q", "previous", 1) == "broader query"
    assert "response_json_schema" not in models.requests[0]["config"]


def test_usage_counts_thinking_as_output():
    llm, _ = _llm(_response(text=DIRECT, prompt_tokens=400, output_tokens=25, thoughts=75))

    result = llm.extract("q", "text", EVIDENCE)

    assert isinstance(result, ExtractionDirect) and result.value == "3,875"
    assert llm.last_usage == {"prompt_tokens": 400, "completion_tokens": 100}


@pytest.mark.parametrize("question_type", ["numerical", "text"])
def test_truncation_is_an_error_not_an_answer(question_type):
    llm, _ = _llm(_response(text='{"shape": "dir', finish="MAX_TOKENS"))
    with pytest.raises(ModelOutputError, match="max_output_tokens"):
        llm.extract("q", question_type, EVIDENCE)


def test_safety_stop_is_an_error_not_an_answer():
    llm, _ = _llm(_response(text="", finish="SAFETY"))
    with pytest.raises(ModelOutputError, match="SAFETY"):
        llm.extract("q", "text", EVIDENCE)


def test_blocked_prompt_is_an_error():
    llm, _ = _llm(_response(finish=None, block_reason="PROHIBITED_CONTENT"))
    with pytest.raises(ModelOutputError, match="PROHIBITED_CONTENT"):
        llm.classify("q")


@pytest.mark.parametrize("question_type", ["numerical", "text"])
def test_api_failure_propagates_instead_of_scoring_as_insufficient(question_type):
    llm, _ = _llm(FakeQuotaExhausted("400 INVALID_ARGUMENT"))
    with pytest.raises(FakeQuotaExhausted):
        llm.extract("q", question_type, EVIDENCE)


@pytest.mark.parametrize("question_type", ["numerical", "text"])
def test_complete_response_off_schema_degrades_to_insufficient(question_type):
    llm, _ = _llm(_response(text='{"unexpected": true}'))
    assert isinstance(llm.extract("q", question_type, EVIDENCE), ExtractionInsufficient)


def test_complete_response_off_schema_in_classify_raises():
    llm, _ = _llm(_response(text='{"unexpected": true}'))
    with pytest.raises(StructuredOutputError):
        llm.classify("q")


def test_uses_the_same_prompts_as_anthropic(monkeypatch):
    monkeypatch.setattr(llm_module.settings, "GRADE_REQUIRE_ENTITY_MATCH", True)
    llm, models = _llm(_response(text='{"sufficient": false, "confidence": 0.9, "reason": "r"}'))

    llm.grade("q", EVIDENCE)

    assert "If the question names a company" in models.requests[0]["contents"]


# --------------------------------------------------------------- API keys --


def _two_keys(monkeypatch, first, second):
    monkeypatch.setattr(llm_module.settings, "GEMINI_API_KEYS", "key-1, key-2")
    llm = GeminiLLM.__new__(GeminiLLM)
    llm._clients = {"key-1": SimpleNamespace(models=first), "key-2": SimpleNamespace(models=second)}
    llm.last_usage = {}
    return llm


def test_keys_come_from_the_list_then_the_single_key_without_repeats(monkeypatch):
    monkeypatch.setattr(llm_module.settings, "GEMINI_API_KEYS", " key-2 ,key-3,, key-2")
    monkeypatch.setattr(llm_module.settings, "GEMINI_API_KEY", "key-1")

    assert llm_module.gemini_keys() == ["key-2", "key-3", "key-1"]


def test_a_daily_quota_moves_the_request_to_the_next_key(monkeypatch):
    first, second = FakeModels(DAILY), FakeModels(_response(text=CLASSIFICATION))
    llm = _two_keys(monkeypatch, first, second)

    assert isinstance(llm.classify("q"), Classification)

    assert len(first.requests) == 1 and len(second.requests) == 1
    assert llm_module.gemini_keys_exhausted() == 1


def test_an_exhausted_key_stays_skipped_for_the_next_question(monkeypatch):
    first, second = FakeModels(DAILY), FakeModels(_response(text=CLASSIFICATION))
    _two_keys(monkeypatch, first, second).classify("q")

    # The graph builds a new LLM per question; the exhausted key is remembered.
    _two_keys(monkeypatch, first, second).classify("q")

    assert len(first.requests) == 1 and len(second.requests) == 2


def test_a_per_minute_limit_waits_the_named_delay_on_the_same_key(monkeypatch, gemini):
    first, second = FakeModels(PER_MINUTE, _response(text=CLASSIFICATION)), FakeModels(_response(text=CLASSIFICATION))
    llm = _two_keys(monkeypatch, first, second)

    assert isinstance(llm.classify("q"), Classification)

    assert gemini == [8.0]
    assert len(first.requests) == 2 and second.requests == []
    assert llm_module.gemini_keys_exhausted() == 0


def test_a_per_minute_limit_that_persists_propagates(monkeypatch, gemini):
    monkeypatch.setattr(llm_module.settings, "GEMINI_MAX_ATTEMPTS", 3)
    llm, models = _llm(PER_MINUTE)

    with pytest.raises(Fake429):
        llm.classify("q")

    assert len(models.requests) == 3 and len(gemini) == 2


def test_every_key_out_of_daily_quota_is_an_error_not_an_answer(monkeypatch):
    llm = _two_keys(monkeypatch, FakeModels(DAILY), FakeModels(DAILY))

    with pytest.raises(GeminiQuotaExhausted, match="all 2"):
        llm.extract("q", "text", EVIDENCE)


def test_an_exhausted_key_is_tried_again_after_the_retry_period(monkeypatch):
    monkeypatch.setattr(llm_module.settings, "GEMINI_EXHAUSTED_KEY_RETRY_S", 0.0)
    first = FakeModels(DAILY, _response(text=CLASSIFICATION))
    llm = _two_keys(monkeypatch, first, FakeModels(DAILY))

    with pytest.raises(GeminiQuotaExhausted):
        llm.classify("q")
    monkeypatch.setattr(llm_module, "_GEMINI_EXHAUSTED_UNTIL", {})

    assert isinstance(llm.classify("q"), Classification)


def test_no_key_configured_is_an_error(monkeypatch):
    monkeypatch.setattr(llm_module.settings, "GEMINI_API_KEY", "")
    llm = GeminiLLM.__new__(GeminiLLM)
    llm._clients, llm.last_usage = {}, {}

    with pytest.raises(GeminiQuotaExhausted, match="no Gemini API key"):
        llm.classify("q")


def test_clients_retry_only_server_errors_inside_the_sdk(monkeypatch):
    captured = []

    class FakeClient:
        def __init__(self, **kwargs):
            captured.append(kwargs)
            self.models = FakeModels(_response(text=CLASSIFICATION))

    monkeypatch.setattr("google.genai.Client", FakeClient)
    monkeypatch.setattr(llm_module.settings, "GEMINI_API_KEY", "test-key")

    GeminiLLM().classify("q")

    retry = captured[0]["http_options"]["retry_options"]
    assert captured[0]["api_key"] == "test-key"
    assert retry["attempts"] == 8
    assert 429 not in retry["http_status_codes"] and 503 in retry["http_status_codes"]


def test_get_llm_selects_gemini(monkeypatch):
    monkeypatch.setattr("google.genai.Client", lambda **kwargs: SimpleNamespace())
    monkeypatch.setattr(llm_module.settings, "LLM_PROVIDER", "gemini")

    assert isinstance(llm_module.get_llm(), GeminiLLM)


def test_health_reports_key_counts_but_never_keys(monkeypatch):
    from fastapi.testclient import TestClient

    from app import main

    monkeypatch.setattr(main.settings, "LLM_PROVIDER", "gemini")
    monkeypatch.setattr(main.settings, "GEMINI_API_KEYS", "secret-one,secret-two")

    body = TestClient(main.app).get("/health").json()

    assert body["config"]["model"] == "gemini-test"
    assert body["gemini"] == {"api_keys": 3, "keys_out_of_daily_quota": 0}
    assert "secret" not in str(body)


def test_health_flags_a_missing_key(monkeypatch):
    from fastapi.testclient import TestClient

    from app import main

    monkeypatch.setattr(main.settings, "LLM_PROVIDER", "gemini")
    monkeypatch.setattr(main.settings, "GEMINI_API_KEY", "")

    body = TestClient(main.app).get("/health").json()

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


def test_a_model_written_value_in_a_fixed_field_does_not_reject_the_formula():
    """Gemini wrote \"shape\": \"percentage_change\" into ExtractionCalculated,
    whose shape can only be \"calculated\"; the correct formula was declined as
    'Could not extract numeric operands.'"""
    reply = (
        '{"shape": "percentage_change", "formula": "(25282320-22095416)/22095416*100", '
        '"operand_count": 2, "evidence_indexes": [1]}'
    )
    llm, _ = _llm(_response(text=reply))

    result = llm.extract("q", "numerical", EVIDENCE)

    assert result.shape == "calculated"
    assert result.formula == "(25282320-22095416)/22095416*100"
    assert result.evidence_indexes == [1]


def test_a_field_with_a_real_choice_is_still_validated():
    llm, _ = _llm(_response(text='{"shape": "percentage_change", "value": "3,875"}'))

    assert isinstance(llm.extract("q", "text", EVIDENCE), ExtractionInsufficient)


def test_invalid_json_is_a_structured_output_error():
    llm, _ = _llm(_response(text="not json"))

    with pytest.raises(StructuredOutputError):
        llm.classify("q")
