"""AnthropicLLM must tell a failure to answer apart from an answer that does
not fit the schema, and must build requests Claude Opus 5 accepts.

A rate limit, a refusal, or output truncated at max_tokens has to reach the
caller as an error. Scored as "insufficient evidence" it would count a
question the model never answered as a wrong answer -- which is how a
throttled evaluation run quietly reports near-zero accuracy.
"""

from types import SimpleNamespace

import pytest

from app import llm as llm_module
from app.llm import (
    AnthropicLLM,
    Classification,
    ExtractionDirect,
    ExtractionInsufficient,
    ModelOutputError,
    StructuredOutputError,
)


class FakeRateLimit(Exception):
    """Stands in for anthropic.RateLimitError once the SDK's retries run out."""


class FakeMessages:
    def __init__(self, outcome):
        self.outcome = outcome
        self.requests = []

    def create(self, **request):
        self.requests.append(request)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def _response(text="", stop_reason="end_turn", blocks=None, input_tokens=120, output_tokens=30):
    if blocks is None:
        blocks = [SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)]
    return SimpleNamespace(
        content=blocks,
        stop_reason=stop_reason,
        stop_details=SimpleNamespace(category="cyber") if stop_reason == "refusal" else None,
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=None,
        ),
    )


def _llm(outcome):
    llm = AnthropicLLM.__new__(AnthropicLLM)  # skip the real client: no key, no network
    messages = FakeMessages(outcome)
    llm._client = SimpleNamespace(beta=SimpleNamespace(messages=messages))
    llm._transform_schema = lambda schema: {"title": schema.__name__}
    llm.last_usage = {}
    return llm, messages


@pytest.fixture
def opus5(monkeypatch):
    for name, value in {
        "ANTHROPIC_MODEL": "claude-opus-5",
        "ANTHROPIC_MAX_TOKENS": 16000,
        "ANTHROPIC_EFFORT": "",
        "ANTHROPIC_FALLBACKS": "default",
    }.items():
        monkeypatch.setattr(llm_module.settings, name, value)


EVIDENCE = [{"document_id": "d", "page": 1, "text": "Revenue was 3,875."}]
DIRECT = '{"shape": "direct", "value": "3,875"}'


@pytest.mark.parametrize("question_type", ["numerical", "text"])
def test_api_failure_propagates_instead_of_scoring_as_insufficient(opus5, question_type):
    llm, _ = _llm(FakeRateLimit("429"))
    with pytest.raises(FakeRateLimit):
        llm.extract("q", question_type, EVIDENCE)


def test_api_failure_propagates_from_classify(opus5):
    llm, _ = _llm(FakeRateLimit("529"))
    with pytest.raises(FakeRateLimit):
        llm.classify("q")


@pytest.mark.parametrize("question_type", ["numerical", "text"])
def test_refusal_is_an_error_not_an_answer(opus5, question_type):
    llm, _ = _llm(_response(stop_reason="refusal"))
    with pytest.raises(ModelOutputError, match="declined"):
        llm.extract("q", question_type, EVIDENCE)


@pytest.mark.parametrize("question_type", ["numerical", "text"])
def test_truncation_is_an_error_not_an_answer(opus5, question_type):
    """Truncated JSON would also fail schema validation -- checking stop_reason
    first is what keeps it from scoring as insufficient evidence."""
    llm, _ = _llm(_response(text='{"shape": "dir', stop_reason="max_tokens"))
    with pytest.raises(ModelOutputError, match="max_tokens"):
        llm.extract("q", question_type, EVIDENCE)


@pytest.mark.parametrize("question_type", ["numerical", "text"])
def test_complete_response_off_schema_degrades_to_insufficient(opus5, question_type):
    llm, _ = _llm(_response(text='{"unexpected": true}'))
    assert isinstance(llm.extract("q", question_type, EVIDENCE), ExtractionInsufficient)


def test_complete_response_off_schema_in_classify_raises(opus5):
    llm, _ = _llm(_response(text='{"unexpected": true}'))
    with pytest.raises(StructuredOutputError):
        llm.classify("q")


def test_successful_extraction_returns_value_and_records_usage(opus5):
    llm, _ = _llm(_response(text=DIRECT, input_tokens=400, output_tokens=25))

    result = llm.extract("q", "text", EVIDENCE)

    assert isinstance(result, ExtractionDirect) and result.value == "3,875"
    assert llm.last_usage == {"prompt_tokens": 400, "completion_tokens": 25}


def test_request_is_valid_for_claude_opus_5(opus5):
    llm, messages = _llm(_response(text='{"question_type": "table", "search_query": "revenue"}'))

    llm.classify("q")

    request = messages.requests[0]
    assert not {"temperature", "top_p", "top_k"} & request.keys(), "rejected with a 400 on Opus 5"
    assert "thinking" not in request, "Opus 5 thinks by default; disabling it has failure modes"
    assert request["model"] == "claude-opus-5"
    assert request["max_tokens"] == 16000
    assert request["fallbacks"] == "default"
    assert request["betas"] == ["server-side-fallback-2026-07-01"]
    assert request["output_config"] == {"format": {"type": "json_schema", "schema": {"title": "Classification"}}}


def test_effort_and_fallbacks_are_configurable(opus5, monkeypatch):
    monkeypatch.setattr(llm_module.settings, "ANTHROPIC_EFFORT", "medium")
    monkeypatch.setattr(llm_module.settings, "ANTHROPIC_FALLBACKS", "")
    llm, messages = _llm(_response(text='{"question_type": "text", "search_query": "x"}'))

    assert isinstance(llm.classify("q"), Classification)

    request = messages.requests[0]
    assert request["output_config"]["effort"] == "medium"
    assert "fallbacks" not in request and "betas" not in request


def test_text_after_a_fallback_switch_point_is_the_answer(opus5):
    blocks = [
        SimpleNamespace(type="text", text='{"shape": "dir'),
        SimpleNamespace(type="fallback"),
        SimpleNamespace(type="text", text=DIRECT),
    ]
    llm, _ = _llm(_response(blocks=blocks))

    assert llm.extract("q", "text", EVIDENCE).value == "3,875"


def test_reformulate_reads_only_text_blocks(opus5):
    blocks = [SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text="  broader query \n")]
    llm, _ = _llm(_response(blocks=blocks))

    assert llm.reformulate("q", "previous", 1) == "broader query"


def _prompt(messages) -> str:
    return messages.requests[-1]["messages"][0]["content"]


def test_extraction_asks_for_the_exact_span_without_decoration(opus5):
    """Opus 5 decorated correct answers -- "2019: $2,657 thousand" for gold
    "2,657", "0 per cent (2018: 0 per cent) - per the..." for "0 per cent" --
    and exact match scores every one of those as wrong."""
    llm, messages = _llm(_response(text=DIRECT))
    llm.extract("q", "text", EVIDENCE)

    prompt = _prompt(messages)
    assert "exactly as it is written in the evidence" in prompt
    assert "Do not add labels, years, units" in prompt


def test_grade_prompt_is_unchanged_when_entity_rule_is_off(opus5, monkeypatch):
    monkeypatch.setattr(llm_module.settings, "GRADE_REQUIRE_ENTITY_MATCH", False)
    llm, messages = _llm(_response(text='{"sufficient": true, "confidence": 0.9, "reason": "r"}'))
    llm.grade("q", EVIDENCE)

    prompt = _prompt(messages)
    assert "names a company" not in prompt
    assert "sufficient=false.\n\nQuestion: q" in prompt


def test_grade_prompt_adds_entity_rule_when_enabled(opus5, monkeypatch):
    monkeypatch.setattr(llm_module.settings, "GRADE_REQUIRE_ENTITY_MATCH", True)
    llm, messages = _llm(_response(text='{"sufficient": false, "confidence": 0.9, "reason": "r"}'))
    llm.grade("q", EVIDENCE)

    assert "If the question names a company" in _prompt(messages)


def test_health_reports_the_settings_experiments_vary():
    from fastapi.testclient import TestClient

    from app import main

    config = TestClient(main.app).get("/health").json()["config"]
    assert {"top_k_final", "max_retries", "grade_require_entity_match"} <= config.keys()


@pytest.mark.parametrize(
    "question_type,reply",
    [
        ("text", DIRECT),
        ("numerical", '{"formula": "3875-1200", "operand_count": 2, "evidence_indexes": [1, 2]}'),
    ],
)
def test_extraction_numbers_the_evidence_and_asks_which_passages_were_used(opus5, question_type, reply):
    evidence = EVIDENCE + [{"document_id": "d2", "page": 4, "text": "Costs were 1,200."}]
    llm, messages = _llm(_response(text=reply))

    llm.extract("q", question_type, evidence)

    prompt = _prompt(messages)
    assert "[1] d p1" in prompt and "[2] d2 p4" in prompt
    assert "evidence_indexes" in prompt


def test_evidence_indexes_reach_the_extraction_the_graph_receives(opus5):
    llm, _ = _llm(_response(text='{"shape": "direct", "value": "3,875", "evidence_indexes": [1]}'))
    assert llm.extract("q", "text", EVIDENCE).evidence_indexes == [1]

    llm, _ = _llm(_response(text='{"formula": "3875-1", "operand_count": 2, "evidence_indexes": [1]}'))
    assert llm.extract("q", "numerical", EVIDENCE).evidence_indexes == [1]


def test_answer_endpoint_reports_usage_from_the_trace(monkeypatch):
    """Tested through the endpoint, not the helper: _usage_summary once existed
    and passed its own test while /answer never called it, so every live run
    reported 0 LLM calls and 0 tokens."""
    from fastapi.testclient import TestClient

    from app import main

    async def fake_run_agent(question, document_id=None):
        return {
            "answer": {"answer_type": "insufficient_evidence", "evidence": [], "params": {"reason": "x"}},
            "question_type": "text",
            "retry_count": 0,
            "trace": [
                {"step": "classify", "usage": {"prompt_tokens": 441, "completion_tokens": 46}},
                {"step": "retrieve", "n_hits": 5},
                {"step": "grade", "usage": {"prompt_tokens": 1500, "completion_tokens": 120}},
            ],
        }

    monkeypatch.setattr(main, "run_agent", fake_run_agent)
    body = TestClient(main.app).post("/answer", json={"question": "q"}).json()

    assert body["llm_calls"] == 2
    assert (body["input_tokens"], body["output_tokens"], body["tokens_used"]) == (1941, 166, 2107)


def test_usage_summary_totals_calls_tokens_and_cost(monkeypatch):
    from app import main

    monkeypatch.setattr(main.settings, "LLM_PROVIDER", "anthropic")
    trace = [
        {"step": "classify", "usage": {"prompt_tokens": 100, "completion_tokens": 10}},
        {"step": "retrieve", "n_hits": 5},
        {"step": "grade", "usage": {"prompt_tokens": 900, "completion_tokens": 40}},
    ]

    summary = main._usage_summary(trace)

    assert summary["llm_calls"] == 2
    assert (summary["input_tokens"], summary["output_tokens"], summary["tokens_used"]) == (1000, 50, 1050)
    expected = (
        1000 / 1_000_000 * main.settings.ANTHROPIC_INPUT_USD_PER_MTOK
        + 50 / 1_000_000 * main.settings.ANTHROPIC_OUTPUT_USD_PER_MTOK
    )
    assert summary["cost_usd"] == pytest.approx(expected)
