"""Regression tests for the no-reindex answer-shape optimization."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app import llm as llm_module
from app.llm import AnthropicLLM, GroqLLM, OllamaLLM, answer_shape_rules


EVIDENCE = [{"document_id": "doc-1", "page": 7, "text": "Evidence text."}]


def _capture_json_prompt(provider_cls):
    provider = provider_cls.__new__(provider_cls)
    captured = []

    def generate(prompt):
        captured.append(prompt)
        return {"shape": "direct", "value": "answer"}

    provider._generate = generate
    provider._mock = SimpleNamespace()
    provider.extract(
        "Which company performed the audit, and which page supports it?",
        "text",
        EVIDENCE,
    )
    return captured[0]


def test_guidance_is_empty_when_feature_is_disabled(monkeypatch):
    monkeypatch.setattr(llm_module.settings, "ANSWER_SHAPE_GUIDANCE", False)
    assert answer_shape_rules() == ""


def test_guidance_targets_observed_answer_shape_failures(monkeypatch):
    monkeypatch.setattr(llm_module.settings, "ANSWER_SHAPE_GUIDANCE", True)
    rules = answer_shape_rules()

    assert "Use direct for one requested fact" in rules
    assert "Use multi_span only when the question explicitly requests multiple" in rules
    assert "never include a page number in value or values" in rules
    assert "currency, unit, and scale" in rules
    assert "full name as written" in rules


@pytest.mark.parametrize("provider_cls", [OllamaLLM, GroqLLM])
def test_json_providers_receive_guidance_only_when_enabled(monkeypatch, provider_cls):
    monkeypatch.setattr(llm_module.settings, "ANSWER_SHAPE_GUIDANCE", False)
    baseline_prompt = _capture_json_prompt(provider_cls)
    assert "answer-shape rules" not in baseline_prompt
    assert "no explanation text.\n\nQuestion:" in baseline_prompt

    monkeypatch.setattr(llm_module.settings, "ANSWER_SHAPE_GUIDANCE", True)
    prompt = _capture_json_prompt(provider_cls)
    assert "answer-shape rules" in prompt
    assert "evidence citation/evidence_indexes" in prompt


def test_anthropic_uses_guidance_without_conflicting_shortest_span_rule(monkeypatch):
    provider = AnthropicLLM.__new__(AnthropicLLM)
    prompts = []

    def structured(schema, prompt):
        prompts.append(prompt)
        return schema(shape="direct", value="answer", evidence_indexes=[1])

    provider._structured = structured
    monkeypatch.setattr(llm_module.settings, "ANSWER_SHAPE_GUIDANCE", True)

    provider.extract("Why did the expense increase?", "text", EVIDENCE)

    assert "answer-shape rules" in prompts[0]
    assert "shortest span" not in prompts[0]
    assert "Set evidence_indexes" in prompts[0]


def test_anthropic_baseline_prompt_is_preserved_when_disabled(monkeypatch):
    provider = AnthropicLLM.__new__(AnthropicLLM)
    prompts = []

    def structured(schema, prompt):
        prompts.append(prompt)
        return schema(shape="direct", value="answer")

    provider._structured = structured
    monkeypatch.setattr(llm_module.settings, "ANSWER_SHAPE_GUIDANCE", False)

    provider.extract("What was revenue?", "text", EVIDENCE)

    assert "shortest span" in prompts[0]
    assert "answer-shape rules" not in prompts[0]
