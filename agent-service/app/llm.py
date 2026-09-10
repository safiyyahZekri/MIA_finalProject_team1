"""
LLM abstraction used by the LangGraph nodes.

Two interchangeable providers implement the same interface:
  - MockLLM       : pure heuristics/regex, zero external dependencies.
                    Default, so the service runs immediately with no API key.
  - AnthropicLLM  : Claude (claude-opus-5 by default) through the official
                    Anthropic SDK, via structured outputs.
  - GeminiLLM     : Gemini (gemini-2.5-flash by default) through Google's
                    official google-genai SDK, with the same prompts.

Only classification, query reformulation, evidence grading, and *proposing*
extraction candidates go through the LLM. The actual arithmetic always goes
through app.calculator.calculate() -- never trusted from the LLM directly.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from typing import List, Literal, Optional

import httpx
from pydantic import BaseModel, Field, ValidationError

from app.config import settings
from app.decomposition import QueryPlan, decomposition_prompt

logger = logging.getLogger("agent-service.llm")

NUMERIC_KEYWORDS = [
    "percent", "%", "increase", "decrease", "difference", "change",
    "how much more", "how many times", "sum of", "total of", "average",
    "ratio", "compared to", "versus", " vs ", "growth", "combined",
]
TABLE_KEYWORDS = [
    "table", "expense", "revenue", "income", "balance", "asset",
    "liability", "cost", "categor", "line item", "margin",
]
LIST_KEYWORDS = ["which", "list", "categories", "items that", "name the", "what are the"]

_NUMBER_RE = re.compile(r"-?\$?\d[\d,]*\.?\d*")
_MONEY_RE = re.compile(r"\$-?\d[\d,]*\.?\d*")


def _looks_like_year(raw_digits: str, value: float) -> bool:
    return raw_digits.isdigit() and len(raw_digits) == 4 and 1900 <= value <= 2100


def _extract_numbers(text: str) -> List[float]:
    """Pull numeric operands out of evidence text for the numerical path.

    Prefers explicit currency amounts ("$3,875") over bare numbers, since
    bare 4-digit numbers in financial text are very often years (e.g. "in
    2021") rather than figures -- those are filtered out unless nothing
    else is available.
    """
    money_vals = []
    for m in _MONEY_RE.findall(text):
        cleaned = m.replace("$", "").replace(",", "")
        try:
            money_vals.append(float(cleaned))
        except ValueError:
            continue
    if len(money_vals) >= 2:
        return money_vals

    generic_vals = []
    for m in _NUMBER_RE.findall(text):
        raw_digits = m.replace("$", "").replace(",", "").replace(".", "")
        cleaned = m.replace("$", "").replace(",", "")
        try:
            value = float(cleaned)
        except ValueError:
            continue
        if _looks_like_year(raw_digits, value):
            continue
        generic_vals.append(value)
    return generic_vals or money_vals


# ---------------------------------------------------------------- schemas --
class Classification(BaseModel):
    question_type: Literal["numerical", "table", "text"]
    search_query: str


class EvidenceGrade(BaseModel):
    sufficient: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class ExtractionDirect(BaseModel):
    shape: Literal["direct"] = "direct"
    value: str
    # 1-based numbers of the evidence passages the answer was taken from, as
    # numbered in the extraction prompt. Empty means the model named none.
    evidence_indexes: List[int] = Field(default_factory=list)


class ExtractionMultiSpan(BaseModel):
    shape: Literal["multi_span"] = "multi_span"
    values: List[str]
    evidence_indexes: List[int] = Field(default_factory=list)


class ExtractionCalculated(BaseModel):
    shape: Literal["calculated"] = "calculated"
    formula: str  # numeric literals only, e.g. "(3875-3410)/3410*100"
    operand_count: int
    evidence_indexes: List[int] = Field(default_factory=list)


class ExtractionInsufficient(BaseModel):
    shape: Literal["insufficient"] = "insufficient"
    reason: str


class ExtractionChoice(BaseModel):
    """Flat carrier for the three extraction shapes.

    A tool-calling API takes one JSON schema per call, and a bare
    `Union[...]` is not a schema -- handing one to
    `with_structured_output()` raises TypeError before any request goes
    out. A discriminated union works in pydantic but nests `oneOf` inside
    the tool schema, which providers accept unevenly, so the three shapes
    are flattened into optional fields here and narrowed back to the
    concrete models by `as_extraction()`. Callers keep receiving
    ExtractionDirect / ExtractionMultiSpan / ExtractionInsufficient, so
    the isinstance branches in app.graph are unaffected.
    """

    shape: Literal["direct", "multi_span", "insufficient"]
    value: Optional[str] = None
    values: Optional[List[str]] = None
    reason: Optional[str] = None
    evidence_indexes: List[int] = Field(default_factory=list)

    def as_extraction(self):
        if self.shape == "direct" and self.value is not None:
            return ExtractionDirect(value=str(self.value), evidence_indexes=self.evidence_indexes)
        if self.shape == "multi_span" and self.values:
            return ExtractionMultiSpan(
                values=[str(v) for v in self.values], evidence_indexes=self.evidence_indexes
            )
        if self.shape == "insufficient":
            return ExtractionInsufficient(reason=self.reason or "Insufficient evidence.")
        # A shape was declared but its payload field came back empty.
        return ExtractionInsufficient(
            reason=f"Model returned shape '{self.shape}' without a usable payload."
        )


# ------------------------------------------------------------- mock impl --
class MockLLM:
    """Deterministic, offline heuristic stand-in for a real LLM."""

    def decompose(self, question: str) -> QueryPlan:
        # Mock mode makes no claims about semantic query planning.
        return QueryPlan()

    def classify(self, question: str) -> Classification:
        lowered = question.lower()
        # "which/list ..." style questions name items rather than compute a
        # number, even if a word like "increased" also appears -- so they're
        # checked first and routed as table/text rather than numerical.
        if any(k in lowered for k in LIST_KEYWORDS):
            qtype = "table" if any(k in lowered for k in TABLE_KEYWORDS) else "text"
        elif any(k in lowered for k in NUMERIC_KEYWORDS):
            qtype = "numerical"
        elif any(k in lowered for k in TABLE_KEYWORDS):
            qtype = "table"
        else:
            qtype = "text"
        return Classification(question_type=qtype, search_query=question)

    def reformulate(self, question: str, previous_query: str, attempt: int) -> str:
        # Broaden by stripping the most specific (longest) word each retry.
        words = previous_query.split()
        if len(words) <= 3:
            return question  # fall back to the raw question
        longest = max(words, key=len)
        broadened = [w for w in words if w != longest]
        return " ".join(broadened) if broadened else question

    def grade(self, question: str, evidence: List[dict]) -> EvidenceGrade:
        if not evidence:
            return EvidenceGrade(sufficient=False, confidence=0.0, reason="No evidence retrieved.")
        top_score = max(e.get("score", 0.0) for e in evidence)
        sufficient = top_score >= settings.MIN_EVIDENCE_SCORE
        reason = (
            f"Top evidence score {top_score:.2f} "
            f"{'meets' if sufficient else 'is below'} threshold {settings.MIN_EVIDENCE_SCORE}."
        )
        return EvidenceGrade(sufficient=sufficient, confidence=top_score, reason=reason)

    def extract(self, question: str, question_type: str, evidence: List[dict]):
        combined_text = " ".join(e.get("text", "") for e in evidence)
        lowered = question.lower()

        # "which ... categories/items" is a naming question, not arithmetic,
        # even when it shares vocabulary ("increased") with numeric questions.
        if any(k in lowered for k in LIST_KEYWORDS):
            match = re.search(r":\s*([A-Z][\w &/-]*(?:,\s*[A-Z][\w &/-]*)+)", combined_text)
            if match:
                values = [v.strip() for v in match.group(1).split(",")]
                return ExtractionMultiSpan(values=values)
            # fall through to other extraction strategies if no list found

        if question_type == "numerical":
            numbers = _extract_numbers(combined_text)
            if len(numbers) < 2:
                return ExtractionInsufficient(
                    reason="Could not find two numeric operands required for the calculation."
                )
            a, b = numbers[0], numbers[1]
            if any(k in lowered for k in ["percent", "%", "increase", "decrease", "change", "growth"]):
                formula = f"({a}-{b})/{b}*100"
            elif "sum" in lowered or "total" in lowered or "combined" in lowered:
                formula = f"{a}+{b}"
            elif "ratio" in lowered or "times" in lowered:
                formula = f"{a}/{b}"
            else:
                formula = f"{a}-{b}"
            return ExtractionCalculated(formula=formula, operand_count=2)

        if any(k in lowered for k in LIST_KEYWORDS):
            # crude list extraction: capitalized comma-separated run near the end of a sentence
            match = re.search(r":\s*([A-Z][\w &/-]*(?:,\s*[A-Z][\w &/-]*)+)", combined_text)
            if match:
                values = [v.strip() for v in match.group(1).split(",")]
                return ExtractionMultiSpan(values=values)
            return ExtractionInsufficient(reason="Could not find a list of items answering the question.")

        numbers = _extract_numbers(combined_text)
        if numbers:
            # prefer a $ value if present
            dollar_match = re.search(r"\$[\d,]+\.?\d*[MBK]?", combined_text)
            value = dollar_match.group(0) if dollar_match else numbers[0]
            return ExtractionDirect(value=str(value))

        return ExtractionInsufficient(reason="No factual value found in the retrieved evidence.")


# ------------------------------------------------------------ ollama impl --
class OllamaError(RuntimeError):
    pass


class OllamaLLM:
    """Real local LLM via Ollama (e.g. qwen2.5), zero cloud API key needed.

    Uses Ollama's JSON-mode generation (`format: "json"`) and validates
    every response against the same pydantic schemas the Anthropic provider
    uses. A malformed reply is retried once with a corrective prompt; if it
    still can't be parsed, this provider degrades to MockLLM's heuristic
    for that single call rather than raising -- a flaky/undersized local
    model must never crash the graph.
    """

    def __init__(self) -> None:
        self.base_url = settings.OLLAMA_BASE_URL.rstrip("/")
        self.model = settings.OLLAMA_MODEL
        self._client = httpx.Client(timeout=settings.OLLAMA_TIMEOUT_S)
        self._mock = MockLLM()  # safety net only, not the source of answers
        self.last_usage: dict = {}

    def ping(self) -> bool:
        """Best-effort connectivity check, used by GET /health."""
        try:
            resp = self._client.get(f"{self.base_url}/api/tags", timeout=3)
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    def _generate(self, prompt: str) -> dict:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {"temperature": 0},
        }
        last_err: Exception | None = None
        for attempt in range(2):
            try:
                resp = self._client.post(f"{self.base_url}/api/generate", json=payload)
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                raise OllamaError(
                    f"Could not reach Ollama at {self.base_url} with model "
                    f"'{self.model}' -- is `ollama serve` running and has the "
                    f"model been pulled (`ollama pull {self.model}`)? {exc}"
                ) from exc
            data = resp.json()
            self.last_usage = {
                "prompt_tokens": data.get("prompt_eval_count"),
                "completion_tokens": data.get("eval_count"),
            }
            try:
                return json.loads(data["response"])
            except (KeyError, json.JSONDecodeError) as exc:
                last_err = exc
                payload["prompt"] = (
                    prompt
                    + f"\n\nYour previous reply was not valid JSON ({exc}). "
                    "Reply with ONLY the JSON object and nothing else."
                )
        raise OllamaError(f"Model returned non-JSON output twice: {last_err}")

    def decompose(self, question: str) -> QueryPlan:
        return QueryPlan.model_validate(self._generate(decomposition_prompt(question)))

    def classify(self, question: str) -> Classification:
        prompt = (
            "Classify this financial-document question for retrieval routing.\n"
            'Return ONLY JSON: {"question_type": "numerical"|"table"|"text", '
            '"search_query": "<focused retrieval query>"}\n'
            "- numerical: requires arithmetic (comparison, % change, sum, ratio, sort, etc.)\n"
            "- table: the answer likely lives in a financial table/line item, no math needed\n"
            "- text: narrative/prose answer\n\n"
            f"Question: {question}"
        )
        try:
            return Classification.model_validate(self._generate(prompt))
        except (OllamaError, ValidationError, KeyError, TypeError) as exc:
            logger.warning("Ollama classify() failed, falling back to mock: %s", exc)
            return self._mock.classify(question)

    def reformulate(self, question: str, previous_query: str, attempt: int) -> str:
        prompt = (
            "The previous retrieval query returned insufficient evidence.\n"
            f"Original question: {question}\nPrevious query: {previous_query}\n"
            'Return ONLY JSON: {"query": "<one broader or differently phrased retrieval query>"}'
        )
        try:
            return str(self._generate(prompt)["query"])
        except (OllamaError, KeyError, TypeError) as exc:
            logger.warning("Ollama reformulate() failed, falling back to mock: %s", exc)
            return self._mock.reformulate(question, previous_query, attempt)

    def grade(self, question: str, evidence: List[dict]) -> EvidenceGrade:
        snippets = "\n---\n".join(
            f"[{e.get('document_id')} p{e.get('page')} score={e.get('score')}]\n{e.get('text', '')}"
            for e in evidence
        ) or "(no evidence retrieved)"
        prompt = (
            "Judge whether the evidence below is sufficient to confidently and faithfully "
            "answer the question. Be strict: if the exact fact/number needed isn't present, "
            "sufficient must be false.\n"
            'Return ONLY JSON: {"sufficient": true|false, "confidence": 0.0-1.0, "reason": "<short reason>"}\n\n'
            f"Question: {question}\n\nEvidence:\n{snippets}"
        )
        try:
            return EvidenceGrade.model_validate(self._generate(prompt))
        except (OllamaError, ValidationError, KeyError, TypeError) as exc:
            logger.warning("Ollama grade() failed, falling back to mock: %s", exc)
            return self._mock.grade(question, evidence)

    def extract(self, question: str, question_type: str, evidence: List[dict]):
        snippets = "\n---\n".join(
            f"[{e.get('document_id')} p{e.get('page')}]\n{e.get('text', '')}" for e in evidence
        ) or "(no evidence retrieved)"

        if question_type == "numerical":
            prompt = (
                "Extract the numeric operands from the evidence needed to answer the question "
                "and express the calculation as a pure-arithmetic formula using ONLY the literal "
                "numbers found in the evidence (e.g. '(3875-3410)/3410*100'). Do NOT compute the "
                "result yourself -- a separate deterministic tool will evaluate the formula.\n"
                'Return ONLY JSON: {"formula": "<arithmetic expression>", "operand_count": <int>}\n\n'
                f"Question: {question}\n\nEvidence:\n{snippets}"
            )
            try:
                data = self._generate(prompt)
                return ExtractionCalculated(
                    formula=str(data["formula"]), operand_count=int(data["operand_count"])
                )
            except (OllamaError, KeyError, TypeError, ValueError, ValidationError) as exc:
                logger.warning("Ollama extract(numerical) failed, falling back to mock: %s", exc)
                return self._mock.extract(question, question_type, evidence)

        prompt = (
            "Answer the question using ONLY the evidence below. Choose exactly one shape:\n"
            '- {"shape": "direct", "value": "<single value>"}\n'
            '- {"shape": "multi_span", "values": ["<value1>", "<value2>", ...]}\n'
            '- {"shape": "insufficient", "reason": "<why the evidence doesn\'t answer it>"}\n'
            "Return ONLY one JSON object matching one of the shapes above -- no explanation text.\n\n"
            f"Question: {question}\n\nEvidence:\n{snippets}"
        )
        try:
            data = self._generate(prompt)
            shape = data.get("shape")
            if shape == "direct":
                return ExtractionDirect(value=str(data["value"]))
            if shape == "multi_span":
                return ExtractionMultiSpan(values=[str(v) for v in data["values"]])
            if shape == "insufficient":
                return ExtractionInsufficient(reason=str(data.get("reason", "Insufficient evidence.")))
            raise ValueError(f"Unknown shape '{shape}'")
        except (OllamaError, KeyError, TypeError, ValueError, ValidationError) as exc:
            logger.warning("Ollama extract() failed, falling back to mock: %s", exc)
            return self._mock.extract(question, question_type, evidence)


# ------------------------------------------------------------ groq impl ---
class GroqError(RuntimeError):
    pass


class GroqLLM:
    """Hosted LLM via Groq's OpenAI-compatible chat-completions API.

    Same reliability contract as OllamaLLM: every response is requested in
    JSON mode, validated against the same pydantic schemas used everywhere
    else, retried once on malformed output, and degraded to MockLLM's
    heuristic as a last resort rather than ever raising out of the graph.
    A shared/hosted key means the agent's reasoning no longer depends on
    any single teammate's machine being on.
    """

    def __init__(self) -> None:
        if not settings.GROQ_API_KEY:
            raise GroqError(
                "GROQ_API_KEY is not set. Get one from https://console.groq.com "
                "and set it in .env before using LLM_PROVIDER=groq."
            )
        self.base_url = settings.GROQ_BASE_URL.rstrip("/")
        self.model = settings.GROQ_MODEL
        self._client = httpx.Client(
            timeout=settings.GROQ_TIMEOUT_S,
            headers={
                "Authorization": f"Bearer {settings.GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
        )
        self._mock = MockLLM()  # safety net only, not the source of answers
        self.last_usage: dict = {}

    def ping(self) -> bool:
        """Best-effort connectivity + key check, used by GET /health."""
        try:
            resp = self._client.get(f"{self.base_url}/models", timeout=5)
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    def _generate(self, prompt: str) -> dict:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
            "temperature": 0,
        }
        last_err: Exception | None = None
        for attempt in range(2):
            try:
                resp = self._client.post(f"{self.base_url}/chat/completions", json=payload)
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                detail = ""
                if isinstance(exc, httpx.HTTPStatusError):
                    detail = f" ({exc.response.status_code}: {exc.response.text[:200]})"
                raise GroqError(
                    f"Could not reach Groq at {self.base_url} with model "
                    f"'{self.model}'{detail} -- check GROQ_API_KEY and GROQ_MODEL "
                    f"in .env: {exc}"
                ) from exc
            data = resp.json()
            usage = data.get("usage", {})
            self.last_usage = {
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
            }
            try:
                content = data["choices"][0]["message"]["content"]
                return json.loads(content)
            except (KeyError, IndexError, json.JSONDecodeError) as exc:
                last_err = exc
                payload["messages"] = [
                    {
                        "role": "user",
                        "content": prompt
                        + f"\n\nYour previous reply was not valid JSON ({exc}). "
                        "Reply with ONLY the JSON object and nothing else.",
                    }
                ]
        raise GroqError(f"Model returned non-JSON output twice: {last_err}")

    def decompose(self, question: str) -> QueryPlan:
        return QueryPlan.model_validate(self._generate(decomposition_prompt(question)))

    def classify(self, question: str) -> Classification:
        prompt = (
            "Classify this financial-document question for retrieval routing.\n"
            'Return ONLY JSON: {"question_type": "numerical"|"table"|"text", '
            '"search_query": "<focused retrieval query>"}\n'
            "- numerical: requires arithmetic (comparison, % change, sum, ratio, sort, etc.)\n"
            "- table: the answer likely lives in a financial table/line item, no math needed\n"
            "- text: narrative/prose answer\n\n"
            f"Question: {question}"
        )
        try:
            return Classification.model_validate(self._generate(prompt))
        except (GroqError, ValidationError, KeyError, TypeError) as exc:
            logger.warning("Groq classify() failed, falling back to mock: %s", exc)
            return self._mock.classify(question)

    def reformulate(self, question: str, previous_query: str, attempt: int) -> str:
        prompt = (
            "The previous retrieval query returned insufficient evidence.\n"
            f"Original question: {question}\nPrevious query: {previous_query}\n"
            'Return ONLY JSON: {"query": "<one broader or differently phrased retrieval query>"}'
        )
        try:
            return str(self._generate(prompt)["query"])
        except (GroqError, KeyError, TypeError) as exc:
            logger.warning("Groq reformulate() failed, falling back to mock: %s", exc)
            return self._mock.reformulate(question, previous_query, attempt)

    def grade(self, question: str, evidence: List[dict]) -> EvidenceGrade:
        snippets = "\n---\n".join(
            f"[{e.get('document_id')} p{e.get('page')} score={e.get('score')}]\n{e.get('text', '')}"
            for e in evidence
        ) or "(no evidence retrieved)"
        prompt = (
            "Judge whether the evidence below is sufficient to confidently and faithfully "
            "answer the question. Be strict: if the exact fact/number needed isn't present, "
            "sufficient must be false.\n"
            'Return ONLY JSON: {"sufficient": true|false, "confidence": 0.0-1.0, "reason": "<short reason>"}\n\n'
            f"Question: {question}\n\nEvidence:\n{snippets}"
        )
        try:
            return EvidenceGrade.model_validate(self._generate(prompt))
        except (GroqError, ValidationError, KeyError, TypeError) as exc:
            logger.warning("Groq grade() failed, falling back to mock: %s", exc)
            return self._mock.grade(question, evidence)

    def extract(self, question: str, question_type: str, evidence: List[dict]):
        snippets = "\n---\n".join(
            f"[{e.get('document_id')} p{e.get('page')}]\n{e.get('text', '')}" for e in evidence
        ) or "(no evidence retrieved)"

        if question_type == "numerical":
            prompt = (
                "Extract the numeric operands from the evidence needed to answer the question "
                "and express the calculation as a pure-arithmetic formula using ONLY the literal "
                "numbers found in the evidence (e.g. '(3875-3410)/3410*100'). Do NOT compute the "
                "result yourself -- a separate deterministic tool will evaluate the formula.\n"
                'Return ONLY JSON: {"formula": "<arithmetic expression>", "operand_count": <int>}\n\n'
                f"Question: {question}\n\nEvidence:\n{snippets}"
            )
            try:
                data = self._generate(prompt)
                return ExtractionCalculated(
                    formula=str(data["formula"]), operand_count=int(data["operand_count"])
                )
            except (GroqError, KeyError, TypeError, ValueError, ValidationError) as exc:
                logger.warning("Groq extract(numerical) failed, falling back to mock: %s", exc)
                return self._mock.extract(question, question_type, evidence)

        prompt = (
            "Answer the question using ONLY the evidence below. Choose exactly one shape:\n"
            '- {"shape": "direct", "value": "<single value>"}\n'
            '- {"shape": "multi_span", "values": ["<value1>", "<value2>", ...]}\n'
            '- {"shape": "insufficient", "reason": "<why the evidence doesn\'t answer it>"}\n'
            "Return ONLY one JSON object matching one of the shapes above -- no explanation text.\n\n"
            f"Question: {question}\n\nEvidence:\n{snippets}"
        )
        try:
            data = self._generate(prompt)
            shape = data.get("shape")
            if shape == "direct":
                return ExtractionDirect(value=str(data["value"]))
            if shape == "multi_span":
                return ExtractionMultiSpan(values=[str(v) for v in data["values"]])
            if shape == "insufficient":
                return ExtractionInsufficient(reason=str(data.get("reason", "Insufficient evidence.")))
            raise ValueError(f"Unknown shape '{shape}'")
        except (GroqError, KeyError, TypeError, ValueError, ValidationError) as exc:
            logger.warning("Groq extract() failed, falling back to mock: %s", exc)
            return self._mock.extract(question, question_type, evidence)


# --------------------------------------------------------- anthropic impl --
class StructuredOutputError(ValueError):
    """The model answered, but not in the requested schema."""


class ModelOutputError(RuntimeError):
    """The model produced no usable answer at all: it declined the request
    (after any server-side fallback), or its output was cut off at max_tokens.

    Raised rather than folded into insufficient evidence: a question the model
    never answered must surface as an error, not score as a wrong answer."""


class _PromptedLLM:
    """Prompts and structured-output handling shared by the hosted providers
    that return schema-constrained JSON (AnthropicLLM, GeminiLLM).

    A subclass supplies `_create(prompt, schema=None) -> str`: the complete
    answer text, or an exception. API failures propagate, and refused,
    blocked or truncated output raises ModelOutputError, so the caller records
    an error. Only a complete response that does not fit the schema becomes
    an insufficient-evidence extraction.

    Enhancements from eval-service/FAILURE_ANALYSIS_E2E.md are switched by
    settings (ANSWER_FORMAT_FIXES, GRADE_COMPANY_CONTEXT,
    GRADE_REQUIRE_NAMED_TABLE), so an experiment can turn on exactly one.
    With all of them off, the prompts are the ones run `full-100-v1` measured.
    """

    last_usage: dict

    def _create(self, prompt: str, schema=None) -> str:
        raise NotImplementedError

    def _structured(self, schema, prompt: str):
        text = self._create(prompt, schema)
        try:
            return schema.model_validate_json(text)
        except ValidationError as exc:
            raise StructuredOutputError(f"{schema.__name__}: {exc}") from exc

    def decompose(self, question: str) -> QueryPlan:
        return self._structured(QueryPlan, decomposition_prompt(question))

    def classify(self, question: str) -> Classification:
        return self._structured(
            Classification,
            "Classify this financial-document question for retrieval routing.\n"
            "question_type must be exactly one of: numerical, table, text.\n"
            "- numerical: requires arithmetic (comparison, % change, sum, ratio, etc.)\n"
            "- table: answer likely lives in a financial table/line item, no math needed\n"
            "- text: narrative/prose answer\n"
            "search_query: a focused retrieval query derived from the question.\n\n"
            f"Question: {question}",
        )

    def reformulate(self, question: str, previous_query: str, attempt: int) -> str:
        query = self._create(
            "The previous retrieval query returned insufficient evidence.\n"
            f"Original question: {question}\n"
            f"Previous query: {previous_query}\n"
            "Propose ONE broader or differently-phrased retrieval query "
            "(no explanation, just the query text)."
        ).strip()
        return query or previous_query

    def grade(self, question: str, evidence: List[dict]) -> EvidenceGrade:
        snippets = "\n---\n".join(
            f"[{e.get('document_id')} p{e.get('page')} score={e.get('score')}]\n{e.get('text','')}"
            for e in evidence
        ) or "(no evidence retrieved)"
        rules: List[str] = []
        if settings.ANSWER_FORMAT_FIXES:
            # A013: the grader refused a passage holding the answer because the
            # question also asked for a page, which the citation supplies.
            rules.append(
                "Page numbers come from the citations, so a question asking which page "
                "supports the answer does not need a page number in the evidence.\n"
            )
        if settings.GRADE_REQUIRE_ENTITY_MATCH and settings.GRADE_COMPANY_CONTEXT:
            # A035, A001: correct passages were refused only because their page
            # never names the company. Evidence tied to a different company is
            # still refused, which is what stopped A017.
            rules.append(
                "If the question names a company, never accept evidence that belongs to a "
                "different company: when any passage shows the figures are another "
                "company's (its name, or segments, products or a fiscal year-end that "
                "belong to another company), sufficient=false. A passage that names no "
                "company may be accepted when it fits the question exactly and nothing in "
                "the evidence points to a different company; state in the reason what ties "
                "it to the named company, or that nothing contradicts it.\n"
            )
        elif settings.GRADE_REQUIRE_ENTITY_MATCH:
            # Many excerpts never name their company, so figures that fit the
            # question can come from another company's table entirely.
            rules.append(
                "If the question names a company, the evidence must be identifiably that "
                "company's: its name, or something unambiguous about it, appears in the "
                "evidence. Figures that fit the question but cannot be tied to the named "
                "company are not sufficient.\n"
            )
        if settings.GRADE_REQUIRE_NAMED_TABLE:
            # A068: a three-year table was accepted for a question about the
            # five-year "financial highlights" table, at confidence 0.58.
            rules.append(
                "If the question names a specific table, statement or section (for example "
                "'financial highlights'), the evidence must come from that table or section; "
                "a different table with similar figures or fewer periods is not sufficient.\n"
            )
        return self._structured(
            EvidenceGrade,
            "Judge whether the evidence below is sufficient to confidently and "
            "faithfully answer the question. Be strict: if the exact fact/number "
            "needed isn't present, sufficient=false.\n"
            f"{''.join(rules)}\n"
            f"Question: {question}\n\nEvidence:\n{snippets}",
        )

    def extract(self, question: str, question_type: str, evidence: List[dict]):
        # Numbered, so extraction can name the passages its answer came from
        # and the answer can cite exactly those.
        snippets = "\n---\n".join(
            f"[{i}] {e.get('document_id')} p{e.get('page')}\n{e.get('text','')}"
            for i, e in enumerate(evidence, start=1)
        )
        if question_type == "numerical":
            calculation_rules = ""
            if settings.ANSWER_FORMAT_FIXES:
                # A014 used rounded text figures ($84.7M) over the exact table
                # (84,684), and A007 counted three yearly changes in a
                # two-year period.
                calculation_rules = (
                    "When a figure appears both in a table and rounded in the text, take it "
                    "from the table.\n"
                    "A change over a period from one year to a later year covers only the "
                    "year-on-year changes inside that period: from 2017 to 2019 means "
                    "2017 to 2018 and 2018 to 2019.\n"
                )
            try:
                return self._structured(
                    ExtractionCalculated,
                    "Extract the numeric operands from the evidence needed to answer the "
                    "question and express the calculation as a pure-arithmetic formula "
                    "using ONLY the literal numbers found in the evidence "
                    "(e.g. '(3875-3410)/3410*100'). Do not compute the result yourself.\n"
                    f"{calculation_rules}"
                    "Set evidence_indexes to the numbers of the passages the operands were "
                    "taken from.\n\n"
                    f"Question: {question}\n\nEvidence:\n{snippets}",
                )
            except StructuredOutputError:
                return ExtractionInsufficient(reason="Could not extract numeric operands.")

        format_rules = ""
        if settings.ANSWER_FORMAT_FIXES:
            # A086 dropped "million" from "$12.2 million" under the exact-span
            # rule, and A013 added the page marker "p1" as an answer value.
            format_rules = (
                "Keep a scale word or percent sign that follows the number in the evidence "
                "(\"$12.2 million\", \"3.1%\"); it is part of the span. Do not put page "
                "numbers in the answer; the citations carry them.\n"
            )
        # Let the model choose direct vs multi_span vs insufficient.
        try:
            choice = self._structured(
                ExtractionChoice,
                "Answer the question using ONLY the evidence below. Set shape to "
                "exactly one of:\n"
                "- direct: a single fact/value -- fill `value`\n"
                "- multi_span: two or more distinct values/items -- fill `values`\n"
                "- insufficient: the evidence doesn't contain the answer -- fill `reason`\n"
                "Leave the fields belonging to the other shapes unset.\n"
                # Opus 5 writes fuller answers by default -- "2019: $2,657
                # thousand" for a cell reading "2,657" -- which exact match
                # scores as wrong even when the fact is right.
                "Copy each value exactly as it is written in the evidence, as the shortest "
                "span that answers the question. Do not add labels, years, units, currency "
                "symbols or explanation that are not part of that span.\n"
                f"{format_rules}"
                "Set evidence_indexes to the numbers of the passages the answer was taken "
                "from.\n\n"
                f"Question: {question}\n\nEvidence:\n{snippets}",
            )
        except StructuredOutputError as exc:
            logger.warning("%s extract() returned no usable structure: %s", type(self).__name__, exc)
            return ExtractionInsufficient(reason=f"Model response did not match the schema: {exc}")
        return choice.as_extraction()


class AnthropicLLM(_PromptedLLM):
    """Claude through the official Anthropic SDK, with structured outputs.

    Built for Claude Opus 5, whose request surface differs from earlier models
    in ways that fail loudly or silently:
      * sampling parameters (temperature, top_p, top_k) are rejected with a
        400, so none are sent;
      * thinking is on by default and max_tokens caps thinking plus the answer,
        so the budget is sized for both (ANTHROPIC_MAX_TOKENS);
      * safety classifiers can decline a request with stop_reason "refusal",
        so `fallbacks: "default"` re-runs it server-side, and stop_reason is
        checked before any content is read.

    Structured output goes through output_config.format rather than
    messages.parse(). parse() validates the JSON before returning, so a
    response truncated at max_tokens would raise a schema error before
    stop_reason could be checked -- and score as "insufficient evidence"
    instead of being reported as a truncation.

    Failure handling is deliberately asymmetric. API failures (rate limits,
    overload, network) are retried inside the SDK and otherwise propagate, as
    do refused and truncated outputs, so the caller records an error. Only a
    complete response that does not fit the schema becomes an
    insufficient-evidence answer.
    """

    FALLBACKS_BETA = "server-side-fallback-2026-07-01"

    def __init__(self) -> None:
        import anthropic  # local import by design: the mock path needs no SDK

        self._client = anthropic.Anthropic(
            api_key=settings.ANTHROPIC_API_KEY,
            max_retries=settings.ANTHROPIC_MAX_RETRIES,
        )
        self._transform_schema = anthropic.transform_schema
        # Same shape OllamaLLM/GroqLLM expose, so app.graph attaches per-call
        # token usage to the trace for every provider alike.
        self.last_usage: dict = {}

    def _create(self, prompt: str, schema=None) -> str:
        """One request; returns the answer text once it is known to be complete."""
        output_config: dict = {}
        if settings.ANTHROPIC_EFFORT:
            output_config["effort"] = settings.ANTHROPIC_EFFORT
        if schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": self._transform_schema(schema)}
        request: dict = {
            "model": settings.ANTHROPIC_MODEL,
            "max_tokens": settings.ANTHROPIC_MAX_TOKENS,
            "messages": [{"role": "user", "content": prompt}],
        }
        if output_config:
            request["output_config"] = output_config
        if settings.ANTHROPIC_FALLBACKS:
            request["fallbacks"] = settings.ANTHROPIC_FALLBACKS
            request["betas"] = [self.FALLBACKS_BETA]

        self.last_usage = {}
        response = self._client.beta.messages.create(**request)

        usage = response.usage
        self.last_usage = {
            # Cached input still counts toward input-token rate limits.
            "prompt_tokens": (usage.input_tokens or 0)
            + (getattr(usage, "cache_read_input_tokens", None) or 0)
            + (getattr(usage, "cache_creation_input_tokens", None) or 0),
            # Output tokens include thinking, which is billed as output.
            "completion_tokens": usage.output_tokens,
        }
        if response.stop_reason == "refusal":
            category = getattr(getattr(response, "stop_details", None), "category", None)
            raise ModelOutputError(f"model declined the request (refusal category: {category})")
        if response.stop_reason == "max_tokens":
            raise ModelOutputError(
                f"output truncated at max_tokens={settings.ANTHROPIC_MAX_TOKENS}; thinking "
                "counts toward it, so raise ANTHROPIC_MAX_TOKENS"
            )

        text: List[str] = []
        for block in response.content:
            if block.type == "fallback":
                # A switch point: text before it came from a model that declined.
                text = []
            elif block.type == "text":
                text.append(block.text)
        return "".join(text)


def _enum_name(value) -> Optional[str]:
    return None if value is None else getattr(value, "name", str(value))


class GeminiQuotaExhausted(RuntimeError):
    """No configured Gemini key can take a request: none is set, or every one
    has used its daily quota."""


# Shared by every GeminiLLM instance: the graph builds a new LLM for each
# question, and a key that ran out must stay skipped across questions.
_GEMINI_KEY_LOCK = threading.Lock()
_GEMINI_EXHAUSTED_UNTIL: dict = {}
_RETRY_DELAY_RE = re.compile(r"retryDelay['\"]?\s*:\s*['\"]?(\d+(?:\.\d+)?)s")


def gemini_keys() -> List[str]:
    """Configured keys in order: GEMINI_API_KEYS (comma-separated), then GEMINI_API_KEY."""
    keys = [key.strip() for key in settings.GEMINI_API_KEYS.split(",")]
    keys.append(settings.GEMINI_API_KEY.strip())
    return list(dict.fromkeys(key for key in keys if key))


def gemini_keys_exhausted() -> int:
    """How many configured keys are waiting out a daily quota."""
    now = time.monotonic()
    with _GEMINI_KEY_LOCK:
        return sum(1 for key in gemini_keys() if _GEMINI_EXHAUSTED_UNTIL.get(key, 0) > now)


def _retry_delay_s(message: str) -> float:
    match = _RETRY_DELAY_RE.search(message)
    return min(60.0, max(2.0, float(match.group(1)) + 1.0)) if match else 30.0


class GeminiLLM(_PromptedLLM):
    """Gemini through Google's official google-genai SDK, with JSON-schema
    structured output. Same prompts and failure handling as AnthropicLLM.

      * Structured output sends `response_mime_type="application/json"` with
        the pydantic model's JSON schema as `response_json_schema`.
      * Several keys can be configured (GEMINI_API_KEYS). A daily quota on one
        moves the request to the next; a per-minute limit waits the delay
        Google names. When every key has used its daily quota the error
        propagates and the question is recorded as an error. Transient
        server errors are retried inside the SDK.
      * A blocked prompt, or a candidate that stopped for any reason other than
        STOP, raises ModelOutputError: output cut off at max_output_tokens
        (which thinking also consumes) or withheld for safety is not an answer.
      * Thinking tokens are billed as output, so they count as completion tokens.
    """

    # 429 is handled in _generate, where a per-minute limit and a daily quota
    # can be told apart; the SDK would retry an exhausted key for minutes.
    RETRY_STATUS_CODES = (500, 502, 503, 504)

    def __init__(self) -> None:
        from google import genai  # local import by design: the mock path needs no SDK

        self._genai = genai
        self._clients: dict = {}
        self.last_usage: dict = {}

    def _client_for(self, key: str):
        if key not in self._clients:
            self._clients[key] = self._genai.Client(
                api_key=key,
                http_options={
                    "retry_options": {
                        "attempts": settings.GEMINI_MAX_ATTEMPTS,
                        "initial_delay": 2.0,
                        "max_delay": 60.0,
                        "http_status_codes": list(self.RETRY_STATUS_CODES),
                    }
                },
            )
        return self._clients[key]

    def _generate(self, **request):
        """generate_content on the first key with quota left.

        A daily quota marks the key exhausted for GEMINI_EXHAUSTED_KEY_RETRY_S
        and moves on to the next key. A per-minute limit waits the delay the
        error names and retries the same key, up to GEMINI_MAX_ATTEMPTS times.
        Keys are logged by position, never by value.
        """
        keys = gemini_keys()
        if not keys:
            raise GeminiQuotaExhausted("no Gemini API key is configured: set GEMINI_API_KEY or GEMINI_API_KEYS")
        waits = 0
        # A key that ran out during this request is not tried again in it,
        # whatever GEMINI_EXHAUSTED_KEY_RETRY_S is: no loop over spent keys.
        spent: set = set()
        while True:
            now = time.monotonic()
            with _GEMINI_KEY_LOCK:
                available = [
                    key for key in keys if key not in spent and _GEMINI_EXHAUSTED_UNTIL.get(key, 0) <= now
                ]
            if not available:
                raise GeminiQuotaExhausted(f"all {len(keys)} Gemini API key(s) have used their daily quota")
            key = available[0]
            try:
                return self._client_for(key).models.generate_content(**request)
            except Exception as exc:
                if getattr(exc, "code", None) != 429:
                    raise
                message = str(exc)
                position = keys.index(key) + 1
                if "PerDay" in message:
                    spent.add(key)
                    with _GEMINI_KEY_LOCK:
                        _GEMINI_EXHAUSTED_UNTIL[key] = time.monotonic() + settings.GEMINI_EXHAUSTED_KEY_RETRY_S
                    logger.warning("Gemini key %d of %d used its daily quota; trying the next key", position, len(keys))
                    continue
                waits += 1
                if waits >= settings.GEMINI_MAX_ATTEMPTS:
                    raise
                delay = _retry_delay_s(message)
                logger.info("Gemini key %d of %d hit a per-minute limit; waiting %.0f s", position, len(keys), delay)
                time.sleep(delay)

    def _create(self, prompt: str, schema=None) -> str:
        """One request; returns the answer text once it is known to be complete."""
        config: dict = {"max_output_tokens": settings.GEMINI_MAX_OUTPUT_TOKENS}
        if settings.GEMINI_TEMPERATURE is not None:
            config["temperature"] = settings.GEMINI_TEMPERATURE
        if schema is not None:
            config["response_mime_type"] = "application/json"
            config["response_json_schema"] = schema.model_json_schema()

        self.last_usage = {}
        response = self._generate(model=settings.GEMINI_MODEL, contents=prompt, config=config)

        usage = getattr(response, "usage_metadata", None)
        if usage is not None:
            self.last_usage = {
                "prompt_tokens": usage.prompt_token_count or 0,
                "completion_tokens": (usage.candidates_token_count or 0)
                + (usage.thoughts_token_count or 0),
            }
        feedback = getattr(response, "prompt_feedback", None)
        if feedback is not None and getattr(feedback, "block_reason", None):
            raise ModelOutputError(f"prompt blocked (block reason: {_enum_name(feedback.block_reason)})")
        candidates = response.candidates or []
        if not candidates:
            raise ModelOutputError("model returned no candidates")
        finish = _enum_name(candidates[0].finish_reason)
        if finish == "MAX_TOKENS":
            raise ModelOutputError(
                f"output truncated at max_output_tokens={settings.GEMINI_MAX_OUTPUT_TOKENS}; "
                "thinking counts toward it, so raise GEMINI_MAX_OUTPUT_TOKENS"
            )
        if finish not in ("STOP", None):
            raise ModelOutputError(f"model stopped without a complete answer (finish reason: {finish})")
        return response.text or ""


def get_llm():
    if settings.LLM_PROVIDER == "anthropic":
        return AnthropicLLM()
    if settings.LLM_PROVIDER == "gemini":
        return GeminiLLM()
    if settings.LLM_PROVIDER == "groq":
        return GroqLLM()
    if settings.LLM_PROVIDER == "ollama":
        return OllamaLLM()
    return MockLLM()
