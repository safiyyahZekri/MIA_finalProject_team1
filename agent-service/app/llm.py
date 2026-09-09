"""
LLM abstraction used by the LangGraph nodes.

Two interchangeable providers implement the same interface:
  - MockLLM       : pure heuristics/regex, zero external dependencies.
                    Default, so the service runs immediately with no API key.
  - AnthropicLLM  : a real, resource-efficient Claude model
                    (claude-haiku-4-5 by default) via structured output.

Only classification, query reformulation, evidence grading, and *proposing*
extraction candidates go through the LLM. The actual arithmetic always goes
through app.calculator.calculate() -- never trusted from the LLM directly.
"""
from __future__ import annotations

import json
import logging
import re
from typing import List, Literal, Optional

import httpx
from pydantic import BaseModel, Field, ValidationError

from app.config import settings

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


class ExtractionMultiSpan(BaseModel):
    shape: Literal["multi_span"] = "multi_span"
    values: List[str]


class ExtractionCalculated(BaseModel):
    shape: Literal["calculated"] = "calculated"
    formula: str  # numeric literals only, e.g. "(3875-3410)/3410*100"
    operand_count: int


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

    def as_extraction(self):
        if self.shape == "direct" and self.value is not None:
            return ExtractionDirect(value=str(self.value))
        if self.shape == "multi_span" and self.values:
            return ExtractionMultiSpan(values=[str(v) for v in self.values])
        if self.shape == "insufficient":
            return ExtractionInsufficient(reason=self.reason or "Insufficient evidence.")
        # A shape was declared but its payload field came back empty.
        return ExtractionInsufficient(
            reason=f"Model returned shape '{self.shape}' without a usable payload."
        )


# ------------------------------------------------------------- mock impl --
class MockLLM:
    """Deterministic, offline heuristic stand-in for a real LLM."""

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
class AnthropicLLM:
    """Real, resource-efficient Claude model via structured output.

    Requires `langchain-anthropic` and ANTHROPIC_API_KEY. Only imported/
    instantiated when LLM_PROVIDER=anthropic, so the mock path has zero
    extra runtime dependencies.
    """

    def __init__(self) -> None:
        from langchain_anthropic import ChatAnthropic  # local import by design

        self._base = ChatAnthropic(
            model=settings.ANTHROPIC_MODEL,
            api_key=settings.ANTHROPIC_API_KEY,
            temperature=0,
            max_tokens=512,
        )

    def classify(self, question: str) -> Classification:
        llm = self._base.with_structured_output(Classification)
        return llm.invoke(
            "Classify this financial-document question for retrieval routing.\n"
            "question_type must be exactly one of: numerical, table, text.\n"
            "- numerical: requires arithmetic (comparison, % change, sum, ratio, etc.)\n"
            "- table: answer likely lives in a financial table/line item, no math needed\n"
            "- text: narrative/prose answer\n"
            "search_query: a focused retrieval query derived from the question.\n\n"
            f"Question: {question}"
        )

    def reformulate(self, question: str, previous_query: str, attempt: int) -> str:
        llm = self._base
        resp = llm.invoke(
            "The previous retrieval query returned insufficient evidence.\n"
            f"Original question: {question}\n"
            f"Previous query: {previous_query}\n"
            "Propose ONE broader or differently-phrased retrieval query "
            "(no explanation, just the query text)."
        )
        return resp.content.strip()

    def grade(self, question: str, evidence: List[dict]) -> EvidenceGrade:
        llm = self._base.with_structured_output(EvidenceGrade)
        snippets = "\n---\n".join(
            f"[{e.get('document_id')} p{e.get('page')} score={e.get('score')}]\n{e.get('text','')}"
            for e in evidence
        ) or "(no evidence retrieved)"
        return llm.invoke(
            "Judge whether the evidence below is sufficient to confidently and "
            "faithfully answer the question. Be strict: if the exact fact/number "
            "needed isn't present, sufficient=false.\n\n"
            f"Question: {question}\n\nEvidence:\n{snippets}"
        )

    def extract(self, question: str, question_type: str, evidence: List[dict]):
        snippets = "\n---\n".join(
            f"[{e.get('document_id')} p{e.get('page')}]\n{e.get('text','')}" for e in evidence
        )
        if question_type == "numerical":
            llm = self._base.with_structured_output(ExtractionCalculated)
            try:
                return llm.invoke(
                    "Extract the numeric operands from the evidence needed to answer the "
                    "question and express the calculation as a pure-arithmetic formula "
                    "using ONLY the literal numbers found in the evidence "
                    "(e.g. '(3875-3410)/3410*100'). Do not compute the result yourself.\n\n"
                    f"Question: {question}\n\nEvidence:\n{snippets}"
                )
            except Exception:
                return ExtractionInsufficient(reason="Could not extract numeric operands.")

        # Let the model choose direct vs multi_span vs insufficient.
        llm = self._base.with_structured_output(ExtractionChoice)
        try:
            choice = llm.invoke(
                "Answer the question using ONLY the evidence below. Set shape to "
                "exactly one of:\n"
                "- direct: a single fact/value -- fill `value`\n"
                "- multi_span: two or more distinct values/items -- fill `values`\n"
                "- insufficient: the evidence doesn't contain the answer -- fill `reason`\n"
                "Leave the fields belonging to the other shapes unset.\n\n"
                f"Question: {question}\n\nEvidence:\n{snippets}"
            )
        except Exception as exc:  # noqa: BLE001 -- one bad call must not kill a run
            # Named distinctly so a systemic failure reads as 100 identical
            # "extraction call failed" reasons in the report rather than as
            # 100 plausible-looking "insufficient evidence" verdicts.
            logger.exception("Anthropic extract() failed: %s", exc)
            return ExtractionInsufficient(reason=f"Extraction call failed: {exc}")
        return choice.as_extraction()


def get_llm():
    if settings.LLM_PROVIDER == "anthropic":
        return AnthropicLLM()
    if settings.LLM_PROVIDER == "groq":
        return GroqLLM()
    if settings.LLM_PROVIDER == "ollama":
        return OllamaLLM()
    return MockLLM()
