"""
Configuration for the LEDGER agent-service.

Everything is env-driven so the service can run standalone (MOCK mode, no
external dependencies) or wired into the real microservice mesh.
"""
import os

from dotenv import load_dotenv

# Auto-load a .env file sitting next to this project (if present) so the
# service works the same way regardless of shell (cmd.exe, PowerShell,
# bash) -- no manual `set`/`export` needed. Real environment variables
# already set take precedence over anything in .env.
load_dotenv()


def _bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


class Settings:
    # --- LLM ---
    # "mock"      -> zero-dependency heuristic/regex "LLM" (default, always works)
    # "ollama"    -> real local LLM (e.g. qwen2.5) served by `ollama serve`
    # "groq"      -> hosted LLM via Groq's fast, OpenAI-compatible API
    # "anthropic" -> real, resource-efficient Claude model via ANTHROPIC_API_KEY
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "mock")
    ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-opus-5")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    # The Anthropic SDK retries 429/529/5xx itself and honours retry-after, so
    # a per-minute rate limit becomes a wait rather than a failed question.
    ANTHROPIC_MAX_RETRIES: int = int(os.getenv("ANTHROPIC_MAX_RETRIES", "8"))
    # Claude Opus 5 thinks by default, and max_tokens caps thinking plus the
    # answer together: too small a budget truncates the answer itself.
    ANTHROPIC_MAX_TOKENS: int = int(os.getenv("ANTHROPIC_MAX_TOKENS", "16000"))
    # low | medium | high | xhigh | max. Empty sends none -- the API default,
    # high on Claude Opus 5. Lower effort is the main token and latency lever.
    ANTHROPIC_EFFORT: str = os.getenv("ANTHROPIC_EFFORT", "").strip()
    # "default" re-runs a declined request server-side on Anthropic's
    # recommended substitute model. Empty disables it.
    ANTHROPIC_FALLBACKS: str = os.getenv("ANTHROPIC_FALLBACKS", "default").strip()
    # USD per million tokens, used only to report approximate cost per query.
    # Defaults are Claude Opus 5 list prices; override for another model.
    ANTHROPIC_INPUT_USD_PER_MTOK: float = float(os.getenv("ANTHROPIC_INPUT_USD_PER_MTOK", "5.0"))
    ANTHROPIC_OUTPUT_USD_PER_MTOK: float = float(os.getenv("ANTHROPIC_OUTPUT_USD_PER_MTOK", "25.0"))

    # --- Groq (hosted LLM, OpenAI-compatible) ---
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    GROQ_BASE_URL: str = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    GROQ_TIMEOUT_S: float = float(os.getenv("GROQ_TIMEOUT_S", "30"))

    # --- Ollama (local LLM) ---
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
    OLLAMA_TIMEOUT_S: float = float(os.getenv("OLLAMA_TIMEOUT_S", "60"))
    # Every LLM call is validated + retried once; if the local model still
    # can't produce valid JSON, fall back to the offline MockLLM heuristic
    # for that one call instead of ever raising out of the graph.
    OLLAMA_FALLBACK_TO_MOCK: bool = _bool("OLLAMA_FALLBACK_TO_MOCK", True)

    # --- Retrieval service (retrieval-api) ---
    RETRIEVAL_API_URL: str = os.getenv("RETRIEVAL_API_URL", "http://localhost:8002")
    RETRIEVAL_TIMEOUT_S: float = float(os.getenv("RETRIEVAL_TIMEOUT_S", "10"))
    # If the retrieval-api is unreachable, fall back to a small in-memory
    # corpus so the agent graph is still demoable / unit-testable in isolation.
    RETRIEVAL_FALLBACK_TO_MOCK: bool = _bool("RETRIEVAL_FALLBACK_TO_MOCK", True)

    # --- Answer validator (answer-validator-api) ---
    # NOTE: per the architecture, the *orchestrator* is responsible for
    # sending the agent's answer to answer-validator-api. This service still
    # performs a local pydantic-schema self-check before returning, and can
    # optionally also call the validator directly if this URL is set
    # (useful when testing agent-service in isolation).
    ANSWER_VALIDATOR_URL: str = os.getenv("ANSWER_VALIDATOR_URL", "")
    SELF_VALIDATE: bool = _bool("SELF_VALIDATE", True)

    # --- Retrieval / retry tuning ---
    TOP_K_OVERRETRIEVE: int = int(os.getenv("TOP_K_OVERRETRIEVE", "30"))
    TOP_K_FINAL: int = int(os.getenv("TOP_K_FINAL", "5"))
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "2"))
    # Grading also requires the evidence to be tied to the company the question
    # names. On the stratified 10-question set it stopped A017 -- an unanswerable
    # Atlassian question answered from STMicroelectronics' table -- in every run,
    # but also refused A009 in two of three runs, where the excerpt holding the
    # fact never names the company (eval-service/EXPERIMENTS.md, finding 3).
    GRADE_REQUIRE_ENTITY_MATCH: bool = _bool("GRADE_REQUIRE_ENTITY_MATCH", True)
    MIN_EVIDENCE_SCORE: float = float(os.getenv("MIN_EVIDENCE_SCORE", "0.35"))

    # --- Service ---
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8003"))


settings = Settings()
