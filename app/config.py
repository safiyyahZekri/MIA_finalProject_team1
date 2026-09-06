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
    # "anthropic" -> real, resource-efficient Claude model via ANTHROPIC_API_KEY
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "mock")
    ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

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
    MIN_EVIDENCE_SCORE: float = float(os.getenv("MIN_EVIDENCE_SCORE", "0.35"))

    # --- Service ---
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8003"))


settings = Settings()
