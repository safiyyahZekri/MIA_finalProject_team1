from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path(os.getenv("LEDGER_RETRIEVAL_DATA_DIR", "./data"))
    embedding_model: str = os.getenv("LEDGER_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    reranker_model: str = os.getenv(
        "LEDGER_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    device: str = os.getenv("LEDGER_MODEL_DEVICE", "cpu")
    chunk_max_chars: int = int(os.getenv("LEDGER_CHUNK_MAX_CHARS", "1800"))
    chunk_overlap_blocks: int = int(os.getenv("LEDGER_CHUNK_OVERLAP_BLOCKS", "1"))
    enable_reranker: bool = _as_bool(os.getenv("LEDGER_ENABLE_RERANKER", "true"))
