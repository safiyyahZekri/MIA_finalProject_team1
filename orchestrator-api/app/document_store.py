"""Persistent source-PDF storage for citation rendering.

The retrieval index owns searchable chunks.  The orchestrator keeps the raw
PDF used to create those chunks so the UI can render a cited page without
teaching retrieval-api how to serve files.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path


def store_directory() -> Path:
    return Path(os.getenv("DOCUMENT_STORE_DIR", "data/documents"))


def _storage_key(document_id: str) -> str:
    """Map arbitrary external IDs to a traversal-safe deterministic name."""
    return hashlib.sha256(document_id.encode("utf-8")).hexdigest()


def pdf_path(document_id: str) -> Path:
    return store_directory() / f"{_storage_key(document_id)}.pdf"


def save_pdf(document_id: str, content: bytes) -> Path:
    """Atomically save a successfully indexed PDF, replacing an older copy."""
    destination = pdf_path(document_id)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=destination.parent, suffix=".tmp", delete=False
        ) as temporary:
            temporary.write(content)
            temporary_path = Path(temporary.name)
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return destination
