"""Append-only persistent storage for human answer reviews."""

from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.schemas import ReviewRecord, ReviewRequest

_LOCK = threading.Lock()


def review_path() -> Path:
    configured = os.getenv("REVIEW_STORE_PATH")
    if configured:
        return Path(configured)
    document_dir = Path(os.getenv("DOCUMENT_STORE_DIR", "data/documents"))
    return document_dir / "reviews.jsonl"


def record_review(review: ReviewRequest) -> ReviewRecord:
    record = ReviewRecord(
        **review.model_dump(),
        review_id=str(uuid.uuid4()),
        created_at=datetime.now(timezone.utc),
    )
    destination = review_path()
    destination.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record.model_dump(mode="json"), ensure_ascii=False) + "\n"
    with _LOCK, destination.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())
    return record


def list_reviews(limit: int = 20) -> list[ReviewRecord]:
    destination = review_path()
    if not destination.exists():
        return []
    records: list[ReviewRecord] = []
    with _LOCK, destination.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(ReviewRecord.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(
                    f"invalid review record at line {line_number}"
                ) from exc
    return list(reversed(records[-limit:]))
