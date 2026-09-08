#!/usr/bin/env python3
"""Process raw TAT-DQA PDFs, batch-index them, then run retrieval ablations."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import httpx

# Roughly 3 minutes of backoff (5+10+20+40+60+60s) -- enough to ride out a
# container restart and its model reload without losing the document.
OCR_MAX_ATTEMPTS = 7

SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.retrieval_benchmark import format_comparison, run_ablation  # noqa: E402


TATDQA_SOURCE_UID_PATTERN = re.compile(r"^[0-9a-fA-F]{32}$")


def load_json_records(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"{path} must contain a JSON array")
    return value


def identity_map(questions: list[dict], manifest: Path | None) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for question in questions:
        for evidence in question.get("gold_evidence", []) or []:
            filename = evidence.get("source_document")
            uid = evidence.get("source_doc_uid")
            if filename and uid:
                mapping[Path(filename).name] = str(uid)
    if manifest:
        raw: Any = json.loads(manifest.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            mapping.update({Path(key).name: str(value) for key, value in raw.items()})
        elif isinstance(raw, list):
            for item in raw:
                filename = item.get("filename") or item.get("source_document")
                uid = item.get("source_doc_uid")
                if filename and uid:
                    mapping[Path(filename).name] = str(uid)
        else:
            raise ValueError("identity manifest must be an object or array")
    return mapping


def stable_id(pdf_bytes: bytes, source_doc_uid: str | None) -> str:
    return source_doc_uid or f"sha256-{hashlib.sha256(pdf_bytes).hexdigest()}"


def source_doc_uid_for_pdf(pdf_path: Path, identities: dict[str, str]) -> str | None:
    """Resolve the external TAT-DQA identity without changing generic uploads.

    The official TAT-DQA archives name each PDF ``<source_doc_uid>.pdf``. A
    supplied/derived manifest remains authoritative for renamed files; the
    filename fallback is limited to the dataset's 32-character hexadecimal UID
    shape so arbitrary filenames still use the content-hash fallback.
    """
    mapped_uid = identities.get(pdf_path.name)
    if mapped_uid:
        return mapped_uid
    if TATDQA_SOURCE_UID_PATTERN.fullmatch(pdf_path.stem):
        return pdf_path.stem.lower()
    return None


def flush_batch(
    client: httpx.Client,
    retrieval_url: str,
    batch: list[dict],
    errors: list[dict],
) -> tuple[int, int]:
    if not batch:
        return 0, 0
    # A failed flush discards up to 32 documents that have already been OCR'd
    # -- the most expensive work in the pipeline -- so a transient blip is
    # worth retrying. A 4xx is not transient (the batch itself is malformed,
    # e.g. duplicate ids) and fails immediately rather than retrying six times.
    last_error: Exception | None = None
    for attempt in range(OCR_MAX_ATTEMPTS):
        try:
            response = client.post(
                f"{retrieval_url.rstrip('/')}/documents/batch",
                json={"documents": batch},
            )
            response.raise_for_status()
            result = response.json()
            return int(result["documents_indexed"]), int(result["chunks_indexed"])
        except httpx.HTTPStatusError as exc:
            last_error = exc
            if exc.response is not None and 400 <= exc.response.status_code < 500:
                break
            if attempt < OCR_MAX_ATTEMPTS - 1:
                time.sleep(min(5 * (2**attempt), 60))
        except Exception as exc:
            last_error = exc
            if attempt < OCR_MAX_ATTEMPTS - 1:
                time.sleep(min(5 * (2**attempt), 60))
    errors.append(
        {
            "stage": "retrieval_batch",
            "documents": [item["document_id"] for item in batch],
            "error": f"{type(last_error).__name__}: {last_error}",
        }
    )
    return 0, 0


def run_pipeline(args: argparse.Namespace) -> dict:
    questions = load_json_records(args.questions)
    identities = identity_map(questions, args.identity_manifest)
    pdfs = sorted(args.pdf_dir.rglob("*.pdf"))
    if args.limit:
        pdfs = pdfs[: args.limit]

    errors: list[dict] = []
    indexed_documents = 0
    indexed_chunks = 0
    processed_documents = 0
    skipped_documents = 0
    duplicate_documents = 0
    seen_this_run: set[str] = set()
    pending: list[dict] = []
    with httpx.Client(timeout=args.timeout_seconds) as client:
        existing_response = client.get(f"{args.retrieval_url.rstrip('/')}/documents")
        existing_response.raise_for_status()
        # Match on document_id alone. Preferring source_doc_uid here would
        # compare a uid against the content-hash id computed below and never
        # agree, so every already-indexed document would be re-processed once
        # uids started being populated.
        existing = {item.get("document_id") for item in existing_response.json()}

        for pdf_path in pdfs:
            pdf_bytes = pdf_path.read_bytes()
            source_doc_uid = source_doc_uid_for_pdf(pdf_path, identities)
            # Keep the id content-addressed even when a uid is available.
            # Passing the uid to stable_id() would make it the document_id and
            # silently repoint the corpus: the 1,300+ documents already indexed
            # by content hash would all look unindexed and be re-OCR'd, and
            # byte-identical files under different names would stop colliding,
            # defeating the duplicate detection below. The uid travels
            # alongside as metadata instead.
            document_id = stable_id(pdf_bytes, None)
            if document_id in seen_this_run:
                # Byte-identical content under a different filename (the
                # dataset has real duplicate PDFs across/within splits).
                # A batch containing two documents with the same id is
                # rejected outright by retrieval-api, taking the whole
                # batch down with it -- so skip it here instead, same as
                # an already-indexed document, since content-addressed
                # ids mean indexing it once already covers this file too.
                duplicate_documents += 1
                skipped_documents += 1
                continue
            if not args.reindex and document_id in existing:
                skipped_documents += 1
                continue
            data = {"document_id": document_id}
            if source_doc_uid:
                data["source_doc_uid"] = source_doc_uid
            # Retry instead of giving up on the first failure.
            #
            # The OCR service gets restarted during long runs -- it is memory
            # hungry enough to be OOM-killed, and a watchdog also recycles it
            # preemptively. Without retries a single ~60s restart is
            # catastrophic rather than survivable: every remaining document
            # fails instantly with "Server disconnected" / WinError 10053. One
            # restart cost 166 documents in a single run, and the script still
            # exited 0, so the loss looked like a successful run.
            #
            # Backing off across a few minutes covers a restart plus its model
            # reload, and the loop resumes the moment the service answers.
            processed = None
            last_error: Exception | None = None
            for attempt in range(OCR_MAX_ATTEMPTS):
                try:
                    response = client.post(
                        f"{args.doc_processor_url.rstrip('/')}/document_processing",
                        data=data,
                        files={"file": (pdf_path.name, pdf_bytes, "application/pdf")},
                    )
                    response.raise_for_status()
                    candidate = response.json()
                    if not isinstance(candidate, dict) or not isinstance(
                        candidate.get("pages"), list
                    ):
                        raise ValueError("processor response is missing pages")
                    processed = candidate
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt < OCR_MAX_ATTEMPTS - 1:
                        time.sleep(min(5 * (2**attempt), 60))
            if processed is None:
                errors.append(
                    {
                        "stage": "document_processor",
                        "filename": pdf_path.name,
                        "attempts": OCR_MAX_ATTEMPTS,
                        "error": f"{type(last_error).__name__}: {last_error}",
                    }
                )
                continue
            processed_documents += 1
            seen_this_run.add(document_id)
            pending.append(
                {
                    "document": processed,
                    "document_id": document_id,
                    "source_doc_uid": source_doc_uid,
                    "source_filename": pdf_path.name,
                    "metadata": {
                        "source_doc_uid": source_doc_uid,
                        "original_filename": pdf_path.name,
                        "source_path": str(pdf_path.relative_to(args.pdf_dir)),
                    },
                }
            )
            if len(pending) >= args.batch_size:
                documents, chunks = flush_batch(
                    client, args.retrieval_url, pending, errors
                )
                indexed_documents += documents
                indexed_chunks += chunks
                pending = []

        documents, chunks = flush_batch(client, args.retrieval_url, pending, errors)
        indexed_documents += documents
        indexed_chunks += chunks

    comparison = run_ablation(
        args.retrieval_url,
        questions,
        candidate_k=args.candidate_k,
        top_k=10,
        output_dir=args.output_dir,
    )
    report = {
        "pdfs_discovered": len(pdfs),
        "processed_documents": processed_documents,
        "indexed_documents": indexed_documents,
        "indexed_chunks": indexed_chunks,
        "skipped_documents": skipped_documents,
        "duplicate_documents": duplicate_documents,
        "errors": errors,
        "ablation_run_id": comparison["run_id"],
        "ablation_comparison_path": comparison.get("comparison_path"),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pipeline_path = args.output_dir / f"tatdqa-pipeline-{comparison['run_id']}.json"
    pipeline_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print()
    print(format_comparison(comparison))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-dir", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--identity-manifest", type=Path)
    parser.add_argument("--doc-processor-url", default="http://localhost:8001")
    parser.add_argument("--retrieval-url", default="http://localhost:8002")
    parser.add_argument("--output-dir", type=Path, default=SERVICE_ROOT / "results")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--reindex", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())
