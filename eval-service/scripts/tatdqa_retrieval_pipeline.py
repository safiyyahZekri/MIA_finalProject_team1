#!/usr/bin/env python3
"""Process raw TAT-DQA PDFs, batch-index them, then run retrieval ablations."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import httpx

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
    try:
        response = client.post(
            f"{retrieval_url.rstrip('/')}/documents/batch",
            json={"documents": batch},
        )
        response.raise_for_status()
        result = response.json()
        return int(result["documents_indexed"]), int(result["chunks_indexed"])
    except Exception as exc:
        errors.append(
            {
                "stage": "retrieval_batch",
                "documents": [item["document_id"] for item in batch],
                "error": f"{type(exc).__name__}: {exc}",
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
        existing = {
            item.get("source_doc_uid") or item.get("document_id")
            for item in existing_response.json()
        }

        for pdf_path in pdfs:
            pdf_bytes = pdf_path.read_bytes()
            source_doc_uid = source_doc_uid_for_pdf(pdf_path, identities)
            document_id = stable_id(pdf_bytes, source_doc_uid)
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
            try:
                response = client.post(
                    f"{args.doc_processor_url.rstrip('/')}/document_processing",
                    data=data,
                    files={"file": (pdf_path.name, pdf_bytes, "application/pdf")},
                )
                response.raise_for_status()
                processed = response.json()
                if not isinstance(processed, dict) or not isinstance(
                    processed.get("pages"), list
                ):
                    raise ValueError("processor response is missing pages")
            except Exception as exc:
                errors.append(
                    {
                        "stage": "document_processor",
                        "filename": pdf_path.name,
                        "error": f"{type(exc).__name__}: {exc}",
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
