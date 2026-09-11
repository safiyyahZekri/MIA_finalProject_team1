"""Rebuild the retrieval index under another index format, with no OCR.

Every chunk in chunks.jsonl keeps what the chunker needs: a table chunk stores
its whole serialized table, its cells and the lines before it, and paragraph
chunks do not depend on the format. rebuild_chunks() re-cuts each document from
them exactly as build_chunks() cuts the original document at upload.

  verify     rebuild every document in the legacy format and require the
             result to equal the stored chunks, field for field
  companies  write the company each document's own headers name, as JSON
  build      write a complete index in a format to a new directory: verify,
             rebuild, reuse vectors wherever the embedded text is unchanged
             (the source index and any --reuse index), embed the rest, then
             reload the result the way retrieval-api does

The source index is only read. Run inside the retrieval-api image, e.g.
  python scripts/rebuild_index.py --data-dir /data build \
      --format contextual_text,company_metadata --output-dir /exp/company
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.chunking import ChunkingConfig, rebuild_chunks  # noqa: E402
from app.engine import RetrievalEngine  # noqa: E402
from app.index_format import LEGACY, IndexFormat, searchable_text  # noqa: E402
from app.models import DEFAULT_SECTION, Chunk  # noqa: E402

MAX_TOKENS = 512


def load_index(data_dir: Path) -> tuple[list[Chunk], dict]:
    chunks_path = data_dir / "chunks.jsonl"
    if not chunks_path.exists():
        raise SystemExit(f"{data_dir} has no chunks.jsonl")
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    with chunks_path.open(encoding="utf-8") as handle:
        chunks = [Chunk.model_validate_json(line) for line in handle if line.strip()]
    return chunks, manifest


def by_document(chunks: list[Chunk]) -> dict[str, list[Chunk]]:
    documents: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        documents.setdefault(chunk.document_id, []).append(chunk)
    return documents


def require_legacy_source(data_dir: Path, manifest: dict) -> None:
    if IndexFormat.from_manifest(manifest.get("index_format")) != LEGACY:
        raise SystemExit(f"{data_dir} is not a legacy-format index; rebuild from the original index")
    if (data_dir / "corrections.jsonl").exists():
        raise SystemExit(f"{data_dir} has human corrections, which a rebuild cannot reproduce")


def verify(chunks: list[Chunk], config: ChunkingConfig) -> list[str]:
    problems = []
    for document_id, stored in by_document(chunks).items():
        try:
            again = rebuild_chunks(stored, config, LEGACY)
        except ValueError as exc:
            problems.append(f"{document_id}: {exc}")
            continue
        if [chunk.model_dump() for chunk in again] != [chunk.model_dump() for chunk in stored]:
            problems.append(f"{document_id}: rebuilt chunks differ from the stored ones")
    return problems


def config_from(args: argparse.Namespace) -> ChunkingConfig:
    return ChunkingConfig(max_chars=args.max_chars, overlap_blocks=args.overlap_blocks)


def command_verify(args: argparse.Namespace) -> None:
    chunks, manifest = load_index(args.data_dir)
    require_legacy_source(args.data_dir, manifest)
    started = time.monotonic()
    problems = verify(chunks, config_from(args))
    documents = len(by_document(chunks))
    print(
        f"{documents - len(problems)}/{documents} documents ({len(chunks)} chunks) rebuild exactly "
        f"in the legacy format [{time.monotonic() - started:.0f}s]"
    )
    for problem in problems[:10]:
        print("  ", problem)
    if problems:
        raise SystemExit(1)


def command_companies(args: argparse.Namespace) -> None:
    chunks, manifest = load_index(args.data_dir)
    require_legacy_source(args.data_dir, manifest)
    config = config_from(args)
    index_format = IndexFormat(company_metadata=True)
    companies = {
        document_id: rebuild_chunks(stored, config, index_format)[0].metadata.get("company")
        for document_id, stored in by_document(chunks).items()
    }
    found = sum(company is not None for company in companies.values())
    print(f"{found}/{len(companies)} documents name their company in their own headers")
    args.output.write_text(json.dumps(companies, indent=2, ensure_ascii=False), encoding="utf-8")


def over_limit(tokenizer, texts: list[str]) -> int:
    count = 0
    for start in range(0, len(texts), 512):
        batch = tokenizer(texts[start : start + 512], add_special_tokens=True)["input_ids"]
        count += sum(len(ids) > MAX_TOKENS for ids in batch)
    return count


def command_build(args: argparse.Namespace) -> None:
    started = time.monotonic()
    target = IndexFormat.parse(args.format)
    if target == LEGACY:
        raise SystemExit("--format must switch something on")
    source_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir == source_dir:
        raise SystemExit("--output-dir must not be the source index")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"{output_dir} is not empty")

    chunks, manifest = load_index(source_dir)
    require_legacy_source(source_dir, manifest)
    if (manifest.get("embedding_model") or args.model) != args.model:
        raise SystemExit(f"the source index was embedded with {manifest['embedding_model']}, not {args.model}")
    config = config_from(args)
    problems = verify(chunks, config)
    if problems:
        for problem in problems[:10]:
            print("  ", problem)
        raise SystemExit(f"{len(problems)} documents do not rebuild exactly; nothing was written")
    print(f"verified: all {len(by_document(chunks))} documents rebuild exactly in the legacy format")

    rebuilt: list[Chunk] = []
    for stored in by_document(chunks).values():
        rebuilt.extend(rebuild_chunks(stored, config, target))
    texts = [searchable_text(chunk, target) for chunk in rebuilt]

    cache: dict[str, tuple[np.ndarray, int]] = {}
    for reuse_dir in [source_dir, *(path.resolve() for path in args.reuse)]:
        reuse_chunks, reuse_manifest = load_index(reuse_dir)
        if (reuse_manifest.get("embedding_model") or args.model) != args.model:
            raise SystemExit(f"{reuse_dir} was embedded with another model")
        reuse_format = IndexFormat.from_manifest(reuse_manifest.get("index_format"))
        reuse_vectors = np.load(reuse_dir / "vectors.npy", mmap_mode="r")
        if len(reuse_vectors) != len(reuse_chunks):
            raise SystemExit(f"{reuse_dir}: chunks and vectors are inconsistent")
        for index, chunk in enumerate(reuse_chunks):
            cache.setdefault(searchable_text(chunk, reuse_format), (reuse_vectors, index))
    dimensions = next(iter(cache.values()))[0].shape[1]
    vectors = np.empty((len(rebuilt), dimensions), dtype=np.float32)
    to_embed = []
    for index, text in enumerate(texts):
        reused = cache.get(text)
        if reused is None:
            to_embed.append(index)
        else:
            vectors[index] = reused[0][reused[1]]
    print(
        f"{len(rebuilt)} chunks in format {target.describe()}: "
        f"{len(rebuilt) - len(to_embed)} vectors reused, {len(to_embed)} to embed",
        flush=True,
    )

    from app.models_runtime import SentenceTransformerEmbedder

    embedder = SentenceTransformerEmbedder(args.model, device="cpu")
    embed_seconds = 0.0
    if to_embed:
        embed_started = time.monotonic()
        new_vectors = embedder.encode_documents([texts[index] for index in to_embed])
        embed_seconds = time.monotonic() - embed_started
        if new_vectors.shape != (len(to_embed), dimensions):
            raise SystemExit(f"the embedder returned {new_vectors.shape} vectors")
        vectors[to_embed] = new_vectors
        print(f"embedded {len(to_embed)} chunks in {embed_seconds:.0f}s", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    engine = RetrievalEngine(output_dir, embedder, None, config, index_format=target)
    engine._chunks = rebuilt
    engine._vectors = vectors
    engine._rebuild_indexes()
    engine._persist()

    # Reload as retrieval-api would, configured for legacy on purpose: the
    # manifest has to carry the format.
    reloaded = RetrievalEngine(output_dir, embedder, None, config)
    if reloaded.index_format != target or len(reloaded._chunks) != len(rebuilt):
        raise SystemExit("the written index does not reload as built")

    tokenizer = embedder._load().tokenizer
    documents = by_document(rebuilt)
    report = {
        "source": str(source_dir),
        "index_format": target.to_manifest(),
        "documents": len(documents),
        "chunks": len(rebuilt),
        "source_chunks": len(chunks),
        "vectors_reused": len(rebuilt) - len(to_embed),
        "vectors_embedded": len(to_embed),
        "embed_seconds": round(embed_seconds, 1),
        "documents_with_company": sum(
            isinstance(stored[0].metadata.get("company"), str) for stored in documents.values()
        ),
        "table_chunks_titled_from_first_row": sum(
            chunk.content_type == "table"
            and chunk.table_title != (chunk.section if chunk.section != DEFAULT_SECTION else None)
            for chunk in rebuilt
        ),
        "over_512_tokens_source": over_limit(tokenizer, [searchable_text(chunk, LEGACY) for chunk in chunks]),
        "over_512_tokens_built": over_limit(tokenizer, texts),
        "seconds": round(time.monotonic() - started, 1),
    }
    (output_dir / "build_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("/data"))
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--max-chars", type=int, default=1800)
    parser.add_argument("--overlap-blocks", type=int, default=1)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify")
    companies = commands.add_parser("companies")
    companies.add_argument("--output", type=Path, required=True)
    build = commands.add_parser("build")
    build.add_argument("--format", required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--reuse", type=Path, action="append", default=[])
    args = parser.parse_args()
    {"verify": command_verify, "companies": command_companies, "build": command_build}[args.command](args)


if __name__ == "__main__":
    main()
