"""Re-split table chunks under another character budget, straight from the index.

The chunker budgets table chunks in characters (LEDGER_CHUNK_MAX_CHARS, 1,800),
but the embedding model reads at most 512 tokens, and table rows ("Row 3:
Inventories, net | 3,875 | 3,410") tokenize far more densely than prose. A
table chunk over 512 tokens loses its last rows before it is embedded, and
sentence-transformers truncates without a warning.

Every table chunk keeps its whole table (parent_text) and cells, so table
chunks can be rebuilt under a different budget with no OCR and no document
processing: the chunker's own _table_parts() splits the same rows. Paragraph
chunks are left exactly as they are.

  measure  confirm that rebuilding at today's budget reproduces every stored
           table chunk exactly, then report how many table chunks exceed the
           model's token limit at each budget
  build    write a complete index for one budget to a new directory: paragraph
           vectors reused, table chunks re-split and re-embedded
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.chunking import _chunk_id, _infer_header_rows, _table_parts  # noqa: E402
from app.engine import RetrievalEngine  # noqa: E402
from app.models import Chunk  # noqa: E402

MAX_TOKENS = 512


def load_chunks(data_dir: Path) -> list[Chunk]:
    with (data_dir / "chunks.jsonl").open(encoding="utf-8") as handle:
        return [Chunk.model_validate_json(line) for line in handle if line.strip()]


def table_groups(chunks: list[Chunk]) -> dict[tuple, list[int]]:
    """Indexes of the chunks cut from each table block, in index order."""
    groups: dict[tuple, list[int]] = {}
    for index, chunk in enumerate(chunks):
        if chunk.content_type == "table":
            key = (chunk.document_id, chunk.page, tuple(chunk.source_block_ids), chunk.parent_text)
            groups.setdefault(key, []).append(index)
    return groups


def split_table(first: Chunk, max_chars: int) -> list[Chunk]:
    """Split one table the way build_chunks() does, from what the index stores."""
    prefix = []
    if first.table_title:
        prefix.append(f"Table: {first.table_title}")
    if first.table_context:
        prefix.append(f"Context: {first.table_context}")
    head = "\n".join(prefix)
    if not first.parent_text.startswith(head):
        raise ValueError(f"chunk {first.chunk_id}: parent_text does not start with its prefix")
    lines = (first.parent_text[len(head) + 1 :] if head else first.parent_text).split("\n")
    if lines and lines[0].startswith("Units: "):
        prefix.append(lines.pop(0))
    rows = lines
    header_rows = (
        min(_infer_header_rows(first.table_cells, len(rows)), len(rows)) if first.table_cells else 1
    )
    available = max(100, max_chars - len("\n".join(prefix)))
    parts = _table_parts(rows, header_rows, available)
    block_id = first.source_block_ids[0]
    rebuilt = []
    for index, part in enumerate(parts):
        text = "\n".join([*prefix, part])
        identity = [block_id] if len(parts) == 1 else [f"{block_id}:part:{index + 1}"]
        chunk_id = _chunk_id(first.document_id, first.page, first.section, "table", identity, text)
        rebuilt.append(first.model_copy(update={"chunk_id": chunk_id, "text": text}))
    return rebuilt


def rebuild(chunks: list[Chunk], max_chars: int) -> tuple[list[Chunk], list[int | None]]:
    """The whole index under a table budget; for each chunk, the index of the
    unchanged original it came from (paragraphs) or None (re-split tables)."""
    groups = table_groups(chunks)
    first_of_group = {indexes[0]: key for key, indexes in groups.items()}
    in_group = {index for indexes in groups.values() for index in indexes}
    result: list[Chunk] = []
    origin: list[int | None] = []
    for index, chunk in enumerate(chunks):
        if index in first_of_group:
            for part in split_table(chunk, max_chars):
                result.append(part)
                origin.append(None)
        elif index not in in_group:
            result.append(chunk)
            origin.append(index)
    return result, origin


def token_counts(tokenizer, texts: list[str]) -> list[int]:
    counts = []
    for start in range(0, len(texts), 512):
        batch = tokenizer(texts[start : start + 512], add_special_tokens=True)["input_ids"]
        counts.extend(len(ids) for ids in batch)
    return counts


def measure(args) -> None:
    from transformers import AutoTokenizer

    chunks = load_chunks(args.data_dir)
    groups = table_groups(chunks)
    reproduced = 0
    for indexes in groups.values():
        stored = [(chunks[i].chunk_id, chunks[i].text) for i in indexes]
        again = [(c.chunk_id, c.text) for c in split_table(chunks[indexes[0]], args.current_max_chars)]
        reproduced += stored == again
    print(f"index: {len(chunks)} chunks, {len(groups)} tables")
    print(f"rebuild at {args.current_max_chars} chars reproduces {reproduced}/{len(groups)} tables exactly")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    report = {"tables": len(groups), "reproduced": reproduced, "budgets": {}}
    for max_chars in args.budgets:
        rebuilt, _ = rebuild(chunks, max_chars)
        tables = [c for c in rebuilt if c.content_type == "table"]
        counts = token_counts(tokenizer, [RetrievalEngine._searchable_text(c) for c in tables])
        over = [n for n in counts if n > MAX_TOKENS]
        row = {
            "table_chunks": len(tables),
            "over_limit": len(over),
            "over_limit_share": round(len(over) / len(tables), 4),
            "tokens_cut": sum(n - MAX_TOKENS for n in over),
            "median_tokens": sorted(counts)[len(counts) // 2],
            "max_tokens": max(counts),
        }
        report["budgets"][max_chars] = row
        print(
            f"tables at {max_chars:>4} chars: {row['table_chunks']:>5} chunks, "
            f"{row['over_limit']:>4} over {MAX_TOKENS} tokens ({row['over_limit_share']:.1%}), "
            f"{row['tokens_cut']:>6} tokens cut, median {row['median_tokens']}, max {row['max_tokens']}"
        )
    if args.output:
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


def build(args) -> None:
    import numpy as np

    from app.models_runtime import SentenceTransformerEmbedder

    chunks = load_chunks(args.data_dir)
    vectors = np.load(args.data_dir / "vectors.npy")
    rebuilt, origin = rebuild(chunks, args.table_max_chars)
    new_indexes = [i for i, source in enumerate(origin) if source is None]
    print(f"{len(rebuilt)} chunks; re-embedding {len(new_indexes)} table chunks")
    embedder = SentenceTransformerEmbedder(args.model, device="cpu")
    texts = [RetrievalEngine._searchable_text(rebuilt[i]) for i in new_indexes]
    new_vectors = embedder.encode_documents(texts)
    combined = np.empty((len(rebuilt), vectors.shape[1]), dtype=np.float32)
    for i, source in enumerate(origin):
        if source is not None:
            combined[i] = vectors[source]
    combined[new_indexes] = new_vectors

    engine = RetrievalEngine.__new__(RetrievalEngine)
    engine.data_dir = args.output_dir
    engine.embedder = embedder
    engine._chunks = rebuilt
    engine._vectors = combined
    engine._rebuild_indexes()
    engine._persist()
    print(f"wrote {args.output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("/data"))
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    commands = parser.add_subparsers(dest="command", required=True)
    measure_parser = commands.add_parser("measure")
    measure_parser.add_argument("--current-max-chars", type=int, default=1800)
    measure_parser.add_argument("--budgets", type=int, nargs="+", default=[1800, 1500, 1200, 1000])
    measure_parser.add_argument("--output", type=Path)
    build_parser = commands.add_parser("build")
    build_parser.add_argument("--table-max-chars", type=int, required=True)
    build_parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    measure(args) if args.command == "measure" else build(args)


if __name__ == "__main__":
    main()
