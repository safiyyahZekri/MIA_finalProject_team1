#!/usr/bin/env python3
"""Build the identity map the evaluators canonicalise through.

Two kinds of entry, from two sources:

  uid -> uid         The corpus ships the same PDF under several uids.
                     Indexing is content-addressed, so only one copy is
                     stored -- under whichever uid was ingested first. Gold
                     evidence may name any of the others. Built from the PDFs
                     with --corpus-root.

  sha256-... -> uid  Documents ingested without a uid are keyed by content
                     hash. Agent citations carry only `document_id`, so those
                     citations can never match a gold uid. Built from the live
                     index with --retrieval-url.

Without either, a correct retrieval scores as a miss purely because the two
sides name the same document differently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

INDEX_ID_PREFIX = "sha256-"


def build_aliases(corpus_root: Path) -> dict[str, str]:
    by_content: dict[str, list[str]] = defaultdict(list)
    for pdf in sorted(corpus_root.rglob("*.pdf")):
        # Jupyter's checkpoint copies are not corpus documents.
        if ".ipynb_checkpoints" in pdf.parts:
            continue
        digest = hashlib.sha256(pdf.read_bytes()).hexdigest()
        by_content[digest].append(pdf.stem)

    aliases: dict[str, str] = {}
    for uids in by_content.values():
        if len(uids) < 2:
            continue
        # Sorted first uid is the representative purely so the mapping is
        # deterministic and reproducible from the corpus alone.
        canonical = sorted(uids)[0]
        for uid in uids:
            if uid != canonical:
                aliases[uid] = canonical
    return aliases


def index_aliases(documents: list[dict], duplicates: dict[str, str]) -> dict[str, str]:
    """Map index document ids that aren't a uid onto the uid they hold.

    The uid comes from `source_doc_uid` when the index has one, otherwise the
    filename stem, which in this corpus is the TAT-DQA uid. It then goes
    through `duplicates`, so a hash-keyed copy of a byte-identical twin
    resolves to the same identity gold is canonicalised to.
    """
    aliases: dict[str, str] = {}
    for doc in documents:
        document_id = doc.get("document_id")
        filename = doc.get("source_filename") or doc.get("filename")
        if not document_id or not (doc.get("source_doc_uid") or filename):
            continue
        uid = doc.get("source_doc_uid") or Path(filename).stem
        canonical = duplicates.get(uid, uid)
        if document_id != canonical:
            aliases[document_id] = canonical
    return aliases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, help="rebuild uid aliases from the PDFs")
    parser.add_argument("--retrieval-url", help="rebuild sha256 aliases from GET /documents")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "identity_aliases.json",
    )
    args = parser.parse_args()
    if not args.corpus_root and not args.retrieval_url:
        parser.error("pass --corpus-root, --retrieval-url, or both")

    existing: dict[str, str] = {}
    if args.output.exists():
        existing = json.loads(args.output.read_text(encoding="utf-8"))

    # Each source rebuilds only its own kind of entry; the other kind is kept
    # from the existing file, since not every machine has the full corpus or
    # a running index.
    if args.corpus_root:
        duplicates = build_aliases(args.corpus_root)
    else:
        duplicates = {k: v for k, v in existing.items() if not k.startswith(INDEX_ID_PREFIX)}

    if args.retrieval_url:
        import httpx

        response = httpx.get(f"{args.retrieval_url.rstrip('/')}/documents", timeout=120)
        response.raise_for_status()
        indexed = index_aliases(response.json(), duplicates)
    else:
        indexed = {k: duplicates.get(v, v) for k, v in existing.items() if k.startswith(INDEX_ID_PREFIX)}

    aliases = {**duplicates, **indexed}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(aliases, indent=2, sort_keys=True), encoding="utf-8")

    print(f"duplicate groups     : {len(set(duplicates.values()))}")
    print(f"aliased uids         : {len(duplicates)}")
    print(f"index document ids   : {len(indexed)}")
    print(f"written to           : {args.output}")


if __name__ == "__main__":
    main()
