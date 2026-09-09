#!/usr/bin/env python3
"""Map TAT-DQA uids that name byte-identical documents onto one identity.

The corpus ships the same PDF under several uids. Indexing is
content-addressed, so only one copy is stored -- under whichever uid was
ingested first. Gold evidence may name any of the others, and the
evaluator then scores a perfectly correct retrieval as a miss purely
because the hit carries the twin's name.

Emitting the equivalence up front lets the evaluator canonicalise both
sides of the comparison instead of duplicating documents in the index to
make the names line up.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "identity_aliases.json",
    )
    args = parser.parse_args()

    aliases = build_aliases(args.corpus_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(aliases, indent=2, sort_keys=True), encoding="utf-8")

    groups = len(set(aliases.values()))
    print(f"corpus root      : {args.corpus_root}")
    print(f"duplicate groups : {groups}")
    print(f"aliased uids     : {len(aliases)}")
    print(f"written to       : {args.output}")


if __name__ == "__main__":
    main()
