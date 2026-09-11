#!/usr/bin/env python3
"""Copy this person's assigned slice of TAT-DQA PDFs out of the full
extracted corpus, based on a manifest file (one relative path per line,
e.g. "train/0038aa...pdf").

Usage:
    python split_from_manifest.py --manifest personN_manifest.txt --source-root C:\\ledger_data\\docs --dest C:\\ledger_data\\my_slice
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="This person's manifest .txt file")
    parser.add_argument("--source-root", type=Path, required=True, help="Folder containing train/dev/test subfolders of extracted PDFs")
    parser.add_argument("--dest", type=Path, required=True, help="Destination folder for this person's slice")
    args = parser.parse_args()

    lines = [line.strip() for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]

    copied = 0
    missing = []
    for rel_path in lines:
        src = args.source_root / rel_path
        if not src.exists():
            missing.append(rel_path)
            continue
        dest_subdir = args.dest / Path(rel_path).parent
        dest_subdir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest_subdir / src.name)
        copied += 1

    print(f"Manifest entries: {len(lines)}")
    print(f"Copied: {copied}")
    print(f"Missing (not found under --source-root): {len(missing)}")
    if missing:
        print("First few missing files:")
        for m in missing[:10]:
            print(f"  {m}")


if __name__ == "__main__":
    main()
