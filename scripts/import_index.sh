#!/usr/bin/env bash
# Load a prepared LEDGER index, and optionally the source PDFs used for
# evidence highlighting, into this Compose project's Docker volumes.
#
# Run from the repository root, before starting the stack (or it stops the
# two services that use these volumes first):
#   bash scripts/import_index.sh path/to/ledger-index-company.tgz [path/to/ledger-source-pdfs.tgz]
#
# Works in Git Bash on Windows and in a macOS/Linux terminal. Replaces the
# current contents of the volumes.
set -euo pipefail

usage="usage: bash scripts/import_index.sh ledger-index-company.tgz [ledger-source-pdfs.tgz]"
INDEX_ARCHIVE=${1:?$usage}
PDF_ARCHIVE=${2:-}

if [ ! -f docker-compose.yml ]; then
  echo "Run this from the repository root (where docker-compose.yml is)." >&2
  exit 1
fi
for archive in "$INDEX_ARCHIVE" $PDF_ARCHIVE; do
  if [ ! -f "$archive" ]; then
    echo "File not found: $archive" >&2
    exit 1
  fi
done

PROJECT=$(docker compose config | sed -n 's/^name: //p' | head -1)
if [ -z "$PROJECT" ]; then
  echo "Could not read the Compose project name; is Docker running?" >&2
  exit 1
fi

# Docker needs a host path it understands: a Windows path in Git Bash.
host_dir () {
  (cd "$(dirname "$1")" && (pwd -W 2>/dev/null || pwd))
}

load () {  # archive volume-suffix
  local archive=$1 volume="${PROJECT}_$2"
  docker volume create \
    --label "com.docker.compose.project=$PROJECT" \
    --label "com.docker.compose.volume=$2" \
    "$volume" > /dev/null
  echo "Loading $(basename "$archive") into volume $volume"
  MSYS_NO_PATHCONV=1 docker run --rm \
    -v "$volume:/target" -v "$(host_dir "$archive"):/in:ro" alpine \
    sh -c "find /target -mindepth 1 -delete && tar xzf '/in/$(basename "$archive")' -C /target"
}

running=$(docker compose ps -q retrieval-api orchestrator-api 2>/dev/null || true)
if [ -n "$running" ]; then
  echo "Stopping retrieval-api and orchestrator-api while their volumes are replaced"
  docker compose stop retrieval-api orchestrator-api > /dev/null
fi

load "$INDEX_ARCHIVE" retrieval-data
MSYS_NO_PATHCONV=1 docker run --rm -v "${PROJECT}_retrieval-data:/data:ro" alpine \
  sh -c 'test -f /data/chunks.jsonl && test -f /data/vectors.npy && echo "Index manifest:" && cat /data/manifest.json && echo' \
  || { echo "The archive does not contain a LEDGER index (chunks.jsonl, vectors.npy)." >&2; exit 1; }

if [ -n "$PDF_ARCHIVE" ]; then
  load "$PDF_ARCHIVE" source-documents
  count=$(MSYS_NO_PATHCONV=1 docker run --rm -v "${PROJECT}_source-documents:/pdfs:ro" alpine sh -c 'ls /pdfs/*.pdf 2>/dev/null | wc -l')
  echo "Source PDFs loaded: $count"
fi

echo
echo "Done. Start LEDGER with the best measured configuration:"
echo "  docker compose -f docker-compose.yml -f docker-compose.best.yml up -d"
