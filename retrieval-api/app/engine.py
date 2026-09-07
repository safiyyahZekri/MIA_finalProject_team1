from __future__ import annotations

import os
import re
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from .chunking import ChunkingConfig, build_chunks
from .models import (
    Chunk,
    CorpusStats,
    DocumentSummary,
    IndexDocumentRequest,
    IndexResponse,
    RetrievalScores,
    SearchFilters,
    SearchHit,
    SearchMode,
    SearchRequest,
    SearchResponse,
)
from .models_runtime import Embedder, Reranker

TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?|\(?-?\$?\d[\d,]*(?:\.\d+)?%?\)?")


def tokenize(text: str) -> list[str]:
    tokens = TOKEN_PATTERN.findall(text.lower())
    expanded: list[str] = []
    for token in tokens:
        expanded.append(token)
        normalized = token.replace(",", "").replace("$", "").strip("()")
        if normalized != token:
            expanded.append(normalized)
    return expanded


class RetrievalEngine:
    def __init__(
        self,
        data_dir: Path,
        embedder: Embedder,
        reranker: Reranker | None,
        chunking: ChunkingConfig | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.embedder = embedder
        self.reranker = reranker
        self.chunking = chunking or ChunkingConfig()
        self._lock = threading.RLock()
        self._chunks: list[Chunk] = []
        self._vectors = np.empty((0, 0), dtype=np.float32)
        self._faiss_index: faiss.Index | None = None
        self._bm25: BM25Okapi | None = None
        self._load()

    @property
    def chunks_path(self) -> Path:
        return self.data_dir / "chunks.jsonl"

    @property
    def vectors_path(self) -> Path:
        return self.data_dir / "vectors.npy"

    @property
    def faiss_path(self) -> Path:
        return self.data_dir / "chunks.faiss"

    def _load(self) -> None:
        with self._lock:
            if not self.chunks_path.exists():
                return
            with self.chunks_path.open("r", encoding="utf-8") as handle:
                self._chunks = [
                    Chunk.model_validate_json(line) for line in handle if line.strip()
                ]
            if self.vectors_path.exists():
                self._vectors = np.load(self.vectors_path).astype(
                    np.float32, copy=False
                )
            if len(self._chunks) != len(self._vectors):
                raise RuntimeError("persisted chunks and vectors are inconsistent")
            self._rebuild_indexes()

    def _rebuild_indexes(self) -> None:
        if not self._chunks:
            self._faiss_index = None
            self._bm25 = None
            return
        dimensions = self._vectors.shape[1]
        index = faiss.IndexFlatIP(dimensions)
        index.add(np.ascontiguousarray(self._vectors, dtype=np.float32))
        self._faiss_index = index
        tokenized = [tokenize(self._searchable_text(chunk)) for chunk in self._chunks]
        self._bm25 = BM25Okapi(tokenized)

    @staticmethod
    def _searchable_text(chunk: Chunk) -> str:
        metadata = " ".join(
            str(value) for value in chunk.metadata.values() if value is not None
        )
        return f"{chunk.source_filename} {metadata} {chunk.section} {chunk.text}"

    def _persist(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        chunks_tmp = self.chunks_path.with_suffix(".jsonl.tmp")
        vectors_tmp = self.vectors_path.with_suffix(".npy.tmp")
        faiss_tmp = self.faiss_path.with_suffix(".faiss.tmp")

        with chunks_tmp.open("w", encoding="utf-8") as handle:
            for chunk in self._chunks:
                handle.write(chunk.model_dump_json() + "\n")
        with vectors_tmp.open("wb") as handle:
            np.save(handle, self._vectors)
        if self._faiss_index is not None:
            faiss.write_index(self._faiss_index, str(faiss_tmp))

        os.replace(chunks_tmp, self.chunks_path)
        os.replace(vectors_tmp, self.vectors_path)
        if self._faiss_index is not None:
            os.replace(faiss_tmp, self.faiss_path)
        elif self.faiss_path.exists():
            self.faiss_path.unlink()

    def index_document(self, request: IndexDocumentRequest) -> IndexResponse:
        new_chunks = build_chunks(request, self.chunking)
        if not new_chunks:
            raise ValueError("document produced no searchable chunks")
        searchable = [self._searchable_text(chunk) for chunk in new_chunks]
        new_vectors = self.embedder.encode_documents(searchable)
        if new_vectors.ndim != 2 or len(new_vectors) != len(new_chunks):
            raise RuntimeError("embedder returned an invalid vector matrix")

        with self._lock:
            keep = [
                index
                for index, chunk in enumerate(self._chunks)
                if chunk.document_id != request.document.document_id
            ]
            replaced = len(self._chunks) - len(keep)
            kept_chunks = [self._chunks[index] for index in keep]
            if keep:
                kept_vectors = self._vectors[keep]
                if kept_vectors.shape[1] != new_vectors.shape[1]:
                    raise RuntimeError(
                        "embedding dimensions changed; clear or migrate the index"
                    )
                self._vectors = np.vstack([kept_vectors, new_vectors]).astype(
                    np.float32
                )
            else:
                self._vectors = np.asarray(new_vectors, dtype=np.float32)
            self._chunks = [*kept_chunks, *new_chunks]
            self._rebuild_indexes()
            self._persist()

        return IndexResponse(
            document_id=request.document.document_id,
            source_filename=request.source_filename or request.document.document_id,
            chunks_indexed=len(new_chunks),
            chunks_replaced=replaced,
        )

    def delete_document(self, document_id: str) -> int:
        with self._lock:
            keep = [
                index
                for index, chunk in enumerate(self._chunks)
                if chunk.document_id != document_id
            ]
            removed = len(self._chunks) - len(keep)
            if not removed:
                return 0
            self._chunks = [self._chunks[index] for index in keep]
            self._vectors = (
                self._vectors[keep] if keep else np.empty((0, 0), dtype=np.float32)
            )
            self._rebuild_indexes()
            self._persist()
            return removed

    @staticmethod
    def _matches(chunk: Chunk, filters: SearchFilters) -> bool:
        if filters.document_id is not None and chunk.document_id != filters.document_id:
            return False
        if filters.source_filename is not None:
            if filters.source_filename.lower() not in chunk.source_filename.lower():
                return False
        if filters.page is not None and chunk.page != filters.page:
            return False
        if filters.section is not None:
            if filters.section.lower() not in chunk.section.lower():
                return False
        if (
            filters.content_type is not None
            and chunk.content_type != filters.content_type
        ):
            return False
        return all(
            chunk.metadata.get(key) == value for key, value in filters.metadata.items()
        )

    @staticmethod
    def _rank_map(indices: Iterable[int]) -> dict[int, int]:
        return {index: rank for rank, index in enumerate(indices, start=1)}

    def search(self, request: SearchRequest) -> SearchResponse:
        started = time.perf_counter()
        with self._lock:
            eligible = [
                index
                for index, chunk in enumerate(self._chunks)
                if self._matches(chunk, request.filters)
            ]
            if not eligible:
                return SearchResponse(
                    query=request.query,
                    mode=request.mode,
                    candidate_count=0,
                    reranked=False,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    hits=[],
                )

            dense_scores: dict[int, float] = {}
            lexical_scores: dict[int, float] = {}
            dense_order: list[int] = []
            lexical_order: list[int] = []

            if request.mode in {SearchMode.hybrid, SearchMode.dense}:
                query_vector = self.embedder.encode_query(request.query)
                if (
                    self._faiss_index is None
                    or query_vector.shape[0] != self._vectors.shape[1]
                ):
                    raise RuntimeError("query embedding is incompatible with the index")
                scores, indices = self._faiss_index.search(
                    np.asarray([query_vector], dtype=np.float32), len(self._chunks)
                )
                eligible_set = set(eligible)
                for index, score in zip(indices[0], scores[0]):
                    if index >= 0 and int(index) in eligible_set:
                        dense_scores[int(index)] = float(score)
                        dense_order.append(int(index))

            if request.mode in {SearchMode.hybrid, SearchMode.lexical}:
                if self._bm25 is None:
                    raw_scores = np.zeros(len(self._chunks), dtype=np.float32)
                else:
                    raw_scores = self._bm25.get_scores(tokenize(request.query))
                lexical_order = sorted(
                    eligible, key=lambda index: (-float(raw_scores[index]), index)
                )
                lexical_scores = {index: float(raw_scores[index]) for index in eligible}

            dense_ranks = self._rank_map(dense_order)
            lexical_ranks = self._rank_map(lexical_order)
            fusion: dict[int, float] = defaultdict(float)
            for index in eligible:
                if index in dense_ranks:
                    fusion[index] += 0.55 / (60 + dense_ranks[index])
                if index in lexical_ranks:
                    fusion[index] += 0.45 / (60 + lexical_ranks[index])

            candidates = sorted(eligible, key=lambda index: (-fusion[index], index))[
                : request.candidate_k
            ]
            reranker_scores: dict[int, float] = {}
            did_rerank = bool(
                request.rerank and self.reranker is not None and candidates
            )
            if did_rerank:
                passages = [
                    self._searchable_text(self._chunks[index]) for index in candidates
                ]
                scores = self.reranker.score(request.query, passages)
                if len(scores) != len(candidates):
                    raise RuntimeError("reranker returned an invalid score count")
                reranker_scores = dict(zip(candidates, scores))
                candidates.sort(
                    key=lambda index: (-reranker_scores[index], -fusion[index], index)
                )

            hits: list[SearchHit] = []
            for rank, index in enumerate(candidates[: request.top_k], start=1):
                chunk = self._chunks[index]
                hits.append(
                    SearchHit(
                        rank=rank,
                        chunk_id=chunk.chunk_id,
                        document_id=chunk.document_id,
                        source_filename=chunk.source_filename,
                        page=chunk.page,
                        section=chunk.section,
                        content_type=chunk.content_type,
                        text=chunk.text,
                        parent_text=chunk.parent_text
                        if request.include_parent_context
                        else None,
                        bbox=chunk.bbox,
                        source_block_ids=chunk.source_block_ids,
                        metadata=chunk.metadata,
                        scores=RetrievalScores(
                            dense=dense_scores.get(index),
                            lexical=lexical_scores.get(index),
                            fusion=fusion[index],
                            reranker=reranker_scores.get(index),
                        ),
                    )
                )

        return SearchResponse(
            query=request.query,
            mode=request.mode,
            candidate_count=len(candidates),
            reranked=did_rerank,
            latency_ms=(time.perf_counter() - started) * 1000,
            hits=hits,
        )

    def documents(self) -> list[DocumentSummary]:
        grouped: dict[str, list[Chunk]] = defaultdict(list)
        with self._lock:
            for chunk in self._chunks:
                grouped[chunk.document_id].append(chunk)
        return [
            DocumentSummary(
                document_id=document_id,
                source_filename=chunks[0].source_filename,
                pages=len({chunk.page for chunk in chunks}),
                chunks=len(chunks),
                tables=sum(chunk.content_type == "table" for chunk in chunks),
                metadata=chunks[0].metadata,
            )
            for document_id, chunks in sorted(grouped.items())
        ]

    def stats(self) -> CorpusStats:
        documents = self.documents()
        return CorpusStats(
            documents=len(documents),
            chunks=sum(document.chunks for document in documents),
            tables=sum(document.tables for document in documents),
            embedding_model=self.embedder.name,
            reranker_model=self.reranker.name if self.reranker else None,
        )
