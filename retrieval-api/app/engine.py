from __future__ import annotations

import json
import math
import os
import re
import threading
import time
import uuid
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from .chunking import ChunkingConfig, build_chunks
from .models import (
    BatchIndexResponse,
    Chunk,
    CorpusStats,
    DocumentSummary,
    ExtractedField,
    ExtractionCorrectionRecord,
    ExtractionCorrectionRequest,
    FilterRequest,
    IndexDocumentRequest,
    IndexResponse,
    RetrievalScores,
    SearchDiagnostics,
    SearchFilters,
    SearchHit,
    SearchMode,
    SearchRequest,
    SearchResponse,
)
from .models_runtime import Embedder, Reranker

TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?|\(?-?\$?\d[\d,]*(?:\.\d+)?%?\)?")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "did",
    "for",
    "from",
    "how",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "were",
    "what",
    "which",
    "with",
}


def tokenize(text: str) -> list[str]:
    tokens = TOKEN_PATTERN.findall(text.lower())
    expanded: list[str] = []
    for token in tokens:
        expanded.append(token)
        normalized = token.replace(",", "").replace("$", "").strip("()")
        if normalized != token:
            expanded.append(normalized)
    return expanded


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


class RetrievalEngine:
    """Persistent hybrid retrieval engine with idempotent document replacement."""

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

    def warmup(self) -> None:
        """Force the embedder (and reranker, if configured) to load their
        model weights now, during startup, instead of lazily on whichever
        request happens to be the first real dense search. Without this,
        the first caller after a fresh container start pays for the
        model download/load inline and can time out client-side even
        though the server is still working correctly in the background."""
        self.embedder.encode_query("warmup")
        if self.reranker is not None:
            self.reranker.score("warmup", ["warmup"])

    @property
    def chunks_path(self) -> Path:
        return self.data_dir / "chunks.jsonl"

    @property
    def vectors_path(self) -> Path:
        return self.data_dir / "vectors.npy"

    @property
    def faiss_path(self) -> Path:
        return self.data_dir / "chunks.faiss"

    @property
    def manifest_path(self) -> Path:
        return self.data_dir / "manifest.json"

    @property
    def corrections_path(self) -> Path:
        return self.data_dir / "corrections.jsonl"

    def _load(self) -> None:
        with self._lock:
            if not self.chunks_path.exists():
                return
            with self.chunks_path.open("r", encoding="utf-8") as handle:
                self._chunks = [
                    Chunk.model_validate_json(line) for line in handle if line.strip()
                ]
            if not self.vectors_path.exists():
                raise RuntimeError("persisted chunks exist without vectors.npy")
            self._vectors = np.load(self.vectors_path).astype(np.float32, copy=False)
            if len(self._chunks) != len(self._vectors):
                raise RuntimeError("persisted chunks and vectors are inconsistent")
            if self.manifest_path.exists():
                manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                stored_model = manifest.get("embedding_model")
                if stored_model and stored_model != self.embedder.name:
                    raise RuntimeError(
                        "persisted index uses embedding model "
                        f"{stored_model!r}, configured model is {self.embedder.name!r}"
                    )
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
        manifest_tmp = self.manifest_path.with_suffix(".json.tmp")

        with chunks_tmp.open("w", encoding="utf-8") as handle:
            for chunk in self._chunks:
                handle.write(chunk.model_dump_json() + "\n")
        with vectors_tmp.open("wb") as handle:
            np.save(handle, self._vectors)
        manifest_tmp.write_text(
            json.dumps(
                {
                    "embedding_model": self.embedder.name,
                    "dimensions": int(self._vectors.shape[1])
                    if self._vectors.ndim == 2 and self._vectors.size
                    else 0,
                    "chunks": len(self._chunks),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if self._faiss_index is not None:
            faiss.write_index(self._faiss_index, str(faiss_tmp))

        os.replace(chunks_tmp, self.chunks_path)
        os.replace(vectors_tmp, self.vectors_path)
        os.replace(manifest_tmp, self.manifest_path)
        if self._faiss_index is not None:
            os.replace(faiss_tmp, self.faiss_path)
        elif self.faiss_path.exists():
            self.faiss_path.unlink()

    def index_document(self, request: IndexDocumentRequest) -> IndexResponse:
        return self.index_documents([request]).documents[0]

    def index_documents(
        self, requests: list[IndexDocumentRequest]
    ) -> BatchIndexResponse:
        """Embed, rebuild and persist once for an entire batch."""
        if not requests:
            raise ValueError("at least one document is required")
        document_ids = [request.document.document_id for request in requests]
        if len(set(document_ids)) != len(document_ids):
            raise ValueError("a batch cannot contain duplicate document_id values")

        chunks_by_document: list[list[Chunk]] = []
        for request in requests:
            chunks = build_chunks(request, self.chunking)
            if not chunks:
                raise ValueError(
                    f"document {request.document.document_id!r} produced no searchable chunks"
                )
            chunks_by_document.append(chunks)
        new_chunks = [chunk for chunks in chunks_by_document for chunk in chunks]
        searchable = [self._searchable_text(chunk) for chunk in new_chunks]
        new_vectors = self.embedder.encode_documents(searchable)
        if new_vectors.ndim != 2 or len(new_vectors) != len(new_chunks):
            raise RuntimeError("embedder returned an invalid vector matrix")

        with self._lock:
            target_ids = set(document_ids)
            old_counts: dict[str, int] = defaultdict(int)
            keep: list[int] = []
            for index, chunk in enumerate(self._chunks):
                if chunk.document_id in target_ids:
                    old_counts[chunk.document_id] += 1
                else:
                    keep.append(index)
            kept_chunks = [self._chunks[index] for index in keep]
            if keep:
                kept_vectors = self._vectors[keep]
                if kept_vectors.shape[1] != new_vectors.shape[1]:
                    raise RuntimeError(
                        "embedding dimensions changed; clear or migrate the index"
                    )
                combined_vectors = np.vstack([kept_vectors, new_vectors]).astype(
                    np.float32
                )
            else:
                combined_vectors = np.asarray(new_vectors, dtype=np.float32)

            self._chunks = [*kept_chunks, *new_chunks]
            self._vectors = combined_vectors
            self._rebuild_indexes()
            self._persist()

        results = [
            IndexResponse(
                document_id=request.document.document_id,
                source_doc_uid=request.source_doc_uid,
                source_filename=request.source_filename
                or request.document.document_id,
                chunks_indexed=len(chunks),
                chunks_replaced=old_counts[request.document.document_id],
            )
            for request, chunks in zip(requests, chunks_by_document)
        ]
        return BatchIndexResponse(
            documents_indexed=len(results),
            chunks_indexed=sum(result.chunks_indexed for result in results),
            chunks_replaced=sum(result.chunks_replaced for result in results),
            documents=results,
        )

    def delete_document(self, document_id: str) -> int:
        with self._lock:
            keep = [
                index
                for index, chunk in enumerate(self._chunks)
                if chunk.document_id != document_id
                and chunk.source_doc_uid != document_id
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

    def document_chunks(self, document_id: str) -> list[ExtractedField]:
        with self._lock:
            chunks = [
                chunk
                for chunk in self._chunks
                if document_id in {chunk.document_id, chunk.source_doc_uid}
            ]
        if not chunks:
            raise KeyError("document not found")
        return [
            ExtractedField(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                page=chunk.page,
                section=chunk.section,
                content_type="table" if chunk.content_type == "table" else "text",
                content=chunk.text,
                bbox=chunk.bbox,
                source_block_ids=chunk.source_block_ids,
            )
            for chunk in sorted(chunks, key=lambda item: (item.page, item.chunk_id))
        ]

    def corrections(self, document_id: str) -> list[ExtractionCorrectionRecord]:
        with self._lock:
            canonical_ids = {
                chunk.document_id
                for chunk in self._chunks
                if document_id in {chunk.document_id, chunk.source_doc_uid}
            }
        if not canonical_ids:
            raise KeyError("document not found")
        if not self.corrections_path.exists():
            return []
        records: list[ExtractionCorrectionRecord] = []
        with self._lock, self.corrections_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = ExtractionCorrectionRecord.model_validate_json(line)
                except ValueError as exc:
                    raise RuntimeError(
                        f"invalid correction record at line {line_number}"
                    ) from exc
                if record.document_id in canonical_ids:
                    records.append(record)
        return list(reversed(records))

    def correct_extracted_field(
        self, document_id: str, request: ExtractionCorrectionRequest
    ) -> ExtractionCorrectionRecord:
        """Apply an audited text correction and rebuild persistent search indexes."""
        with self._lock:
            document_indexes = [
                index
                for index, chunk in enumerate(self._chunks)
                if document_id in {chunk.document_id, chunk.source_doc_uid}
            ]
            if not document_indexes:
                raise KeyError("document not found")
            matching = [
                index
                for index in document_indexes
                if self._chunks[index].chunk_id == request.chunk_id
            ]
            if not matching:
                raise KeyError("chunk not found in document")

            index = matching[0]
            original = self._chunks[index]
            if original.text.strip() == request.corrected_text.strip():
                raise ValueError("corrected_text must differ from the current text")

            record = ExtractionCorrectionRecord(
                correction_id=str(uuid.uuid4()),
                document_id=original.document_id,
                chunk_id=original.chunk_id,
                original_text=original.text,
                corrected_text=request.corrected_text,
                corrected_by=request.corrected_by,
                comment=request.comment,
                created_at=datetime.now(timezone.utc),
            )
            corrected_parent = (
                original.parent_text.replace(original.text, request.corrected_text, 1)
                if original.text in original.parent_text
                else request.corrected_text
            )
            corrected = original.model_copy(
                update={
                    "text": request.corrected_text,
                    "parent_text": corrected_parent,
                }
            )
            replacement = self.embedder.encode_documents(
                [self._searchable_text(corrected)]
            )
            if replacement.ndim != 2 or replacement.shape != (1, self._vectors.shape[1]):
                raise RuntimeError("embedder returned an invalid correction vector")

            self.data_dir.mkdir(parents=True, exist_ok=True)
            previous_log = (
                self.corrections_path.read_bytes()
                if self.corrections_path.exists()
                else None
            )
            log_tmp = self.corrections_path.with_suffix(".jsonl.tmp")
            log_tmp.write_bytes(
                (previous_log or b"")
                + (record.model_dump_json() + "\n").encode("utf-8")
            )

            previous_vectors = self._vectors.copy()
            self._chunks[index] = corrected
            self._vectors[index] = replacement[0]
            try:
                self._rebuild_indexes()
                self._persist()
                os.replace(log_tmp, self.corrections_path)
            except Exception:
                self._chunks[index] = original
                self._vectors = previous_vectors
                self._rebuild_indexes()
                if log_tmp.exists():
                    log_tmp.unlink()
                raise
            return record

    @staticmethod
    def _matches(chunk: Chunk, filters: SearchFilters) -> bool:
        if filters.document_id is not None and filters.document_id not in {
            chunk.document_id,
            chunk.source_doc_uid,
        }:
            return False
        if filters.source_doc_uid is not None and chunk.source_doc_uid != filters.source_doc_uid:
            return False
        if (
            filters.source_filename is not None
            and filters.source_filename.lower() not in chunk.source_filename.lower()
        ):
            return False
        if filters.page is not None and chunk.page != filters.page:
            return False
        if (
            filters.section is not None
            and filters.section.lower() not in chunk.section.lower()
        ):
            return False
        if filters.content_type is not None and chunk.content_type != filters.content_type:
            return False
        return all(
            chunk.metadata.get(key) == value for key, value in filters.metadata.items()
        )

    @staticmethod
    def _rank_map(indices: Iterable[int]) -> dict[int, int]:
        return {index: rank for rank, index in enumerate(indices, start=1)}

    def _lexical_confidences(
        self,
        query: str,
        candidate_indices: list[int],
        raw_scores: dict[int, float],
    ) -> dict[int, float]:
        query_tokens = {token for token in tokenize(query) if token not in STOPWORDS}
        max_positive = max((max(0.0, raw_scores.get(i, 0.0)) for i in candidate_indices), default=0.0)
        confidences: dict[int, float] = {}
        for index in candidate_indices:
            relative = (
                max(0.0, raw_scores.get(index, 0.0)) / max_positive
                if max_positive > 0
                else 0.0
            )
            document_tokens = set(tokenize(self._searchable_text(self._chunks[index])))
            coverage = (
                len(query_tokens & document_tokens) / len(query_tokens)
                if query_tokens
                else 0.0
            )
            confidences[index] = _clamp(0.65 * relative + 0.35 * coverage)
        return confidences

    @staticmethod
    def _dense_confidence(score: float | None) -> float:
        if score is None:
            return 0.0
        # BGE cosine similarities are compressed; center the logistic curve
        # at 0.45 rather than presenting cosine as a probability.
        return _sigmoid(8.0 * (score - 0.45))

    @staticmethod
    def _reranker_confidence(score: float | None) -> float:
        if score is None:
            return 0.0
        # The MS MARCO cross-encoder emits logits. A logit of 1 maps to 0.5,
        # making zero/negative evidence conservative for the agent's 0.35 gate.
        return _sigmoid(score - 1.0)

    def _final_confidence(
        self,
        mode: SearchMode,
        dense_score: float | None,
        lexical_confidence: float,
        fusion_score: float,
        reranker_score: float | None,
        dense_weight: float = 0.55,
        rrf_k: int = 60,
    ) -> float:
        dense_confidence = self._dense_confidence(dense_score)
        fusion_confidence = _clamp(fusion_score * (rrf_k + 1))
        if mode == SearchMode.dense:
            base = dense_confidence
        elif mode == SearchMode.lexical:
            base = lexical_confidence
        else:
            # Preserve the established 0.55/0.35 calibration at the default
            # fusion weight, while keeping nonnegative weights at the extremes.
            calibrated_dense_weight = 0.9 * dense_weight / (dense_weight + (1 - dense_weight) * 7 / 9)
            base = _clamp(
                calibrated_dense_weight * dense_confidence
                + (0.90 - calibrated_dense_weight) * lexical_confidence
                + 0.10 * fusion_confidence
            )
        if reranker_score is not None:
            base = 0.30 * base + 0.70 * self._reranker_confidence(reranker_score)
        return round(_clamp(base), 6)

    def _make_hit(
        self,
        index: int,
        rank: int,
        score: float,
        scores: RetrievalScores,
        include_parent_context: bool,
    ) -> SearchHit:
        chunk = self._chunks[index]
        public_type = "table" if chunk.content_type == "table" else "text"
        return SearchHit(
            rank=rank,
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            source_doc_uid=chunk.source_doc_uid,
            filename=chunk.source_filename,
            source_filename=chunk.source_filename,
            page=chunk.page,
            section=chunk.section,
            content_type=public_type,
            content=chunk.text,
            text=chunk.text,
            score=score,
            parent_text=chunk.parent_text if include_parent_context else None,
            bbox=chunk.bbox,
            source_block_ids=chunk.source_block_ids,
            table_title=chunk.table_title,
            table_context=chunk.table_context,
            table_cells=chunk.table_cells,
            metadata=chunk.metadata,
            scores=scores,
        )

    def filter(self, request: FilterRequest) -> SearchResponse:
        started = time.perf_counter()
        filters = request.as_filters()
        with self._lock:
            eligible = [
                index for index, chunk in enumerate(self._chunks) if self._matches(chunk, filters)
            ]
            eligible.sort(
                key=lambda index: (
                    self._chunks[index].document_id,
                    self._chunks[index].page,
                    self._chunks[index].chunk_id,
                )
            )
            hits = [
                self._make_hit(
                    index,
                    rank,
                    1.0,
                    RetrievalScores(fusion=1.0),
                    request.include_parent_context,
                )
                for rank, index in enumerate(eligible[: request.top_k], start=1)
            ]
        latency = (time.perf_counter() - started) * 1000
        return SearchResponse(
            query="[metadata filter]",
            mode=SearchMode.lexical,
            candidate_count=len(eligible),
            reranked=False,
            latency_ms=latency,
            hits=hits,
            diagnostics=SearchDiagnostics(
                eligible_count=len(eligible),
                fused_candidate_count=len(eligible),
                stage_latency_ms={"filter": latency},
            ),
        )

    def search(self, request: SearchRequest) -> SearchResponse:
        started = time.perf_counter()
        search_settings = {"dense_weight": request.dense_weight, "rrf_k": request.rrf_k,
                           "top_k": request.top_k, "candidate_k": request.candidate_k}
        timings: dict[str, float] = {}
        with self._lock:
            stage_started = time.perf_counter()
            eligible = [
                index
                for index, chunk in enumerate(self._chunks)
                if self._matches(chunk, request.filters)
            ]
            timings["filter"] = (time.perf_counter() - stage_started) * 1000
            if not eligible:
                latency = (time.perf_counter() - started) * 1000
                return SearchResponse(
                    query=request.query,
                    mode=request.mode,
                    candidate_count=0,
                    reranked=False,
                    latency_ms=latency,
                    hits=[],
                    diagnostics=SearchDiagnostics(
                        search_settings=search_settings,
                        eligible_count=0, stage_latency_ms=timings
                    ),
                )

            dense_scores: dict[int, float] = {}
            lexical_scores: dict[int, float] = {}
            dense_order: list[int] = []
            lexical_order: list[int] = []

            if request.mode in {SearchMode.hybrid, SearchMode.dense}:
                stage_started = time.perf_counter()
                query_vector = self.embedder.encode_query(request.query)
                if self._faiss_index is None or query_vector.shape[0] != self._vectors.shape[1]:
                    raise RuntimeError("query embedding is incompatible with the index")
                if len(eligible) == len(self._chunks):
                    dense_k = min(len(self._chunks), request.candidate_k)
                    scores, indices = self._faiss_index.search(
                        np.asarray([query_vector], dtype=np.float32), dense_k
                    )
                    pairs = zip(indices[0], scores[0])
                else:
                    filtered_scores = self._vectors[eligible] @ query_vector
                    positions = np.argsort(-filtered_scores)[: request.candidate_k]
                    pairs = (
                        (eligible[int(position)], filtered_scores[int(position)])
                        for position in positions
                    )
                for index, score in pairs:
                    if int(index) >= 0:
                        dense_scores[int(index)] = float(score)
                        dense_order.append(int(index))
                timings["dense"] = (time.perf_counter() - stage_started) * 1000

            if request.mode in {SearchMode.hybrid, SearchMode.lexical}:
                stage_started = time.perf_counter()
                raw_scores = (
                    self._bm25.get_scores(tokenize(request.query))
                    if self._bm25 is not None
                    else np.zeros(len(self._chunks), dtype=np.float32)
                )
                lexical_order = sorted(
                    eligible, key=lambda index: (-float(raw_scores[index]), index)
                )[: request.candidate_k]
                lexical_scores = {
                    index: float(raw_scores[index]) for index in lexical_order
                }
                timings["bm25"] = (time.perf_counter() - stage_started) * 1000

            stage_started = time.perf_counter()
            dense_ranks = self._rank_map(dense_order)
            lexical_ranks = self._rank_map(lexical_order)
            candidate_union = set(dense_order) | set(lexical_order)
            fusion: dict[int, float] = defaultdict(float)
            for index in candidate_union:
                if index in dense_ranks:
                    fusion[index] += request.dense_weight / (request.rrf_k + dense_ranks[index])
                if index in lexical_ranks:
                    fusion[index] += (1 - request.dense_weight) / (request.rrf_k + lexical_ranks[index])
            candidates = sorted(
                candidate_union, key=lambda index: (-fusion[index], index)
            )[: request.candidate_k]
            timings["fusion"] = (time.perf_counter() - stage_started) * 1000

            reranker_scores: dict[int, float] = {}
            did_rerank = bool(
                request.rerank and self.reranker is not None and candidates
            )
            if did_rerank:
                stage_started = time.perf_counter()
                passages = [
                    self._searchable_text(self._chunks[index]) for index in candidates
                ]
                reranker_values = self.reranker.score(request.query, passages)
                if len(reranker_values) != len(candidates):
                    raise RuntimeError("reranker returned an invalid score count")
                reranker_scores = dict(zip(candidates, reranker_values))
                timings["reranker"] = (time.perf_counter() - stage_started) * 1000

            lexical_confidences = self._lexical_confidences(
                request.query, candidates, lexical_scores
            )
            final_scores = {
                index: self._final_confidence(
                    request.mode,
                    dense_scores.get(index),
                    lexical_confidences.get(index, 0.0),
                    fusion[index],
                    reranker_scores.get(index),
                    request.dense_weight,
                    request.rrf_k,
                )
                for index in candidates
            }
            candidates.sort(
                key=lambda index: (-final_scores[index], -fusion[index], index)
            )
            hits = [
                self._make_hit(
                    index,
                    rank,
                    final_scores[index],
                    RetrievalScores(
                        dense=dense_scores.get(index),
                        lexical=lexical_scores.get(index),
                        fusion=fusion[index],
                        reranker=reranker_scores.get(index),
                    ),
                    request.include_parent_context,
                )
                for rank, index in enumerate(candidates[: request.top_k], start=1)
            ]

        latency = (time.perf_counter() - started) * 1000
        return SearchResponse(
            query=request.query,
            mode=request.mode,
            candidate_count=len(candidates),
            reranked=did_rerank,
            latency_ms=latency,
            hits=hits,
            diagnostics=SearchDiagnostics(
                search_settings=search_settings,
                eligible_count=len(eligible),
                dense_candidate_count=len(dense_order),
                lexical_candidate_count=len(lexical_order),
                fused_candidate_count=len(candidates),
                reranker_candidate_count=len(candidates) if did_rerank else 0,
                stage_latency_ms=timings,
            ),
        )

    def documents(self) -> list[DocumentSummary]:
        grouped: dict[str, list[Chunk]] = defaultdict(list)
        with self._lock:
            for chunk in self._chunks:
                grouped[chunk.document_id].append(chunk)
        return [
            DocumentSummary(
                document_id=document_id,
                source_doc_uid=chunks[0].source_doc_uid,
                filename=chunks[0].source_filename,
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
