from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from app.core.config import Settings
from app.integrations.embeddings.base import EmbeddingProvider, EmbeddingProviderError
from app.integrations.embeddings.catalog import MODEL_BY_ID, EmbeddingModel
from app.integrations.keyword_search.base import SearchRecord
from app.integrations.vector_store.base import VectorRecord
from app.services.chat import (
    RerankerProviderFactory,
    configured_retrieval_candidates,
    rerank_retrieved_chunks,
)
from app.services.embeddings import ProviderFactory

_TOKEN = re.compile(r"[A-Za-z0-9_]+")
_EVALUATION_EMBEDDING_MODEL = MODEL_BY_ID["BAAI/bge-m3"]


@dataclass(frozen=True)
class EvaluationChunk:
    id: str
    path: str
    text: str
    metadata: str
    start_line: int
    end_line: int
    relevance_label: int = 0

    def payload(self) -> dict[str, object]:
        return {
            "path": self.path,
            "language": "text",
            "start_line": self.start_line,
            "end_line": self.end_line,
            "raw_text": self.text,
            "quality_grade": "A",
            "quality_score": 100,
        }


def _tokens(value: str) -> list[str]:
    return [token.lower() for token in _TOKEN.findall(value)]


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=False))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def bm25_records(
    query: str,
    chunks: Sequence[EvaluationChunk],
    *,
    metadata_boost: float,
) -> list[SearchRecord]:
    """Rank an ephemeral benchmark corpus with metadata-boosted BM25."""
    query_terms = Counter(_tokens(query))
    if not query_terms or not chunks:
        return []
    documents: list[dict[str, float]] = []
    lengths: list[float] = []
    document_frequency: Counter[str] = Counter()
    for chunk in chunks:
        raw_terms = Counter(_tokens(chunk.text))
        metadata_terms = Counter(_tokens(chunk.metadata))
        terms = {term: float(count) for term, count in raw_terms.items()}
        for term, count in metadata_terms.items():
            terms[term] = terms.get(term, 0.0) + count * metadata_boost
        documents.append(terms)
        lengths.append(sum(terms.values()))
        document_frequency.update(terms.keys())

    average_length = sum(lengths) / len(lengths) if lengths else 1.0
    count = len(chunks)
    k1 = 1.2
    b = 0.75
    records: list[SearchRecord] = []
    for chunk, terms, length in zip(chunks, documents, lengths, strict=True):
        score = 0.0
        for term, query_count in query_terms.items():
            frequency = terms.get(term, 0.0)
            if frequency <= 0:
                continue
            matches = document_frequency[term]
            inverse_frequency = math.log(1 + (count - matches + 0.5) / (matches + 0.5))
            denominator = frequency + k1 * (1 - b + b * length / max(average_length, 1.0))
            score += query_count * inverse_frequency * frequency * (k1 + 1) / denominator
        if score > 0:
            records.append(SearchRecord(id=chunk.id, score=score, payload=chunk.payload()))
    return sorted(records, key=lambda record: float(record.get("score", 0.0)), reverse=True)


def codequery_chunks(row: dict[str, object]) -> list[EvaluationChunk]:
    result: list[EvaluationChunk] = []
    blocks = row.get("context_blocks")
    if not isinstance(blocks, list):
        return result
    default_path = str(row.get("code_file_path", "unknown"))
    annotated_spans: list[dict[str, object]] = []
    for key in ("answer_spans", "supporting_fact_spans"):
        raw_spans = row.get(key)
        if isinstance(raw_spans, list):
            annotated_spans.extend(span for span in raw_spans if isinstance(span, dict))
    for index, value in enumerate(blocks):
        if not isinstance(value, dict):
            continue
        text = str(value.get("content", ""))
        if not text:
            continue
        path = str(value.get("path") or value.get("file_path") or default_path)
        block_start = int(value.get("index", 0))
        block_end = block_start + text.count("\n")
        relevance_label = int(
            any(
                block_start <= int(str(span.get("start_line", -1)))
                and block_end >= int(str(span.get("end_line", -1)))
                and (
                    not str(span.get("span", "")).strip()
                    or str(span.get("span", "")).strip() in text
                )
                for span in annotated_spans
            )
        )
        metadata = json.dumps(
            {**{key: item for key, item in value.items() if key != "content"}, "path": path},
            default=str,
            separators=(",", ":"),
            sort_keys=True,
        )
        digest = hashlib.sha256(f"{path}:{index}:{text}".encode()).hexdigest()
        result.append(
            EvaluationChunk(
                id=digest,
                path=path,
                text=text,
                metadata=metadata,
                start_line=block_start,
                end_line=block_end,
                relevance_label=relevance_label,
            )
        )
    return result


def repository_chunks(
    repository: dict[str, object],
    language: str,
    *,
    target_chars: int = 4_000,
    overlap_ratio: float = 0.20,
) -> list[EvaluationChunk]:
    """Split every RepoQA file into overlapping, line-aligned retrieval chunks."""
    contents = repository.get("content")
    if not isinstance(contents, dict):
        return []
    result: list[EvaluationChunk] = []
    for path_value, source_value in contents.items():
        path = str(path_value)
        source = str(source_value)
        lines = source.splitlines(keepends=True) or [source]
        start = 0
        while start < len(lines):
            end = start
            size = 0
            while end < len(lines) and (size + len(lines[end]) <= target_chars or end == start):
                size += len(lines[end])
                end += 1
            text = "".join(lines[start:end])
            digest = hashlib.sha256(f"{path}:{start}:{end}:{text}".encode()).hexdigest()
            result.append(
                EvaluationChunk(
                    id=digest,
                    path=path,
                    text=text,
                    metadata=json.dumps(
                        {"path": path, "language": language},
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                    start_line=start + 1,
                    end_line=end,
                )
            )
            if end >= len(lines):
                break
            overlap_target = max(round(len(text) * overlap_ratio), 1)
            overlap_size = 0
            next_start = end
            while next_start > start + 1 and overlap_size < overlap_target:
                next_start -= 1
                overlap_size += len(lines[next_start])
            start = next_start if next_start > start else end
    return result


class EvaluationHybridRetriever:
    def __init__(
        self,
        settings: Settings,
        embedding_factory: ProviderFactory,
        reranker_factory: RerankerProviderFactory,
    ) -> None:
        self._settings = settings
        self._embedding_factory = embedding_factory
        self._reranker_factory = reranker_factory
        self._model: EmbeddingModel = _EVALUATION_EMBEDDING_MODEL
        self._document_vectors: dict[str, list[list[float]]] = {}
        self._document_locks: dict[str, asyncio.Lock] = {}
        self._partial_document_vectors: dict[str, list[list[float] | None]] = {}
        self._document_batch_sizes: dict[str, int] = {}
        self._retrieval_slots = asyncio.Semaphore(settings.evaluation_retrieval_concurrency)
        self._embedding_slots = asyncio.Semaphore(settings.evaluation_embedding_request_concurrency)

    async def retrieve(self, query: str, chunks: Sequence[EvaluationChunk]) -> list[SearchRecord]:
        if not chunks:
            return []
        async with self._retrieval_slots:
            return await self._retrieve(query, chunks)

    async def _retrieve(self, query: str, chunks: Sequence[EvaluationChunk]) -> list[SearchRecord]:
        async with asyncio.TaskGroup() as tasks:
            dense_task = tasks.create_task(self._dense_records(query, chunks))
            keyword_task = tasks.create_task(
                asyncio.to_thread(
                    bm25_records,
                    query,
                    chunks,
                    metadata_boost=self._settings.chat_keyword_metadata_boost,
                )
            )
        candidates = configured_retrieval_candidates(
            dense_task.result(), keyword_task.result(), self._settings
        )
        return await rerank_retrieved_chunks(
            query, candidates, self._settings, self._reranker_factory
        )

    async def _dense_records(
        self, query: str, chunks: Sequence[EvaluationChunk]
    ) -> list[VectorRecord]:
        async with asyncio.TaskGroup() as tasks:
            query_task = tasks.create_task(self._embed_query(query))
            documents_task = tasks.create_task(self._cached_document_vectors(chunks))
        query_vector, document_vectors = query_task.result(), documents_task.result()
        return [
            VectorRecord(
                id=chunk.id,
                score=_cosine(query_vector, vector),
                payload=chunk.payload(),
            )
            for chunk, vector in zip(chunks, document_vectors, strict=True)
        ]

    async def _embed_query(self, query: str) -> list[float]:
        async with self._embedding_slots:
            provider = self._embedding_factory(self._model)
            try:
                return await provider.embed_query(query)
            finally:
                await provider.close()

    async def _cached_document_vectors(
        self, chunks: Sequence[EvaluationChunk]
    ) -> list[list[float]]:
        key = hashlib.sha256("\0".join(chunk.id for chunk in chunks).encode()).hexdigest()
        # A failed/cancelled attempt cannot poison a shared in-flight task. Waiters
        # reuse completed vectors or take ownership after a failed owner releases the lock.
        async with self._document_locks.setdefault(key, asyncio.Lock()):
            cached = self._document_vectors.get(key)
            if cached is None:
                cached = await self._embed_documents(chunks, key)
                self._document_vectors[key] = cached
                self._partial_document_vectors.pop(key, None)
                self._document_batch_sizes.pop(key, None)
            return cached

    async def _embed_documents(
        self, chunks: Sequence[EvaluationChunk], key: str
    ) -> list[list[float]]:
        inputs = [f"{chunk.metadata}\n{chunk.text}" for chunk in chunks]
        vectors = self._partial_document_vectors.setdefault(key, [None] * len(inputs))
        batch_size = self._document_batch_sizes.get(
            key,
            min(
                self._settings.embedding_batch_size,
                self._settings.evaluation_embedding_batch_size,
            ),
        )
        batches: list[tuple[list[int], list[str]]] = []
        index = 0
        while index < len(inputs):
            if vectors[index] is not None:
                index += 1
                continue
            indices: list[int] = []
            batch: list[str] = []
            batch_chars = 0
            while index < len(inputs) and len(batch) < batch_size:
                if vectors[index] is not None:
                    break
                value = inputs[index]
                if (
                    batch
                    and batch_chars + len(value)
                    > self._settings.evaluation_embedding_batch_max_chars
                ):
                    break
                indices.append(index)
                batch.append(value)
                batch_chars += len(value)
                index += 1
            batches.append((indices, batch))

        async def embed_batch(indices: list[int], batch: list[str]) -> None:
            async with self._embedding_slots:
                provider: EmbeddingProvider = self._embedding_factory(self._model)
                try:
                    batch_vectors = await provider.embed_documents(batch)
                except EmbeddingProviderError as exc:
                    if exc.retryable:
                        self._document_batch_sizes[key] = max(1, batch_size // 2)
                    raise
                finally:
                    await provider.close()
            if len(batch_vectors) != len(batch):
                raise ValueError("Embedding provider returned the wrong number of vectors")
            for vector_index, vector in zip(indices, batch_vectors, strict=True):
                vectors[vector_index] = vector

        # Let independent batches finish even if a sibling fails. Successful vectors
        # remain in the partial cache, so the example-level retry only resubmits the
        # missing batches at the adaptively reduced size.
        outcomes = await asyncio.gather(
            *(embed_batch(indices, batch) for indices, batch in batches),
            return_exceptions=True,
        )
        for outcome in outcomes:
            if isinstance(outcome, BaseException):
                raise outcome
        if any(vector is None for vector in vectors):
            raise ValueError("Embedding provider did not complete every document batch")
        return [vector for vector in vectors if vector is not None]

    def metadata(self, candidates: int, results: Sequence[SearchRecord]) -> dict[str, object]:
        return {
            **self.configuration(),
            "corpus_chunks": candidates,
            "retrieved_chunks": len(results),
            "chunk_ids": [str(result.get("id", "")) for result in results],
        }

    def configuration(self) -> dict[str, object]:
        return evaluation_retrieval_configuration(self._settings)


def evaluation_retrieval_configuration(settings: Settings) -> dict[str, object]:
    return {
        "strategy": "hybrid_rrf_cross_encoder_rerank",
        "dense_backend": "in_memory_cosine",
        "keyword_backend": "in_memory_bm25",
        "embedding_model": _EVALUATION_EMBEDDING_MODEL.id,
        "dense_weight": settings.chat_dense_weight,
        "keyword_weight": settings.chat_keyword_weight,
        "rrf_k": settings.chat_rrf_k,
        "candidate_limit_per_channel": max(
            settings.chat_retrieval_candidate_limit, settings.chat_retrieval_limit
        ),
        "rerank_candidate_limit": max(
            settings.chat_rerank_candidate_limit, settings.chat_retrieval_limit
        ),
        "reranker_model": settings.reranker_model,
        "reranker_fallback_model": settings.reranker_fallback_model,
        "result_limit": settings.chat_retrieval_limit,
        "relevance_threshold": settings.chat_relevance_threshold,
        "relative_relevance_threshold": settings.chat_relative_relevance_threshold,
        "relevance_metric": "reranker_normalized_score",
        "metadata_boost": settings.chat_keyword_metadata_boost,
        "retrieval_concurrency": settings.evaluation_retrieval_concurrency,
        "embedding_request_concurrency": settings.evaluation_embedding_request_concurrency,
        "embedding_batch_size": settings.evaluation_embedding_batch_size,
        "embedding_batch_max_chars": settings.evaluation_embedding_batch_max_chars,
        "embedding_timeout_seconds": settings.evaluation_embedding_timeout_seconds,
    }
