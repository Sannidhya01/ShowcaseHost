from __future__ import annotations

import asyncio
from typing import Any

import pytest

from app.core.config import Settings
from app.evaluation.retrieval import (
    EvaluationChunk,
    EvaluationHybridRetriever,
    bm25_records,
    codequery_chunks,
)
from app.integrations.embeddings.base import EmbeddingProviderError
from app.integrations.embeddings.catalog import EmbeddingModel
from app.integrations.embeddings.mock import MockEmbeddingProvider
from app.integrations.keyword_search.base import SearchRecord
from app.integrations.reranking.mock import MockRerankerProvider


def embedding_provider(model: EmbeddingModel) -> MockEmbeddingProvider:
    return MockEmbeddingProvider(dimension=model.dimension)


def reranker_provider(model_id: str) -> MockRerankerProvider:
    return MockRerankerProvider(model_id, [1.0] * 128)


def test_bm25_metadata_boost_prioritizes_metadata_match_over_raw_text_match() -> None:
    metadata_match = EvaluationChunk(
        id="metadata",
        path="src/target.py",
        text="unrelated source words",
        metadata="target",
        start_line=1,
        end_line=1,
    )
    raw_match = EvaluationChunk(
        id="raw",
        path="src/other.py",
        text="target unrelated words",
        metadata="other",
        start_line=1,
        end_line=1,
    )

    records = bm25_records("target", [raw_match, metadata_match], metadata_boost=2.0)

    assert [record["id"] for record in records] == ["metadata", "raw"]


def test_codequery_chunks_attach_gold_span_only_at_annotated_location() -> None:
    gold = {
        "span": "import sys",
        "start_line": 20,
        "start_column": 0,
        "end_line": 20,
        "end_column": 10,
    }
    chunks = codequery_chunks(
        {
            "code_file_path": "owner/repo/example.py",
            "context_blocks": [
                {"index": 0, "content": "import sys\nprint('wrong location')"},
                {"index": 20, "content": "import sys\nprint('gold location')"},
            ],
            "answer_spans": [gold],
        }
    )

    assert chunks[0].relevance_label == 0
    assert chunks[1].relevance_label == 1


def test_codequery_chunks_include_supporting_facts_in_official_relevance_label() -> None:
    chunks = codequery_chunks(
        {
            "context_blocks": [{"index": 10, "content": "supporting_call()\nreturn result"}],
            "answer_spans": [],
            "supporting_fact_spans": [
                {"span": "supporting_call()", "start_line": 10, "end_line": 10}
            ],
        }
    )

    assert chunks[0].relevance_label == 1


async def test_evaluation_hybrid_retrieval_uses_configured_rrf_tuning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(
        app_env="test",
        chat_retrieval_candidate_limit=16,
        chat_retrieval_limit=5,
        chat_rrf_k=42,
        chat_dense_weight=1.7,
        chat_keyword_weight=0.6,
        chat_keyword_metadata_boost=3.0,
    )
    captured: dict[str, Any] = {}

    def capture_fusion(
        dense: list[SearchRecord],
        keyword: list[SearchRecord],
        **options: object,
    ) -> list[SearchRecord]:
        captured["dense_count"] = len(dense)
        captured["keyword_count"] = len(keyword)
        captured.update(options)
        return []

    monkeypatch.setattr("app.services.chat.reciprocal_rank_fusion", capture_fusion)
    chunks = [
        EvaluationChunk(
            id=str(index),
            path=f"src/{index}.py",
            text=f"hybrid target {index}",
            metadata=f"path src/{index}.py",
            start_line=1,
            end_line=1,
        )
        for index in range(24)
    ]

    retriever = EvaluationHybridRetriever(settings, embedding_provider, reranker_provider)
    results = await retriever.retrieve("hybrid target", chunks)

    assert results == []
    assert captured == {
        "dense_count": 16,
        "keyword_count": 16,
        "limit": 20,
        "k": 42,
        "dense_weight": 1.7,
        "keyword_weight": 0.6,
    }


async def test_failed_query_cleans_up_documents_and_next_attempt_can_reuse_corpus() -> None:
    documents_started = asyncio.Event()
    documents_cancelled = asyncio.Event()
    fail = True

    class Provider(MockEmbeddingProvider):
        async def embed_query(self, text: str) -> list[float]:
            if fail:
                await documents_started.wait()
                raise EmbeddingProviderError("temporary query failure", retryable=True)
            return [1.0, 0.0, 1.0]

        async def embed_documents(self, texts: list[str]) -> list[list[float]]:
            if fail:
                documents_started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    documents_cancelled.set()
            return [[1.0, 0.0, 1.0] for _ in texts]

    retriever = EvaluationHybridRetriever(
        Settings(app_env="test"), lambda model: Provider(), reranker_provider
    )
    chunks = [EvaluationChunk("one", "test.py", "target", "target", 1, 1)]
    with pytest.raises(ExceptionGroup):
        await retriever.retrieve("target", chunks)
    assert documents_cancelled.is_set()
    fail = False
    results = await retriever.retrieve("target", chunks)
    assert [item["id"] for item in results] == ["one"]


async def test_failed_batch_retries_smaller_without_reembedding_successful_batches() -> None:
    batches: list[list[int]] = []

    class Provider(MockEmbeddingProvider):
        async def embed_documents(self, texts: list[str]) -> list[list[float]]:
            indices = [int(text.strip()) for text in texts]
            batches.append(indices)
            if len(batches) == 2:
                raise EmbeddingProviderError("timeout", retryable=True)
            return [[float(index)] for index in indices]

    retriever = EvaluationHybridRetriever(
        Settings(app_env="test", evaluation_embedding_batch_size=4),
        lambda model: Provider(),
        reranker_provider,
    )
    chunks = [EvaluationChunk(str(i), "test.py", str(i), "", 1, 1) for i in range(10)]
    with pytest.raises(EmbeddingProviderError):
        await retriever._cached_document_vectors(chunks)
    vectors = await retriever._cached_document_vectors(chunks)
    assert batches == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9], [4, 5], [6, 7]]
    assert vectors == [[float(i)] for i in range(10)]
    assert await retriever._cached_document_vectors(chunks) == vectors
    assert len(batches) == 5


async def test_embedding_batches_respect_character_budget() -> None:
    batches: list[list[int]] = []

    class Provider(MockEmbeddingProvider):
        async def embed_documents(self, texts: list[str]) -> list[list[float]]:
            batches.append([len(text) for text in texts])
            return [[1.0] for _ in texts]

    retriever = EvaluationHybridRetriever(
        Settings(
            app_env="test",
            evaluation_embedding_batch_size=32,
            evaluation_embedding_batch_max_chars=1_000,
        ),
        lambda model: Provider(),
        reranker_provider,
    )
    chunks = [EvaluationChunk(str(i), "test.py", "x" * 600, "", 1, 1) for i in range(3)]

    await retriever._cached_document_vectors(chunks)

    assert [len(batch) for batch in batches] == [1, 1, 1]
