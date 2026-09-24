from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from app.core.config import Settings
from app.evaluation.catalog import BenchmarkId
from app.evaluation.datasets import BenchmarkDatasetClient
from app.evaluation.service import EvaluationManager, EvaluationRunner, RunRecord
from app.integrations.embeddings.catalog import EmbeddingModel
from app.integrations.embeddings.mock import MockEmbeddingProvider
from app.integrations.generation.base import ChatMessage, GenerationProvider
from app.integrations.reranking.mock import MockRerankerProvider


def embedding_provider(model: EmbeddingModel) -> MockEmbeddingProvider:
    return MockEmbeddingProvider(dimension=model.dimension)


def reranker_provider(model_id: str) -> MockRerankerProvider:
    return MockRerankerProvider(model_id, [1.0] * 128)


class FakeDatasetClient:
    async def rows(
        self,
        benchmark_id: BenchmarkId,
        config: str,
        split: str,
        offset: int,
        limit: int,
        seed: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        assert benchmark_id == "codequeries"
        assert (config, split, offset, limit) == ("ideal", "test", 0, 1)
        assert seed == 7
        return [
            {
                "query_name": "Unused import",
                "code_file_path": "owner/repo/example.py",
                "context_blocks": [{"index": 0, "content": "import os\nprint('ok')"}],
                "answer_spans": [
                    {
                        "span": "import os",
                        "start_line": 0,
                        "start_column": 0,
                        "end_line": 0,
                        "end_column": 9,
                    }
                ],
            }
        ], {}


async def test_codequeries_runner_scores_retrieved_chunks_without_generation() -> None:
    def unexpected_generation(model: str) -> GenerationProvider:
        raise AssertionError("CodeQueries must not invoke generation")

    runner = EvaluationRunner(
        Settings(app_env="test"),
        cast(BenchmarkDatasetClient, FakeDatasetClient()),
        unexpected_generation,
        embedding_provider,
        reranker_provider,
    )
    progress: list[tuple[int, int]] = []

    async def record_progress(completed: int, total: int, result: RunRecord) -> None:
        progress.append((completed, total))

    result = await runner.run(
        {"benchmark": "codequeries", "model": "qwen/qwen3.8-27b", "limit": 1, "seed": 7},
        record_progress,
    )

    assert result["metrics"] == {
        "examples": 1,
        "relevance_accuracy": 1.0,
        "relevance_precision": 1.0,
        "relevance_recall": 1.0,
        "relevance_f1": 1.0,
    }
    assert result["examples"][0]["scores"]["relevance_recall"] == 1.0
    assert result["examples"][0]["evaluation_stage"] == "retrieval_only"
    assert "candidate_answer" not in result["examples"][0]
    assert "raw_response" not in result["examples"][0]
    assert result["retrieval"] == {
        "strategy": "hybrid_rrf_cross_encoder_rerank",
        "dense_backend": "in_memory_cosine",
        "keyword_backend": "in_memory_bm25",
        "embedding_model": "BAAI/bge-m3",
        "dense_weight": 1.0,
        "keyword_weight": 1.0,
        "rrf_k": 60,
        "candidate_limit_per_channel": 32,
        "rerank_candidate_limit": 20,
        "reranker_model": "BAAI/bge-reranker-v2-m3",
        "reranker_fallback_model": "BAAI/bge-reranker-large",
        "result_limit": 5,
        "relevance_threshold": 0.00017189,
        "relative_relevance_threshold": 0.855,
        "relevance_metric": "reranker_normalized_score",
        "metadata_boost": 2.0,
        "retrieval_concurrency": 2,
        "embedding_request_concurrency": 4,
        "embedding_batch_size": 32,
        "embedding_batch_max_chars": 32_000,
        "embedding_timeout_seconds": 180.0,
    }
    assert result["examples"][0]["retrieval"]["corpus_chunks"] == 1
    assert result["examples"][0]["retrieval"]["retrieved_chunks"] == 1
    assert progress == [(0, 1), (1, 1)]


class FixedProvider(GenerationProvider):
    def __init__(self, response: str) -> None:
        self.response = response

    async def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        yield self.response

    async def close(self) -> None:
        return None


async def test_tuning_changes_codequery_retrieval_without_generation() -> None:
    class DirectionalEmbeddings(MockEmbeddingProvider):
        async def embed_query(self, text: str) -> list[float]:
            return [1.0, 0.0]

        async def embed_documents(self, texts: list[str]) -> list[list[float]]:
            # Both candidates pass the relevance gate so this test isolates RRF weights.
            return [[1.0, 0.0] if "semantic_marker" in text else [0.6, 0.8] for text in texts]

    dense_source = "def semantic_marker():\n    return lexical_target + unrelated_identifier\n"
    keyword_source = "def lexical_target():\n    return 2\n"
    row: RunRecord = {
        "query_name": "lexical_target",
        "code_file_path": "repo/example.py",
        "context_blocks": [
            {"index": 0, "content": dense_source},
            {"index": 20, "content": keyword_source},
        ],
        "answer_spans": [
            {
                "span": "def lexical_target()",
                "start_line": 20,
                "start_column": 0,
                "end_line": 20,
                "end_column": 20,
            }
        ],
    }
    details = []
    for dense_weight, keyword_weight in [(1.25, 1.0), (1.0, 1.25)]:
        runner = EvaluationRunner(
            Settings(
                app_env="test",
                chat_dense_weight=dense_weight,
                chat_keyword_weight=keyword_weight,
                chat_retrieval_limit=1,
            ),
            cast(BenchmarkDatasetClient, FakeDatasetClient()),
            lambda model: cast(GenerationProvider, None),
            lambda model: DirectionalEmbeddings(),
            reranker_provider,
        )
        details.append(await runner._run_codequery({"model": "openai/gpt-oss-20b"}, row, 0))

    assert details[0]["retrieval"]["dense_weight"] == 1.25
    assert details[1]["retrieval"]["keyword_weight"] == 1.25
    assert details[0]["retrieval"]["chunk_ids"] != details[1]["retrieval"]["chunk_ids"]
    assert details[0]["scores"]["relevance_recall"] == 0
    assert details[1]["scores"]["relevance_recall"] == 1


async def test_repoqa_chunks_embeds_retrieves_then_scores_generated_answer() -> None:
    messages_seen: list[str] = []
    embedded_documents: list[str] = []

    class CapturingProvider(FixedProvider):
        async def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
            messages_seen.append(messages[-1]["content"])
            yield self.response

    source = "def lexical_target():\n    return 2\n"
    row: RunRecord = {
        "language": "python",
        "repository": {
            "repo": "owner/repo",
            "content": {"target.py": source},
            "needles": [
                {"name": "lexical_target", "path": "target.py", "start_line": 0, "end_line": 2}
            ],
        },
        "needle": {
            "name": "lexical_target",
            "path": "target.py",
            "description": "lexical_target",
            "start_line": 0,
            "end_line": 2,
        },
    }

    class CapturingEmbeddings(MockEmbeddingProvider):
        async def embed_documents(self, texts: list[str]) -> list[list[float]]:
            embedded_documents.extend(texts)
            return await super().embed_documents(texts)

    runner = EvaluationRunner(
        Settings(app_env="test"),
        cast(BenchmarkDatasetClient, FakeDatasetClient()),
        lambda model: CapturingProvider("```python\n" + source + "```"),
        lambda model: CapturingEmbeddings(dimension=model.dimension),
        reranker_provider,
    )

    detail = await runner._run_repoqa({"model": "openai/gpt-oss-20b"}, row, 0)

    assert source.strip() in messages_seen[0]
    assert embedded_documents
    assert detail["scores"]["pass_at_1"] == 1
    assert detail["scores"]["reciprocal_rank"] == 1
    assert detail["target_chunk_rank"] == 1
    assert detail["evaluation_stage"] == "retrieve_then_generate"
    assert detail["retrieval"]["corpus_chunks"] == 1


def test_codequery_aggregate_uses_official_block_classification_metrics() -> None:
    details = [
        {
            "relevance_confusion": {
                "true_positive": 1,
                "false_positive": 1,
                "false_negative": 0,
                "true_negative": 2,
            },
            "scores": {},
        },
        {
            "relevance_confusion": {
                "true_positive": 0,
                "false_positive": 0,
                "false_negative": 1,
                "true_negative": 1,
            },
            "scores": {},
        },
    ]

    assert EvaluationRunner._aggregate("codequeries", details) == {
        "examples": 2,
        "relevance_accuracy": 4 / 6,
        "relevance_precision": 0.5,
        "relevance_recall": 0.5,
        "relevance_f1": 0.5,
    }


def test_repoqa_aggregate_scores_generated_answer_and_recovered_chunk_mrr() -> None:
    details = [
        {"scores": {"pass_at_1": 1, "best_similarity": 1.0, "reciprocal_rank": 1.0}},
        {"scores": {"pass_at_1": 0, "best_similarity": 0.5, "reciprocal_rank": 0.5}},
        {"scores": {"pass_at_1": 0, "best_similarity": 0.2, "reciprocal_rank": 0.0}},
    ]

    assert EvaluationRunner._aggregate("repoqa", details) == {
        "examples": 3,
        "official_pass_at_1": 1 / 3,
        "mean_reciprocal_rank": 0.75,
    }


async def test_repoqa_retains_raw_and_normalized_candidate_answers() -> None:
    raw_response = "```python\ndef target():\n    return 42\n```"
    runner = EvaluationRunner(
        Settings(app_env="test"),
        cast(BenchmarkDatasetClient, FakeDatasetClient()),
        lambda model: FixedProvider(raw_response),
        embedding_provider,
        reranker_provider,
    )
    source = "def target():\n    return 42\n"
    row: RunRecord = {
        "language": "python",
        "repository": {
            "repo": "owner/repo",
            "commit_sha": "abc123",
            "content": {"example.py": source},
            "needles": [
                {
                    "name": "target",
                    "path": "example.py",
                    "start_line": 0,
                    "end_line": 2,
                }
            ],
        },
        "needle": {
            "name": "target",
            "path": "example.py",
            "description": "Returns the answer.",
            "start_line": 0,
            "end_line": 2,
            "start_byte": 0,
            "end_byte": len(source.encode()),
        },
    }

    detail = await runner._run_repoqa({"model": "qwen/qwen3.8-27b"}, row, 0)

    assert detail["candidate_answer"] == "def target():\n    return 42"
    assert detail["raw_response"] == raw_response

    wrong_answer_runner = EvaluationRunner(
        Settings(app_env="test"),
        cast(BenchmarkDatasetClient, FakeDatasetClient()),
        lambda model: FixedProvider("```python\ndef unrelated():\n    return 0\n```"),
        embedding_provider,
        reranker_provider,
    )
    wrong_answer = await wrong_answer_runner._run_repoqa({"model": "openai/gpt-oss-20b"}, row, 0)
    assert wrong_answer["scores"]["pass_at_1"] == 0


async def test_openrouter_uses_upstream_backpressure_instead_of_local_limiter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = EvaluationRunner(
        Settings(app_env="test"),
        cast(BenchmarkDatasetClient, FakeDatasetClient()),
        lambda model: FixedProvider('{"answer_spans":["import os"]}'),
        embedding_provider,
        reranker_provider,
    )
    acquire = AsyncMock()
    monkeypatch.setattr(runner._rate_limiter, "acquire", acquire)

    await runner._generate(
        "openai/gpt-oss-20b",
        cast(list[ChatMessage], [{"role": "user", "content": "Find it"}]),
    )

    acquire.assert_not_awaited()


class ConcurrentDatasetClient:
    async def rows(
        self,
        benchmark_id: BenchmarkId,
        config: str,
        split: str,
        offset: int,
        limit: int,
        seed: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        source = "def target():\n    return 42\n"
        row = {
            "language": "python",
            "repository": {
                "repo": "owner/repo",
                "content": {"example.py": source},
                "needles": [
                    {
                        "name": "target",
                        "path": "example.py",
                        "start_line": 0,
                        "end_line": 2,
                    }
                ],
            },
            "needle": {
                "name": "target",
                "path": "example.py",
                "description": "Returns 42.",
                "start_line": 0,
                "end_line": 2,
            },
        }
        return [dict(row) for _ in range(limit)], {}


class ConcurrencyTracker:
    def __init__(self) -> None:
        self.active = 0
        self.maximum = 0


class TrackedProvider(GenerationProvider):
    def __init__(self, tracker: ConcurrencyTracker) -> None:
        self.tracker = tracker

    async def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        self.tracker.active += 1
        self.tracker.maximum = max(self.tracker.maximum, self.tracker.active)
        await asyncio.sleep(0.01)
        self.tracker.active -= 1
        yield "```python\ndef target():\n    return 42\n```"

    async def close(self) -> None:
        return None


async def test_runner_uses_configured_parallel_generation_slots() -> None:
    tracker = ConcurrencyTracker()
    runner = EvaluationRunner(
        Settings(app_env="test", evaluation_max_concurrency=2),
        cast(BenchmarkDatasetClient, ConcurrentDatasetClient()),
        lambda model: TrackedProvider(tracker),
        embedding_provider,
        reranker_provider,
    )
    completed: list[int] = []

    async def record_progress(count: int, total: int, result: RunRecord) -> None:
        completed.append(count)

    result = await runner.run(
        {
            "benchmark": "repoqa",
            "model": "openai/gpt-oss-20b",
            "limit": 4,
            "seed": 7,
        },
        record_progress,
    )

    assert tracker.maximum == 2
    assert completed == [0, 1, 2, 3, 4]
    assert result["metrics"]["examples"] == 4


class FakeRunner:
    async def run(self, request: RunRecord, progress: object) -> RunRecord:
        return {"metrics": {"examples": 0}, "examples": []}


async def test_manager_persists_completed_run(tmp_path: Path) -> None:
    manager = EvaluationManager(tmp_path, cast(EvaluationRunner, FakeRunner()))
    record = await manager.start({"benchmark": "codequeries"})
    await manager._tasks[record["id"]]
    stored = await manager.get(str(record["id"]))
    assert stored is not None
    assert stored["status"] == "succeeded"
    assert (tmp_path / f"{record['id']}.json").is_file()
    await manager.close()


class BlockingRunner:
    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def run(self, request: RunRecord, progress: object) -> RunRecord:
        self.started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


async def test_manager_cancels_a_running_evaluation(tmp_path: Path) -> None:
    runner = BlockingRunner()
    manager = EvaluationManager(tmp_path, cast(EvaluationRunner, runner))
    record = await manager.start({"benchmark": "codequeries"})
    await runner.started.wait()

    cancelled = await manager.cancel(str(record["id"]))

    assert cancelled is not None
    assert cancelled["status"] == "cancelled"
    persisted = json.loads((tmp_path / f"{record['id']}.json").read_text(encoding="utf-8"))
    assert persisted["status"] == "cancelled"
    await manager.close()


def test_read_backfills_legacy_example_response_fields(tmp_path: Path) -> None:
    path = tmp_path / "legacy.json"
    path.write_text(
        json.dumps(
            {
                "result": {
                    "examples": [
                        {"candidate_answer": "legacy response"},
                        {"raw_response": "raw only"},
                        {"candidate_answer": "", "raw_response": ""},
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    record = EvaluationManager._read(path)

    assert record["result"]["examples"][0]["raw_response"] == "legacy response"
    assert record["result"]["examples"][1]["candidate_answer"] == "raw only"
    assert record["result"]["examples"][2]["response_status"] == "empty_legacy_response"
