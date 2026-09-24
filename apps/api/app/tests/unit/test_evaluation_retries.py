from __future__ import annotations

import asyncio
from collections import Counter
from pathlib import Path
from typing import Any, cast

import httpx
import pytest

from app.core.config import Settings
from app.evaluation.datasets import BenchmarkDatasetClient
from app.evaluation.service import EvaluationManager, EvaluationRunner, RunRecord, evaluation_error
from app.integrations.embeddings.base import EmbeddingProviderError
from app.integrations.embeddings.mock import MockEmbeddingProvider
from app.integrations.generation.base import GenerationProvider
from app.integrations.reranking.mock import MockRerankerProvider


class Dataset:
    fingerprint = "fixed-sample"

    async def rows(self, *args: Any) -> tuple[list[RunRecord], RunRecord]:
        return [{"query_name": str(index)} for index in range(3)], {
            "sample_fingerprint": self.fingerprint,
        }


class RetryRunner(EvaluationRunner):
    def __init__(self, failures: int, retries: int = 2) -> None:
        self.dataset = Dataset()
        super().__init__(
            Settings(
                app_env="test",
                evaluation_example_max_retries=retries,
                evaluation_example_retry_delay_seconds=0,
            ),
            cast(BenchmarkDatasetClient, self.dataset),
            lambda model: cast(GenerationProvider, None),
            lambda model: MockEmbeddingProvider(),
            lambda model_id: MockRerankerProvider(model_id, [1.0] * 128),
        )
        self.failures = failures
        self.calls: Counter[int] = Counter()

    async def _run_codequery(self, request: RunRecord, row: RunRecord, index: int) -> RunRecord:
        self.calls[index] += 1
        await asyncio.sleep(0)
        if index == 1 and self.calls[index] <= self.failures:
            raise ExceptionGroup(
                "retrieval",
                [
                    ExceptionGroup(
                        "dense",
                        [EmbeddingProviderError("Hugging Face returned 503", retryable=True)],
                    )
                ],
            )
        return {
            "index": index,
            "evaluation_stage": "retrieval_only",
            "scores": {
                "relevance_accuracy": 1.0,
                "relevance_precision": 1.0,
                "relevance_recall": 1.0,
                "relevance_f1": 1.0,
            },
            "relevance_confusion": {
                "true_positive": 1,
                "false_positive": 0,
                "false_negative": 0,
                "true_negative": 0,
            },
        }


REQUEST: RunRecord = {
    "benchmark": "codequeries",
    "model": "openai/gpt-oss-20b",
    "seed": 42,
    "limit": 3,
}


def test_nested_error_reports_empty_timeout_cause() -> None:
    try:
        try:
            raise httpx.ReadTimeout("")
        except httpx.ReadTimeout as cause:
            raise EmbeddingProviderError("Hugging Face request failed: ", retryable=True) from cause
    except EmbeddingProviderError as exc:
        message = evaluation_error(ExceptionGroup("retrieval", [exc]))
    assert "EmbeddingProviderError" in message
    assert "ReadTimeout" in message


async def test_only_failed_example_retries_before_final_scores() -> None:
    runner = RetryRunner(failures=2)
    updates: list[RunRecord] = []

    async def progress(completed: int, total: int, result: RunRecord) -> None:
        updates.append(result)

    result = await runner.run(REQUEST, progress)

    assert runner.calls == {0: 1, 1: 3, 2: 1}
    assert result["metrics"]["examples"] == 3
    assert result["metrics"]["relevance_recall"] == 1
    assert result["summary"]["failed_examples"] == 0
    assert any(update["summary"]["retrying_examples"] == 1 for update in updates)
    recovered = result["examples"][1]
    assert recovered["attempts"] == 3
    assert recovered["retry_errors"] == ["EmbeddingProviderError: Hugging Face returned 503"] * 2


async def test_retries_wait_until_every_example_finishes_its_first_attempt() -> None:
    slow_example_started = asyncio.Event()
    release_slow_example = asyncio.Event()

    class PhasedRetryRunner(RetryRunner):
        async def _run_codequery(self, request: RunRecord, row: RunRecord, index: int) -> RunRecord:
            if index == 2 and not self.calls[index]:
                slow_example_started.set()
                await release_slow_example.wait()
            return await super()._run_codequery(request, row, index)

    runner = PhasedRetryRunner(failures=1)

    async def progress(completed: int, total: int, result: RunRecord) -> None:
        pass

    task = asyncio.create_task(runner.run(REQUEST, progress))
    await slow_example_started.wait()
    for _ in range(10):
        await asyncio.sleep(0)
    assert runner.calls[1] == 1
    release_slow_example.set()
    result = await task
    assert runner.calls == {0: 1, 1: 2, 2: 1}
    assert result["metrics"]["examples"] == 3


async def test_exhaustion_preserves_successes_and_recovery_only_runs_missing_example(
    tmp_path: Path,
) -> None:
    runner = RetryRunner(failures=100)
    manager = EvaluationManager(tmp_path, runner)
    original = await manager.start(dict(REQUEST))
    await manager._tasks[original["id"]]
    assert original["status"] == "completed_with_errors"
    assert original["completed_examples"] == 3
    assert original["result"]["summary"]["failed_examples"] == 1
    assert original["result"]["metrics"]["examples"] == 2
    assert original["result"]["summary"]["score_scope"] == "successful_examples_only"

    # Exercise loading from disk, as after an API restart.
    await manager.close()
    manager = EvaluationManager(tmp_path, runner)
    runner.failures = 0
    recovery = await manager.retry_failed(original["id"])
    assert recovery is not None
    await manager._tasks[recovery["id"]]
    assert runner.calls == {0: 1, 1: 4, 2: 1}
    assert recovery["status"] == "succeeded"
    assert recovery["retry_of"] == original["id"]
    assert recovery["result"]["metrics"]["examples"] == 3
    assert recovery["result"]["metrics"]["relevance_recall"] == 1
    assert original["result"]["summary"]["failed_examples"] == 1
    await manager.close()


async def test_recovery_refuses_different_tuning_or_sample(tmp_path: Path) -> None:
    runner = RetryRunner(failures=100, retries=0)
    manager = EvaluationManager(tmp_path, runner)
    original = await manager.start(dict(REQUEST))
    await manager._tasks[original["id"]]
    runner._settings.chat_dense_weight = 1.25
    with pytest.raises(ValueError, match="Retrieval settings changed"):
        await manager.retry_failed(original["id"])
    runner._settings.chat_dense_weight = 1.0
    runner.dataset.fingerprint = "different-sample"
    recovery = await manager.retry_failed(original["id"])
    assert recovery is not None
    await manager._tasks[recovery["id"]]
    assert recovery["status"] == "failed"
    assert "sampled examples changed" in recovery["error"]
    assert runner.calls == {0: 1, 1: 1, 2: 1}
    await manager.close()


async def test_cancel_during_retry_stops_further_attempts(tmp_path: Path) -> None:
    runner = RetryRunner(failures=100)
    runner._settings.evaluation_example_retry_delay_seconds = 60
    manager = EvaluationManager(tmp_path, runner)
    original = await manager.start(dict(REQUEST))
    for _ in range(100):
        if original.get("result", {}) and original["result"]["summary"]["retrying_examples"]:
            break
        await asyncio.sleep(0.001)
    assert original["result"]["summary"]["retrying_examples"] == 1
    await manager.cancel(original["id"])
    assert original["status"] == "cancelled"
    assert runner.calls[1] == 1
    await manager.close()
