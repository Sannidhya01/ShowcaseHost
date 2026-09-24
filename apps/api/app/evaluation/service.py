from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from app.core.config import Settings
from app.evaluation.catalog import BENCHMARK_BY_ID, BenchmarkId
from app.evaluation.datasets import BenchmarkDatasetClient
from app.evaluation.rate_limit import EvaluationRateLimiter
from app.evaluation.retrieval import EvaluationHybridRetriever, codequery_chunks, repository_chunks
from app.evaluation.scoring import (
    codequery_reference,
    codequery_relevance_scores,
    relevance_classification_scores,
    repoqa_score,
    repoqa_target_rank,
    sanitize_repoqa_output,
)
from app.integrations.generation.base import ChatMessage, GenerationProvider
from app.integrations.generation.catalog import require_generation_model
from app.integrations.keyword_search.base import SearchRecord
from app.services.chat import GenerationProviderFactory, RerankerProviderFactory
from app.services.embeddings import ProviderFactory

RunRecord = dict[str, Any]
ProgressCallback = Callable[[int, int, RunRecord], Awaitable[None]]
logger = logging.getLogger(__name__)


def exception_leaves(exc: BaseException) -> list[BaseException]:
    if isinstance(exc, BaseExceptionGroup):
        return [leaf for child in exc.exceptions for leaf in exception_leaves(child)]
    return [exc]


def evaluation_error(exc: BaseException) -> str:
    messages = []
    for leaf in exception_leaves(exc):
        message = f"{type(leaf).__name__}: {str(leaf) or 'no details'}"
        if leaf.__cause__ is not None:
            cause = leaf.__cause__
            message += f" (caused by {type(cause).__name__}: {str(cause) or 'no details'})"
        messages.append(message)
    return "; ".join(messages)[:4000]


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


async def generate(provider: GenerationProvider, messages: list[ChatMessage]) -> str:
    try:
        return "".join([part async for part in provider.stream(messages)])
    finally:
        await provider.close()


class EvaluationRunner:
    def __init__(
        self,
        settings: Settings,
        dataset_client: BenchmarkDatasetClient,
        generation_factory: GenerationProviderFactory,
        embedding_factory: ProviderFactory,
        reranker_factory: RerankerProviderFactory,
    ) -> None:
        self._settings = settings
        self._datasets = dataset_client
        self._generation_factory = generation_factory
        self._retriever = EvaluationHybridRetriever(settings, embedding_factory, reranker_factory)
        self._generation_slots = asyncio.Semaphore(settings.evaluation_max_concurrency)
        self._rate_limiter = EvaluationRateLimiter(
            settings.evaluation_requests_per_minute,
            settings.evaluation_tokens_per_minute,
        )

    async def run(
        self,
        request: RunRecord,
        progress: ProgressCallback,
        previous_result: RunRecord | None = None,
    ) -> RunRecord:
        benchmark = BENCHMARK_BY_ID[cast(BenchmarkId, request["benchmark"])]
        rows, dataset_metadata = await self._datasets.rows(
            benchmark.id,
            str(request.get("config") or benchmark.default_config),
            str(request.get("split") or benchmark.default_split),
            int(request.get("offset", 0)),
            int(request["limit"]),
            int(request["seed"]),
        )
        details_by_index: list[RunRecord | None] = [None] * len(rows)
        if previous_result is not None:
            self.validate_recovery(previous_result)
            if dataset_metadata.get("sample_fingerprint") != previous_result["dataset"].get(
                "sample_fingerprint"
            ):
                raise ValueError("The sampled examples changed; start a new evaluation instead")
            for detail in previous_result.get("examples", []):
                if "scores" not in detail:
                    continue
                index = detail.get("index")
                if (
                    not isinstance(index, int)
                    or not 0 <= index < len(rows)
                    or details_by_index[index] is not None
                ):
                    raise ValueError("Saved example indices are invalid; start a new evaluation")
                details_by_index[index] = detail
        progress_lock = asyncio.Lock()
        completed = sum(detail is not None for detail in details_by_index)
        await progress(
            completed,
            len(rows),
            self._result(
                benchmark.id,
                [item for item in details_by_index if item is not None],
                dataset_metadata,
            ),
        )

        retry_errors: dict[int, list[str]] = {}
        retry_causes: dict[int, Exception] = {}

        async def run_attempt(
            index: int,
            row: RunRecord,
            attempt: int,
            errors: list[str],
        ) -> tuple[RunRecord, Exception | None]:
            try:
                if benchmark.id == "codequeries":
                    detail = await self._run_codequery(request, row, index)
                else:
                    detail = await self._run_repoqa(request, row, index)
                detail.update(status="succeeded", attempts=attempt, retry_errors=list(errors))
                return detail, None
            except Exception as exc:
                error = evaluation_error(exc)
                errors.append(error)
                retrying = attempt <= self._settings.evaluation_example_max_retries
                detail = {
                    "index": index,
                    "status": "retrying" if retrying else "failed",
                    "attempts": attempt,
                    "error": error,
                    "retry_errors": list(errors),
                    "question": row.get("query_name") or row.get("needle", {}).get("description"),
                    "response_status": "failed",
                }
                logger.warning("Evaluation example %s attempt %s failed: %s", index, attempt, error)
                return detail, exc

        async def save_attempt(index: int, detail: RunRecord, *, final: bool) -> None:
            nonlocal completed
            async with progress_lock:
                details_by_index[index] = detail
                if final:
                    completed += 1
                completed_details = [item for item in details_by_index if item is not None]
                await progress(
                    completed,
                    len(rows),
                    self._result(benchmark.id, completed_details, dataset_metadata),
                )

        async def first_attempt(index: int, row: RunRecord) -> None:
            errors: list[str] = []
            detail, exc = await run_attempt(index, row, 1, errors)
            retry_errors[index] = errors
            if exc is not None:
                retry_causes[index] = exc
            final = exc is None or self._settings.evaluation_example_max_retries == 0
            await save_attempt(index, detail, final=final)

        # Keep the fast path fast: every pending example gets exactly one attempt
        # before any failed example enters backoff or adaptive retry handling.
        async with asyncio.TaskGroup() as tasks:
            for index, row in enumerate(rows):
                if details_by_index[index] is None:
                    tasks.create_task(first_attempt(index, row))

        async def retry_failed(index: int, row: RunRecord) -> None:
            exc = retry_causes[index]
            errors = retry_errors[index]
            for attempt in range(2, self._settings.evaluation_example_max_retries + 2):
                retry_after = max(
                    (
                        float(getattr(leaf, "retry_after", None) or 0)
                        for leaf in exception_leaves(exc)
                    ),
                    default=0,
                )
                delay = self._settings.evaluation_example_retry_delay_seconds * 2 ** (attempt - 2)
                await asyncio.sleep(max(delay, retry_after))
                detail, next_exc = await run_attempt(index, row, attempt, errors)
                final = next_exc is None or attempt > self._settings.evaluation_example_max_retries
                await save_attempt(index, detail, final=final)
                if final:
                    return
                assert next_exc is not None
                exc = next_exc

        # Retry only failures, after all first-pass work has completed. These retries
        # can use the retriever's smaller adaptive batches without slowing healthy work.
        async with asyncio.TaskGroup() as tasks:
            for index, row in enumerate(rows):
                if index in retry_causes and self._settings.evaluation_example_max_retries:
                    tasks.create_task(retry_failed(index, row))
        details = [item for item in details_by_index if item is not None]
        return self._result(benchmark.id, details, dataset_metadata)

    def validate_recovery(self, result: RunRecord) -> None:
        if not result.get("dataset", {}).get("sample_fingerprint"):
            raise ValueError("This run has no saved sample fingerprint; rerun the full evaluation")
        previous_tuning = result.get("retrieval", {})
        if not previous_tuning:
            raise ValueError(
                "This run predates retrieval-backed evaluation; rerun all examples"
            )
        if any(
            previous_tuning.get(key) != value
            for key, value in self._retriever.configuration().items()
        ):
            raise ValueError("Retrieval settings changed; rerun all examples to compare tuning")

    def _result(
        self,
        benchmark_id: BenchmarkId,
        details: list[RunRecord],
        dataset_metadata: dict[str, Any],
    ) -> RunRecord:
        scored = [detail for detail in details if "scores" in detail]
        result = {
            "metrics": self._aggregate(benchmark_id, scored),
            "summary": {
                "scored_examples": len(scored),
                "failed_examples": sum(detail.get("status") == "failed" for detail in details),
                "retrying_examples": sum(detail.get("status") == "retrying" for detail in details),
                "retried_examples": sum(detail.get("attempts", 1) > 1 for detail in details),
                "score_scope": "successful_examples_only",
            },
            "dataset": dataset_metadata,
            "examples": list(details),
        }
        result["retrieval"] = self._retriever.configuration()
        return result

    async def _generate(self, model: str, messages: list[ChatMessage]) -> str:
        async with self._generation_slots:
            # OpenRouter routes across inference providers and publishes no stable paid
            # RPM/TPM ceiling. Let its 429 + Retry-After response control backpressure
            # instead of imposing the conservative Groq-oriented local token window.
            if require_generation_model(model).provider == "groq":
                estimated_tokens = self._rate_limiter.estimate_tokens(
                    cast(list[dict[str, str]], messages),
                    self._settings.evaluation_max_completion_tokens,
                )
                await self._rate_limiter.acquire(estimated_tokens)
            return await generate(self._generation_factory(model), messages)

    async def _run_codequery(
        self,
        request: RunRecord,
        row: RunRecord,
        index: int,
    ) -> RunRecord:
        candidates = codequery_chunks(row)
        question = str(row.get("query_name", ""))
        retrieved = await self._retriever.retrieve(question, candidates)
        reference_spans = codequery_reference(row)
        gold_labels = {candidate.id: candidate.relevance_label for candidate in candidates}
        retrieved_ids = [str(record.get("id", "")) for record in retrieved]
        scores, relevance_confusion = codequery_relevance_scores(gold_labels, retrieved_ids)
        return {
            "index": index,
            "question": question,
            "repository": "/".join(str(row.get("code_file_path", "")).split("/", 2)[0:2]),
            "reference_answer": "\n".join(reference_spans),
            "evaluation_stage": "retrieval_only",
            "context_count": len(retrieved),
            "retrieval": self._retriever.metadata(len(candidates), retrieved),
            "scores": scores,
            "relevance_confusion": relevance_confusion,
        }

    async def _run_repoqa(
        self,
        request: RunRecord,
        row: RunRecord,
        index: int,
    ) -> RunRecord:
        repository = cast(dict[str, Any], row["repository"])
        needle = cast(dict[str, Any], row["needle"])
        candidates = repository_chunks(repository, str(row["language"]))
        retrieved = await self._retriever.retrieve(str(needle["description"]), candidates)
        context = self._retrieval_context(retrieved)
        prompt = (
            "The code context below contains chunks retrieved from the repository. Based only on "
            "those chunks and the function description, repeat the exact described function in a "
            "code block wrapped by ```:\n\n"
            f"Function Description:{needle['description']}\n\n{context}"
        )
        raw_response = await self._generate(
            str(request["model"]),
            cast(
                list[ChatMessage],
                [
                    {"role": "system", "content": "You perform exact function retrieval."},
                    {
                        "role": "user",
                        "content": prompt[: self._settings.evaluation_prompt_max_chars],
                    },
                ],
            ),
        )
        candidate = sanitize_repoqa_output(raw_response)
        scores = repoqa_score(raw_response, repository, str(needle["name"]))
        target_rank = repoqa_target_rank(
            [record["payload"] for record in retrieved],
            target_path=str(needle["path"]),
            target_start_line=int(needle["start_line"]) + 1,
            target_end_line=int(needle["end_line"]),
        )
        scores["reciprocal_rank"] = 1.0 / target_rank if target_rank is not None else 0.0
        return {
            "index": index,
            "language": row["language"],
            "repository": repository["repo"],
            "commit_sha": repository.get("commit_sha"),
            "needle": needle["name"],
            "description": needle["description"],
            "candidate_answer": candidate,
            "raw_response": raw_response,
            "response_status": "captured",
            "evaluation_stage": "retrieve_then_generate",
            "context_count": len(retrieved),
            "target_chunk_rank": target_rank,
            "retrieval": self._retriever.metadata(len(candidates), retrieved),
            "scores": scores,
        }

    def _retrieval_context(self, retrieved: Sequence[SearchRecord]) -> str:
        budget = max(self._settings.evaluation_prompt_max_chars - 3_000, 1_000)
        parts: list[str] = []
        used = 0
        for record in retrieved:
            payload = record["payload"]
            addition = (
                f"// Path: {payload.get('path', 'unknown')} "
                f"(lines {payload.get('start_line', '?')}-{payload.get('end_line', '?')})\n"
                f"{payload.get('raw_text', '')}"
            )
            if used >= budget:
                break
            part = addition[: budget - used]
            parts.append(part)
            used += len(part)
        return "\n\n".join(parts)

    @staticmethod
    def _aggregate(benchmark_id: BenchmarkId, details: list[RunRecord]) -> RunRecord:
        result: RunRecord = {"examples": len(details)}
        if benchmark_id == "codequeries":
            confusion = {
                key: sum(int(detail["relevance_confusion"][key]) for detail in details)
                for key in ("true_positive", "false_positive", "false_negative", "true_negative")
            }
            result.update(relevance_classification_scores(**confusion))
        else:
            count = len(details)
            result["official_pass_at_1"] = (
                sum(float(detail["scores"]["pass_at_1"]) for detail in details) / count
                if count
                else 0.0
            )
            recovered_ranks = [
                float(detail["scores"]["reciprocal_rank"])
                for detail in details
                if float(detail["scores"].get("reciprocal_rank", 0.0)) > 0
            ]
            result["mean_reciprocal_rank"] = (
                sum(recovered_ranks) / len(recovered_ranks) if recovered_ranks else 0.0
            )
        return result


class EvaluationManager:
    def __init__(self, root: Path, runner: EvaluationRunner) -> None:
        self._root = root
        self._runner = runner
        self._runs: dict[str, RunRecord] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}

    async def start(
        self,
        request: RunRecord,
        *,
        retry_of: str | None = None,
        previous_result: RunRecord | None = None,
    ) -> RunRecord:
        run_id = str(uuid.uuid4())
        record: RunRecord = {
            "id": run_id,
            "status": "queued",
            "created_at": utc_now(),
            "started_at": None,
            "finished_at": None,
            "completed_examples": 0,
            "total_examples": None,
            "request": request,
            "result": previous_result,
            "retry_of": retry_of,
            "error": None,
        }
        self._runs[run_id] = record
        await self._persist(record)
        task = asyncio.create_task(self._execute(record))
        self._tasks[run_id] = task
        task.add_done_callback(lambda _task: self._tasks.pop(run_id, None))
        return record

    async def retry_failed(self, run_id: str) -> RunRecord | None:
        previous = await self.get(run_id)
        if previous is None:
            return None
        if previous["status"] not in {"failed", "completed_with_errors", "cancelled"}:
            raise ValueError("Only a finished run with missing or failed examples can be retried")
        result = previous.get("result")
        if not isinstance(result, dict) or previous["request"].get("seed") is None:
            raise ValueError("This run has no saved sample; start a new evaluation")
        self._runner.validate_recovery(result)
        if (
            sum("scores" in detail for detail in result.get("examples", []))
            >= previous["request"]["limit"]
        ):
            raise ValueError("This run has no failed examples to retry")
        return await self.start(dict(previous["request"]), retry_of=run_id, previous_result=result)

    async def _execute(self, record: RunRecord) -> None:
        record["status"] = "running"
        record["started_at"] = utc_now()
        await self._persist(record)

        async def progress(completed: int, total: int, partial_result: RunRecord) -> None:
            record["completed_examples"] = completed
            record["total_examples"] = total
            record["result"] = partial_result
            await self._persist(record)

        try:
            if record.get("retry_of"):
                record["result"] = await self._runner.run(
                    record["request"], progress, previous_result=record["result"]
                )
            else:
                record["result"] = await self._runner.run(record["request"], progress)
            failed = record["result"].get("summary", {}).get("failed_examples", 0)
            record["status"] = "completed_with_errors" if failed else "succeeded"
        except asyncio.CancelledError:
            record["status"] = "cancelled"
            record["error"] = None
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = evaluation_error(exc)
        record["finished_at"] = utc_now()
        await self._persist(record)

    async def cancel(self, run_id: str) -> RunRecord | None:
        record = await self.get(run_id)
        if record is None:
            return None
        if record["status"] not in {"queued", "running"}:
            raise ValueError("Only queued or running evaluations can be cancelled")
        task = self._tasks.get(run_id)
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if record["status"] in {"queued", "running"}:
            record["status"] = "cancelled"
            record["error"] = None
            record["finished_at"] = utc_now()
            await self._persist(record)
        return record

    async def get(self, run_id: str) -> RunRecord | None:
        if run_id in self._runs:
            return self._runs[run_id]
        path = self._root / f"{run_id}.json"
        if not path.is_file():
            return None
        record = await asyncio.to_thread(self._read, path)
        self._runs[run_id] = record
        return record

    async def list(self) -> list[RunRecord]:
        await asyncio.to_thread(self._root.mkdir, parents=True, exist_ok=True)
        for path in await asyncio.to_thread(lambda: list(self._root.glob("*.json"))):
            if path.stem not in self._runs:
                self._runs[path.stem] = await asyncio.to_thread(self._read, path)
        return sorted(self._runs.values(), key=lambda item: str(item["created_at"]), reverse=True)

    async def close(self) -> None:
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _persist(self, record: RunRecord) -> None:
        await asyncio.to_thread(self._write, self._root / f"{record['id']}.json", record)

    @staticmethod
    def _read(path: Path) -> RunRecord:
        record = cast(RunRecord, json.loads(path.read_text(encoding="utf-8")))
        result = record.get("result")
        if isinstance(result, dict):
            examples = result.get("examples")
            if isinstance(examples, list):
                for example in examples:
                    if not isinstance(example, dict):
                        continue
                    candidate = example.get("candidate_answer")
                    raw = example.get("raw_response")
                    if "raw_response" not in example and isinstance(candidate, str):
                        example["raw_response"] = candidate
                    if "candidate_answer" not in example and isinstance(raw, str):
                        example["candidate_answer"] = raw
                    if candidate == "" and raw == "":
                        example.setdefault("response_status", "empty_legacy_response")
        return record

    @staticmethod
    def _write(path: Path, record: RunRecord) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)
