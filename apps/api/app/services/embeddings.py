from __future__ import annotations

import asyncio
import hashlib
import logging
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import TypeVar, cast

from sqlalchemy import or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.chunking.repository import EmbeddingInput, list_embedding_inputs
from app.core.config import Settings
from app.db.models import EmbeddingJob, EmbeddingStatus
from app.integrations.embeddings.base import EmbeddingProvider, EmbeddingProviderError
from app.integrations.embeddings.catalog import EmbeddingModel, model_candidates
from app.integrations.vector_store.base import VectorRecord, VectorStore

logger = logging.getLogger(__name__)
ProviderFactory = Callable[[EmbeddingModel], EmbeddingProvider]
T = TypeVar("T")


class EmbeddingProcessError(RuntimeError):
    def __init__(self, code: str, message: str, *, transient: bool) -> None:
        super().__init__(message)
        self.code = code
        self.transient = transient


async def ensure_embedding_job(
    session: AsyncSession,
    snapshot_id: uuid.UUID,
    profile_key: str,
    requested_model: str | None = None,
) -> EmbeddingJob:
    selection_key = requested_model or "auto"
    existing = await session.scalar(
        select(EmbeddingJob).where(
            EmbeddingJob.snapshot_id == snapshot_id,
            EmbeddingJob.profile_key == profile_key,
            EmbeddingJob.selection_key == selection_key,
        )
    )
    if existing is not None:
        return existing
    job = EmbeddingJob(
        snapshot_id=snapshot_id,
        profile_key=profile_key,
        selection_key=selection_key,
        requested_model=requested_model,
    )
    try:
        async with session.begin_nested():
            session.add(job)
            await session.flush()
    except IntegrityError:
        existing = await session.scalar(
            select(EmbeddingJob).where(
                EmbeddingJob.snapshot_id == snapshot_id,
                EmbeddingJob.profile_key == profile_key,
                EmbeddingJob.selection_key == selection_key,
            )
        )
        if existing is None:
            raise
        return cast(EmbeddingJob, existing)
    return job


def chunk_content_hash(item: EmbeddingInput) -> str:
    return hashlib.sha256(item.text.encode()).hexdigest()


def vector_record(item: EmbeddingInput, vector: list[float], model: EmbeddingModel) -> VectorRecord:
    return VectorRecord(
        id=str(item.chunk_id),
        vector=vector,
        payload={
            "text": item.text,
            "raw_text": item.raw_text,
            "repository_id": str(item.repository_id),
            "snapshot_id": str(item.snapshot_id),
            "path": item.path,
            "language": item.language,
            "start_line": item.start_line,
            "end_line": item.end_line,
            "byte_start": item.byte_start,
            "byte_end": item.byte_end,
            "entities": item.entities,
            "scope": item.scope,
            "context": item.context,
            "chunk_fingerprint": item.chunk_fingerprint,
            "content_hash": chunk_content_hash(item),
            "profile_key": item.profile_key,
            "tokenizer_id": item.tokenizer_id,
            "token_count": item.token_count,
            "quality_grade": item.quality_grade,
            "quality_score": item.quality_score,
            "model_id": model.id,
            "dimension": model.dimension,
        },
    )


class EmbeddingProcessor:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        settings: Settings,
        provider_factory: ProviderFactory,
        vector_store: VectorStore,
    ) -> None:
        self.session_factory = session_factory
        self.settings = settings
        self.provider_factory = provider_factory
        self.vector_store = vector_store

    async def process(self, job_id: uuid.UUID) -> None:
        try:
            async with self.session_factory() as session:
                job = await session.get(EmbeddingJob, job_id)
                if job is None or job.status != EmbeddingStatus.running:
                    return
                inputs = await list_embedding_inputs(session, job.snapshot_id, job.profile_key)
                job.total_chunks = len(inputs)
                await session.commit()
            if not inputs:
                raise EmbeddingProcessError(
                    "no_embedding_inputs", "No chunks are available for embedding", transient=False
                )

            candidates = model_candidates(
                inputs,
                manual_model=job.requested_model,
                large_repository_chunks=self.settings.embedding_large_repository_chunks,
            )
            errors: list[str] = []
            for fallback_count, model in enumerate(candidates):
                provider = self.provider_factory(model)
                try:
                    await self._retry(provider.validate)
                    await self.vector_store.prepare(model.id, model.dimension)
                    stored, skipped = await self._embed_and_store(job_id, inputs, model, provider)
                except (EmbeddingProviderError, Exception) as exc:
                    # Vector-store failures are not solved by changing the embedding model.
                    if not isinstance(exc, EmbeddingProviderError):
                        raise
                    errors.append(f"{model.id}: {exc}")
                    logger.warning(
                        "Embedding model failed; trying fallback",
                        extra={"job_id": str(job_id), "model_id": model.id, "error": str(exc)},
                    )
                    continue
                finally:
                    await provider.close()
                await self._succeed(job_id, model, stored, skipped, fallback_count)
                return
            raise EmbeddingProcessError(
                "embedding_models_unavailable",
                "All supported embedding models failed: " + "; ".join(errors),
                transient=True,
            )
        except EmbeddingProcessError as exc:
            await self._fail_or_retry(job_id, exc.code, str(exc), exc.transient)
        except Exception:
            logger.exception("Repository embedding failed", extra={"job_id": str(job_id)})
            await self._fail_or_retry(
                job_id, "embedding_internal_error", "Repository embedding failed", True
            )

    async def _embed_and_store(
        self,
        job_id: uuid.UUID,
        inputs: list[EmbeddingInput],
        model: EmbeddingModel,
        provider: EmbeddingProvider,
    ) -> tuple[int, int]:
        hashes = await self.vector_store.existing_hashes(
            [str(item.chunk_id) for item in inputs], model.id, model.dimension
        )
        pending = [
            item for item in inputs if hashes.get(str(item.chunk_id)) != chunk_content_hash(item)
        ]
        skipped = len(inputs) - len(pending)
        batches = [
            pending[index : index + self.settings.embedding_batch_size]
            for index in range(0, len(pending), self.settings.embedding_batch_size)
        ]
        semaphore = asyncio.Semaphore(
            self.settings.embedding_worker_concurrency
            if len(inputs) >= self.settings.embedding_large_repository_chunks
            else 1
        )

        async def run_batch(batch: list[EmbeddingInput]) -> list[VectorRecord]:
            async with semaphore:
                vectors = await self._retry(
                    lambda: provider.embed_documents([item.text for item in batch])
                )
                return [
                    vector_record(item, vector, model)
                    for item, vector in zip(batch, vectors, strict=True)
                ]

        # Finish one model consistently before writing; a model fallback can then
        # never leave a snapshot split across incompatible vector dimensions.
        record_batches = await asyncio.gather(*(run_batch(batch) for batch in batches))
        stored = 0
        for records in record_batches:
            await self.vector_store.upsert(records, model.id, model.dimension)
            stored += len(records)
            await self._progress(job_id, stored + skipped, stored, skipped, model)
        if not record_batches:
            await self._progress(job_id, skipped, 0, skipped, model)
        return stored, skipped

    async def _retry(self, operation: Callable[[], Awaitable[T]]) -> T:
        last_error: EmbeddingProviderError | None = None
        for attempt in range(self.settings.embedding_max_retries + 1):
            try:
                return await operation()
            except EmbeddingProviderError as exc:
                last_error = exc
                if not exc.retryable or attempt >= self.settings.embedding_max_retries:
                    raise
                delay = exc.retry_after or self.settings.embedding_backoff_base_seconds * (
                    2**attempt
                )
                await asyncio.sleep(min(delay, 30.0))
        assert last_error is not None
        raise last_error

    async def _progress(
        self,
        job_id: uuid.UUID,
        processed: int,
        stored: int,
        skipped: int,
        model: EmbeddingModel,
    ) -> None:
        async with self.session_factory() as session:
            job = await session.get(EmbeddingJob, job_id)
            if job is None:
                return
            job.processed_chunks = processed
            job.stored_chunks = stored
            job.skipped_chunks = skipped
            job.selected_model = model.id
            job.dimension = model.dimension
            job.lease_expires_at = datetime.now(UTC) + timedelta(
                seconds=self.settings.embedding_lease_seconds
            )
            await session.commit()

    async def _succeed(
        self,
        job_id: uuid.UUID,
        model: EmbeddingModel,
        stored: int,
        skipped: int,
        fallback_count: int,
    ) -> None:
        async with self.session_factory() as session:
            job = await session.get(EmbeddingJob, job_id)
            if job is None:
                return
            job.status = EmbeddingStatus.succeeded
            job.selected_model = model.id
            job.dimension = model.dimension
            job.processed_chunks = job.total_chunks
            job.stored_chunks = stored
            job.skipped_chunks = skipped
            job.fallback_count = fallback_count
            job.finished_at = datetime.now(UTC)
            job.lease_expires_at = None
            job.error_code = None
            job.error_message = None
            await session.commit()

    async def _fail_or_retry(
        self, job_id: uuid.UUID, code: str, message: str, transient: bool
    ) -> None:
        async with self.session_factory() as session:
            job = await session.get(EmbeddingJob, job_id)
            if job is None:
                return
            job.error_code = code[:64]
            job.error_message = message[:1000]
            job.lease_expires_at = None
            if transient and job.attempt_count < self.settings.embedding_max_attempts:
                job.status = EmbeddingStatus.queued
                job.next_attempt_at = datetime.now(UTC) + timedelta(seconds=2**job.attempt_count)
            else:
                job.status = EmbeddingStatus.failed
                job.finished_at = datetime.now(UTC)
            await session.commit()


class EmbeddingRunner:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        settings: Settings,
        processor: EmbeddingProcessor,
    ) -> None:
        self.session_factory = session_factory
        self.settings = settings
        self.processor = processor
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="repository-embedding-runner")

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            await self._task
            self._task = None
        await self.processor.vector_store.close()

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                job_id = await self.claim_next()
            except Exception:
                logger.exception("Unable to claim repository embedding job")
                job_id = None
            if job_id:
                await self.processor.process(job_id)
                continue
            try:
                await asyncio.wait_for(
                    self._stop.wait(), timeout=self.settings.embedding_poll_seconds
                )
            except TimeoutError:
                pass

    async def claim_next(self) -> uuid.UUID | None:
        now = datetime.now(UTC)
        async with self.session_factory() as session:
            await session.execute(
                update(EmbeddingJob)
                .where(
                    EmbeddingJob.status == EmbeddingStatus.running,
                    EmbeddingJob.lease_expires_at < now,
                )
                .values(status=EmbeddingStatus.queued, lease_expires_at=None)
            )
            job = await session.scalar(
                select(EmbeddingJob)
                .where(
                    EmbeddingJob.status == EmbeddingStatus.queued,
                    or_(
                        EmbeddingJob.next_attempt_at.is_(None),
                        EmbeddingJob.next_attempt_at <= now,
                    ),
                )
                .order_by(EmbeddingJob.created_at)
                .with_for_update(skip_locked=True)
                .limit(1)
            )
            if job is None:
                await session.commit()
                return None
            job.status = EmbeddingStatus.running
            job.attempt_count += 1
            job.started_at = job.started_at or now
            job.next_attempt_at = None
            job.lease_expires_at = now + timedelta(seconds=self.settings.embedding_lease_seconds)
            await session.commit()
            return job.id
