from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import PurePosixPath
from typing import cast

from sqlalchemy import delete, exists, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.chunking.client import ChunkingWorkerClient, ChunkingWorkerError
from app.chunking.profile import ChunkingProfile
from app.core.config import Settings
from app.db.models import (
    ChunkedFile,
    ChunkFileStatus,
    ChunkingJob,
    ChunkingStatus,
    CodeChunk,
    IngestionJob,
    SnapshotChunkFile,
    SourceSnapshot,
)
from app.integrations.snapshots.base import ManifestEntry, SnapshotStore
from app.services.embeddings import ensure_embedding_job

logger = logging.getLogger(__name__)

ENGINE_VERSION = (
    "code-chunk@0.1.14+tree-sitter@0.26.3+tree-sitter-wasm@1.1.6+"
    "logical@2+text@2+overlap@2+quality@1"
)
NORMALIZATION_VERSION = "showcasehost-chunk-v2"
NON_SOURCE_EXTENSIONS = {
    ".7z",
    ".avi",
    ".bmp",
    ".class",
    ".doc",
    ".docx",
    ".eot",
    ".gif",
    ".gz",
    ".ico",
    ".jar",
    ".jpeg",
    ".jpg",
    ".lockb",
    ".mov",
    ".mp3",
    ".mp4",
    ".o",
    ".obj",
    ".otf",
    ".pdf",
    ".png",
    ".pyc",
    ".so",
    ".tar",
    ".ttf",
    ".wav",
    ".webm",
    ".webp",
    ".woff",
    ".woff2",
    ".xls",
    ".xlsx",
    ".zip",
}
GENERATED_COMPONENTS = {
    "node_modules",
    "vendor",
    "dist",
    "build",
    "coverage",
    ".next",
    ".venv",
}


class ChunkingProcessError(Exception):
    def __init__(self, code: str, message: str, transient: bool) -> None:
        super().__init__(message)
        self.code = code
        self.transient = transient


async def ensure_chunking_job(
    session: AsyncSession, snapshot_id: uuid.UUID, profile_key: str
) -> ChunkingJob:
    existing = await session.scalar(
        select(ChunkingJob).where(
            ChunkingJob.snapshot_id == snapshot_id,
            ChunkingJob.profile_key == profile_key,
        )
    )
    if existing is not None:
        return existing
    job = ChunkingJob(snapshot_id=snapshot_id, profile_key=profile_key)
    try:
        async with session.begin_nested():
            session.add(job)
            await session.flush()
    except IntegrityError:
        existing = await session.scalar(
            select(ChunkingJob).where(
                ChunkingJob.snapshot_id == snapshot_id,
                ChunkingJob.profile_key == profile_key,
            )
        )
        if existing is None:
            raise
        return cast(ChunkingJob, existing)
    return job


def requeue_failed_chunking_job(job: ChunkingJob) -> bool:
    if job.status != ChunkingStatus.failed:
        return False
    job.status = ChunkingStatus.queued
    job.attempt_count = 0
    job.next_attempt_at = None
    job.lease_expires_at = None
    job.error_code = None
    job.error_message = None
    job.started_at = None
    job.finished_at = None
    job.discovered_files = 0
    job.supported_files = 0
    job.reused_files = 0
    job.skipped_files = 0
    job.failed_files = 0
    job.chunks_created = 0
    job.processed_bytes = 0
    job.adaptive_retries = 0
    job.has_warnings = False
    job.language_timings_ms = {}
    return True


def skip_reason(entry: ManifestEntry) -> str | None:
    path = PurePosixPath(entry.path)
    lowered = {part.lower() for part in path.parts}
    if lowered & GENERATED_COMPONENTS:
        return "generated_or_dependency_path"
    if path.name.endswith((".min.js", ".min.mjs", ".min.cjs")):
        return "minified_file"
    if path.suffix.lower() in NON_SOURCE_EXTENSIONS:
        return "unsupported_language"
    return None


def file_fingerprint(
    repository_id: uuid.UUID,
    entry: ManifestEntry,
    profile_key: str,
    resolved_options: dict[str, object],
) -> str:
    payload = {
        "repository_id": str(repository_id),
        "path": entry.path,
        "sha256": entry.sha256,
        "engine": ENGINE_VERSION,
        "profile": profile_key,
        "options": resolved_options,
        "normalization": NORMALIZATION_VERSION,
    }
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def chunk_identity(file_fingerprint_value: str, chunk: dict[str, object]) -> tuple[uuid.UUID, str]:
    byte_range = cast(dict[str, object], chunk["byteRange"])
    payload = f"{file_fingerprint_value}:{chunk['index']}:{byte_range['start']}:{byte_range['end']}"
    fingerprint = hashlib.sha256(payload.encode()).hexdigest()
    return uuid.uuid5(uuid.NAMESPACE_URL, f"showcasehost:chunk:{fingerprint}"), fingerprint


def searchable_chunk_metadata(path: str, language: str, context: dict[str, object]) -> str:
    """Produce deterministic metadata text for BM25 without mixing in raw chunk source."""
    return json.dumps(
        {"path": path, "language": language, "context": context},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


class ChunkingProcessor:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        settings: Settings,
        snapshot_store: SnapshotStore,
        worker: ChunkingWorkerClient,
    ) -> None:
        self.session_factory = session_factory
        self.settings = settings
        self.snapshot_store = snapshot_store
        self.worker = worker
        self.profile = ChunkingProfile.from_settings(settings)

    async def process(self, job_id: uuid.UUID) -> None:
        materializer = None
        materialized_entered = False
        try:
            async with self.session_factory() as session:
                job = await session.get(ChunkingJob, job_id)
                if job is None or job.status != ChunkingStatus.running:
                    return
                snapshot = await session.get(SourceSnapshot, job.snapshot_id)
                if snapshot is None:
                    raise ChunkingProcessError(
                        "snapshot_missing", "Source snapshot no longer exists", False
                    )
                materializer = self.snapshot_store.materialize(
                    snapshot.archive_key, snapshot.manifest_key
                )
                materialized = await asyncio.to_thread(materializer.__enter__)
                materialized_entered = True
                entries = materialized.files
                selected = tuple(entry for entry in entries if skip_reason(entry) is None)
                reusable = await self._find_reusable(
                    session, snapshot.repository_id, selected, job.profile_key
                )
                worker_entries = tuple(entry for entry in selected if entry.path not in reusable)
                await self._refresh_lease(session, job)

            events: list[dict[str, object]] = []
            heartbeat = asyncio.create_task(self._heartbeat(job_id))
            try:
                async for event in self.worker.chunk(
                    materialized.root, worker_entries, self.profile
                ):
                    events.append(event)
            except ChunkingWorkerError as exc:
                raise ChunkingProcessError("chunking_worker_error", str(exc), True) from exc
            finally:
                heartbeat.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await heartbeat

            async with self.session_factory() as session:
                job = await session.get(ChunkingJob, job_id)
                snapshot = await session.get(SourceSnapshot, job.snapshot_id) if job else None
                if job is None or snapshot is None:
                    return
                await self._publish(session, job, snapshot, entries, reusable, events)
                await session.commit()
                await self._garbage_collect(session)
                await session.commit()
        except ChunkingProcessError as exc:
            await self._fail_or_retry(job_id, exc.code, str(exc), exc.transient)
        except Exception:
            logger.exception("Repository chunking failed", extra={"job_id": str(job_id)})
            await self._fail_or_retry(
                job_id, "chunking_internal_error", "Repository chunking failed", True
            )
        finally:
            if materializer is not None and materialized_entered:
                await asyncio.to_thread(materializer.__exit__, None, None, None)

    async def _find_reusable(
        self,
        session: AsyncSession,
        repository_id: uuid.UUID,
        entries: tuple[ManifestEntry, ...],
        profile_key: str,
    ) -> dict[str, ChunkedFile]:
        if not entries:
            return {}
        hashes = {entry.sha256 for entry in entries}
        candidates = list(
            await session.scalars(
                select(ChunkedFile).where(
                    ChunkedFile.repository_id == repository_id,
                    ChunkedFile.profile_key == profile_key,
                    ChunkedFile.engine_version == ENGINE_VERSION,
                    ChunkedFile.content_sha256.in_(hashes),
                )
            )
        )
        by_identity = {(item.path, item.content_sha256): item for item in candidates}
        return {
            entry.path: artifact
            for entry in entries
            if (artifact := by_identity.get((entry.path, entry.sha256))) is not None
        }

    async def _publish(
        self,
        session: AsyncSession,
        job: ChunkingJob,
        snapshot: SourceSnapshot,
        entries: tuple[ManifestEntry, ...],
        reusable: dict[str, ChunkedFile],
        events: list[dict[str, object]],
    ) -> None:
        await session.execute(
            delete(SnapshotChunkFile).where(
                SnapshotChunkFile.snapshot_id == snapshot.id,
                SnapshotChunkFile.profile_key == job.profile_key,
            )
        )
        entry_by_path = {entry.path: entry for entry in entries}
        event_by_path = {str(event["path"]): event for event in events}
        successful_files = 0
        language_timings: dict[str, int] = {}
        chunks_created = 0
        adaptive_retries = 0
        failed_files = 0
        skipped_files = 0

        for entry in entries:
            reason = skip_reason(entry)
            if reason:
                session.add(
                    SnapshotChunkFile(
                        snapshot_id=snapshot.id,
                        profile_key=job.profile_key,
                        path=entry.path,
                        status=ChunkFileStatus.skipped,
                        reason=reason,
                    )
                )
                skipped_files += 1
                continue
            if artifact := reusable.get(entry.path):
                session.add(
                    SnapshotChunkFile(
                        snapshot_id=snapshot.id,
                        profile_key=job.profile_key,
                        path=entry.path,
                        chunked_file_id=artifact.id,
                        status=ChunkFileStatus.succeeded,
                    )
                )
                successful_files += 1
                continue
            event = event_by_path.get(entry.path)
            if event is None:
                raise ChunkingProcessError(
                    "incomplete_worker_result", f"Worker omitted {entry.path}", True
                )
            status = str(event["status"])
            if status == "skipped":
                session.add(
                    SnapshotChunkFile(
                        snapshot_id=snapshot.id,
                        profile_key=job.profile_key,
                        path=entry.path,
                        status=ChunkFileStatus.skipped,
                        reason=str(event.get("reason", "skipped")),
                    )
                )
                skipped_files += 1
                continue
            if status == "failed":
                session.add(
                    SnapshotChunkFile(
                        snapshot_id=snapshot.id,
                        profile_key=job.profile_key,
                        path=entry.path,
                        status=ChunkFileStatus.failed,
                        reason=str(event.get("reason", "chunking_failed")),
                        error_message=str(event.get("error", ""))[:1000] or None,
                    )
                )
                failed_files += 1
                continue
            chunks = cast(list[dict[str, object]], event.get("chunks", []))
            if not chunks:
                failed_files += 1
                session.add(
                    SnapshotChunkFile(
                        snapshot_id=snapshot.id,
                        profile_key=job.profile_key,
                        path=entry.path,
                        status=ChunkFileStatus.failed,
                        reason="no_chunks",
                    )
                )
                continue
            options = cast(dict[str, object], event["resolvedOptions"])
            fingerprint = file_fingerprint(snapshot.repository_id, entry, job.profile_key, options)
            artifact = ChunkedFile(
                repository_id=snapshot.repository_id,
                path=entry.path,
                content_sha256=entry.sha256,
                language=str(event["language"]),
                fingerprint=fingerprint,
                engine_version=ENGINE_VERSION,
                profile_key=job.profile_key,
                resolved_options=options,
            )
            session.add(artifact)
            await session.flush()
            for chunk_data in chunks:
                chunk = chunk_data
                chunk_id, chunk_fingerprint_value = chunk_identity(fingerprint, chunk)
                byte_range = cast(dict[str, object], chunk["byteRange"])
                line_range = cast(dict[str, object], chunk["lineRange"])
                quality = cast(dict[str, object], chunk["quality"])
                context = dict(cast(dict[str, object], chunk["context"]))
                context["quality"] = quality
                session.add(
                    CodeChunk(
                        id=chunk_id,
                        chunked_file_id=artifact.id,
                        chunk_index=cast(int, chunk["index"]),
                        fingerprint=chunk_fingerprint_value,
                        raw_text=str(chunk["text"]),
                        searchable_metadata=searchable_chunk_metadata(
                            entry.path, str(event["language"]), context
                        ),
                        embedding_text=str(chunk["contextualizedText"]),
                        byte_start=cast(int, byte_range["start"]),
                        byte_end=cast(int, byte_range["end"]),
                        start_line=cast(int, line_range["start"]) + 1,
                        end_line=cast(int, line_range["end"]) + 1,
                        context=context,
                        tokenizer_id=self.profile.tokenizer_id,
                        token_count=cast(int, chunk["tokenCount"]),
                        quality_grade=str(quality["grade"]),
                        quality_score=cast(int, quality["score"]),
                    )
                )
            session.add(
                SnapshotChunkFile(
                    snapshot_id=snapshot.id,
                    profile_key=job.profile_key,
                    path=entry.path,
                    chunked_file_id=artifact.id,
                    status=ChunkFileStatus.succeeded,
                )
            )
            successful_files += 1
            chunks_created += len(chunks)
            adaptive_retries += cast(int, event.get("adaptiveRetries", 0))
            language = str(event["language"])
            language_timings[language] = language_timings.get(language, 0) + int(
                cast(int, event.get("durationMs", 0))
            )

        supported_files = sum(1 for entry in entries if skip_reason(entry) is None)
        if supported_files == 0 or successful_files == 0:
            raise ChunkingProcessError(
                "no_chunkable_files", "Snapshot contains no successfully chunked code files", False
            )
        job.discovered_files = len(entries)
        job.supported_files = supported_files
        job.reused_files = len(reusable)
        job.skipped_files = skipped_files
        job.failed_files = failed_files
        job.chunks_created = chunks_created
        job.processed_bytes = sum(
            entry_by_path[path].size
            for path, event in event_by_path.items()
            if event.get("status") == "succeeded" and path in entry_by_path
        )
        job.adaptive_retries = adaptive_retries
        job.has_warnings = skipped_files > 0 or failed_files > 0
        job.language_timings_ms = cast(dict[str, object], language_timings)
        job.status = ChunkingStatus.succeeded
        job.finished_at = datetime.now(UTC)
        job.lease_expires_at = None
        job.error_code = None
        job.error_message = None
        ingestion = await session.scalar(
            select(IngestionJob)
            .where(IngestionJob.snapshot_id == snapshot.id)
            .order_by(IngestionJob.created_at.desc())
        )
        await ensure_embedding_job(
            session,
            snapshot.id,
            job.profile_key,
            ingestion.embedding_model_override if ingestion else None,
        )

    async def _refresh_lease(self, session: AsyncSession, job: ChunkingJob) -> None:
        job.lease_expires_at = datetime.now(UTC) + timedelta(
            seconds=self.settings.chunking_lease_seconds
        )
        await session.commit()

    async def _heartbeat(self, job_id: uuid.UUID) -> None:
        interval = max(1, self.settings.chunking_lease_seconds // 3)
        while True:
            await asyncio.sleep(interval)
            async with self.session_factory() as session:
                await session.execute(
                    update(ChunkingJob)
                    .where(
                        ChunkingJob.id == job_id,
                        ChunkingJob.status == ChunkingStatus.running,
                    )
                    .values(
                        lease_expires_at=datetime.now(UTC)
                        + timedelta(seconds=self.settings.chunking_lease_seconds)
                    )
                )
                await session.commit()

    async def _fail_or_retry(
        self, job_id: uuid.UUID, code: str, message: str, transient: bool
    ) -> None:
        async with self.session_factory() as session:
            job = await session.get(ChunkingJob, job_id)
            if job is None:
                return
            job.error_code = code[:64]
            job.error_message = message[:1000]
            job.lease_expires_at = None
            if transient and job.attempt_count < self.settings.chunking_max_attempts:
                job.status = ChunkingStatus.queued
                job.next_attempt_at = datetime.now(UTC) + timedelta(seconds=2**job.attempt_count)
            else:
                job.status = ChunkingStatus.failed
                job.finished_at = datetime.now(UTC)
            await session.commit()

    async def _garbage_collect(self, session: AsyncSession) -> None:
        orphan_ids = list(
            await session.scalars(
                select(ChunkedFile.id).where(
                    ~exists().where(SnapshotChunkFile.chunked_file_id == ChunkedFile.id)
                )
            )
        )
        if orphan_ids:
            await session.execute(delete(ChunkedFile).where(ChunkedFile.id.in_(orphan_ids)))


class ChunkingRunner:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        settings: Settings,
        processor: ChunkingProcessor,
    ) -> None:
        self.session_factory = session_factory
        self.settings = settings
        self.processor = processor
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="repository-chunking-runner")

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            await self._task
            self._task = None
        await self.processor.worker.stop()

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                job_id = await self.claim_next()
            except Exception:
                logger.exception("Unable to claim repository chunking job")
                job_id = None
            if job_id:
                await self.processor.process(job_id)
                continue
            try:
                await asyncio.wait_for(
                    self._stop.wait(), timeout=self.settings.chunking_poll_seconds
                )
            except TimeoutError:
                pass

    async def claim_next(self) -> uuid.UUID | None:
        now = datetime.now(UTC)
        async with self.session_factory() as session:
            await session.execute(
                update(ChunkingJob)
                .where(
                    ChunkingJob.status == ChunkingStatus.running,
                    ChunkingJob.lease_expires_at < now,
                )
                .values(status=ChunkingStatus.queued, lease_expires_at=None)
            )
            job = await session.scalar(
                select(ChunkingJob)
                .where(
                    ChunkingJob.status == ChunkingStatus.queued,
                    or_(
                        ChunkingJob.next_attempt_at.is_(None),
                        ChunkingJob.next_attempt_at <= now,
                    ),
                )
                .order_by(ChunkingJob.created_at)
                .with_for_update(skip_locked=True)
                .limit(1)
            )
            if job is None:
                await session.commit()
                return None
            job.status = ChunkingStatus.running
            job.attempt_count += 1
            job.started_at = job.started_at or now
            job.next_attempt_at = None
            job.lease_expires_at = now + timedelta(seconds=self.settings.chunking_lease_seconds)
            await session.commit()
            return job.id
