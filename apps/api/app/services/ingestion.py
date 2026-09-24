from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import cast

from sqlalchemy import or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.chunking.profile import ChunkingProfile
from app.core.config import Settings
from app.db.models import ChunkingStatus, IngestionJob, IngestionStatus, Repository, SourceSnapshot
from app.integrations.snapshots.base import SnapshotError, SnapshotStore
from app.integrations.source_control.base import SourceControlError, SourceControlProvider
from app.services.chunking import ensure_chunking_job, requeue_failed_chunking_job
from app.services.embeddings import ensure_embedding_job

logger = logging.getLogger(__name__)


async def enqueue_ingestion(
    session: AsyncSession,
    repository_id: uuid.UUID,
    user_id: uuid.UUID,
    embedding_model_override: str | None = None,
) -> IngestionJob:
    active = await session.scalar(
        select(IngestionJob).where(
            IngestionJob.repository_id == repository_id,
            IngestionJob.status.in_([IngestionStatus.queued, IngestionStatus.running]),
        )
    )
    if active:
        return active
    job = IngestionJob(
        repository_id=repository_id,
        requested_by_user_id=user_id,
        embedding_model_override=embedding_model_override,
    )
    session.add(job)
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        active = await session.scalar(
            select(IngestionJob).where(
                IngestionJob.repository_id == repository_id,
                IngestionJob.status.in_([IngestionStatus.queued, IngestionStatus.running]),
            )
        )
        if active is None:
            raise
        return cast(IngestionJob, active)
    await session.refresh(job)
    return job


class IngestionProcessor:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        settings: Settings,
        source_control: SourceControlProvider,
        snapshot_store: SnapshotStore,
    ) -> None:
        self.session_factory = session_factory
        self.settings = settings
        self.source_control = source_control
        self.snapshot_store = snapshot_store
        self.chunking_profile = ChunkingProfile.from_settings(settings)

    async def process(self, job_id: uuid.UUID) -> None:
        token: str | None = None
        temporary_archive = None
        try:
            async with self.session_factory() as session:
                job = await session.get(IngestionJob, job_id)
                if job is None or job.status != IngestionStatus.running:
                    return
                repository = await session.get(Repository, job.repository_id)
                if repository is None:
                    raise SnapshotError("repository_missing", "Repository no longer exists")
                token = await self.source_control.create_installation_token(
                    repository.installation_id, repository.github_repository_id
                )
                remote = await self.source_control.get_repository(
                    token, repository.github_repository_id
                )
                if remote.id != repository.github_repository_id:
                    raise SourceControlError(
                        "repository_mismatch", "GitHub repository identity changed"
                    )
                remote = replace(remote, installation_id=repository.installation_id)
                repository.owner = remote.owner
                repository.name = remote.name
                repository.full_name = remote.full_name
                repository.private = remote.private
                repository.default_branch = remote.default_branch
                commit_sha = await self.source_control.resolve_default_branch_sha(token, remote)
                job.commit_sha = commit_sha
                repository.last_observed_sha = commit_sha
                await self._refresh_lease(session, job)
                existing = await session.scalar(
                    select(SourceSnapshot).where(
                        SourceSnapshot.repository_id == repository.id,
                        SourceSnapshot.commit_sha == commit_sha,
                    )
                )
                if existing:
                    chunking = await ensure_chunking_job(
                        session, existing.id, self.chunking_profile.key
                    )
                    if chunking.status == ChunkingStatus.failed:
                        requeue_failed_chunking_job(chunking)
                    elif chunking.status == ChunkingStatus.succeeded:
                        await ensure_embedding_job(
                            session,
                            existing.id,
                            self.chunking_profile.key,
                            job.embedding_model_override,
                        )
                    self._succeed(job, existing.id)
                    await session.commit()
                    return
                temporary_archive = self.snapshot_store.temporary_archive_path()
                await self.source_control.download_archive(
                    token,
                    remote,
                    commit_sha,
                    temporary_archive,
                    self.settings.ingestion_max_archive_bytes,
                )
                await self._refresh_lease(session, job)
                stored = await asyncio.to_thread(
                    self.snapshot_store.validate_and_publish,
                    str(repository.github_repository_id),
                    commit_sha,
                    temporary_archive,
                )
                temporary_archive = None
                snapshot = SourceSnapshot(
                    repository_id=repository.id,
                    commit_sha=commit_sha,
                    archive_key=stored.archive_key,
                    manifest_key=stored.manifest_key,
                    file_count=stored.file_count,
                    total_bytes=stored.total_bytes,
                )
                session.add(snapshot)
                await session.flush()
                await ensure_chunking_job(session, snapshot.id, self.chunking_profile.key)
                self._succeed(job, snapshot.id)
                await session.commit()
                try:
                    await self._enforce_retention(repository.id)
                except Exception:
                    logger.exception(
                        "Unable to enforce source snapshot retention",
                        extra={"repository_id": str(repository.id)},
                    )
        except (SourceControlError, SnapshotError) as exc:
            await self._fail_or_retry(job_id, exc.code, str(exc), getattr(exc, "transient", False))
        except Exception:
            await self._fail_or_retry(
                job_id, "ingestion_internal_error", "Repository ingestion failed", True
            )
        finally:
            if temporary_archive is not None:
                temporary_archive.unlink(missing_ok=True)
            if token:
                await self.source_control.revoke_installation_token(token)

    async def _refresh_lease(self, session: AsyncSession, job: IngestionJob) -> None:
        job.lease_expires_at = datetime.now(UTC) + timedelta(
            seconds=self.settings.ingestion_lease_seconds
        )
        await session.commit()

    def _succeed(self, job: IngestionJob, snapshot_id: uuid.UUID) -> None:
        job.status = IngestionStatus.succeeded
        job.snapshot_id = snapshot_id
        job.finished_at = datetime.now(UTC)
        job.lease_expires_at = None
        job.error_code = None
        job.error_message = None

    async def _fail_or_retry(
        self, job_id: uuid.UUID, code: str, message: str, transient: bool
    ) -> None:
        async with self.session_factory() as session:
            job = await session.get(IngestionJob, job_id)
            if job is None:
                return
            job.error_code = code[:64]
            job.error_message = message[:1000]
            job.lease_expires_at = None
            if transient and job.attempt_count < self.settings.ingestion_max_attempts:
                job.status = IngestionStatus.queued
                job.next_attempt_at = datetime.now(UTC) + timedelta(seconds=2**job.attempt_count)
            else:
                job.status = IngestionStatus.failed
                job.finished_at = datetime.now(UTC)
            await session.commit()

    async def _enforce_retention(self, repository_id: uuid.UUID) -> None:
        async with self.session_factory() as session:
            old = list(
                await session.scalars(
                    select(SourceSnapshot)
                    .where(SourceSnapshot.repository_id == repository_id)
                    .order_by(SourceSnapshot.created_at.desc())
                    .offset(self.settings.snapshot_retention_count)
                )
            )
            for snapshot in old:
                self.snapshot_store.delete(snapshot.archive_key, snapshot.manifest_key)
                await session.delete(snapshot)
            await session.commit()


class IngestionRunner:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        settings: Settings,
        processor: IngestionProcessor,
    ) -> None:
        self.session_factory = session_factory
        self.settings = settings
        self.processor = processor
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="repository-ingestion-runner")

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            await self._task
            self._task = None

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                job_id = await self.claim_next()
            except Exception:
                logger.exception("Unable to claim repository ingestion job")
                job_id = None
            if job_id:
                await self.processor.process(job_id)
                continue
            try:
                await asyncio.wait_for(
                    self._stop.wait(), timeout=self.settings.ingestion_poll_seconds
                )
            except TimeoutError:
                pass

    async def claim_next(self) -> uuid.UUID | None:
        now = datetime.now(UTC)
        async with self.session_factory() as session:
            await session.execute(
                update(IngestionJob)
                .where(
                    IngestionJob.status == IngestionStatus.running,
                    IngestionJob.lease_expires_at < now,
                )
                .values(status=IngestionStatus.queued, lease_expires_at=None)
            )
            job = await session.scalar(
                select(IngestionJob)
                .where(
                    IngestionJob.status == IngestionStatus.queued,
                    or_(
                        IngestionJob.next_attempt_at.is_(None), IngestionJob.next_attempt_at <= now
                    ),
                )
                .order_by(IngestionJob.created_at)
                .with_for_update(skip_locked=True)
                .limit(1)
            )
            if job is None:
                await session.commit()
                return None
            job.status = IngestionStatus.running
            job.attempt_count += 1
            job.started_at = job.started_at or now
            job.next_attempt_at = None
            job.lease_expires_at = now + timedelta(seconds=self.settings.ingestion_lease_seconds)
            await session.commit()
            return job.id
