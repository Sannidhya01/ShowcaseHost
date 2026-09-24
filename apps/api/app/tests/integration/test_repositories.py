from __future__ import annotations

import io
import os
import tarfile
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.chunking.client import ChunkingWorkerClient
from app.chunking.profile import ChunkingProfile
from app.chunking.repository import embed_inputs, list_embedding_inputs
from app.core.config import Settings
from app.db.base import Base
from app.db.models import (
    ChunkingJob,
    ChunkingStatus,
    CodeChunk,
    GitHubInstallation,
    IngestionJob,
    IngestionStatus,
    Repository,
    Session,
    SourceSnapshot,
    User,
    UserInstallation,
    UserRepository,
)
from app.db.session import get_db_session
from app.integrations.embeddings.mock import MockEmbeddingProvider
from app.integrations.snapshots.local import LocalSnapshotStore
from app.integrations.source_control.base import SourceControlProvider, SourceRepository
from app.main import create_app
from app.services.auth import token_hash
from app.services.chunking import ChunkingProcessor, ChunkingRunner, ensure_chunking_job
from app.services.ingestion import IngestionProcessor, IngestionRunner, enqueue_ingestion


@pytest.fixture
async def database() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    url = os.getenv("TEST_DATABASE_URL", "sqlite+aiosqlite:///:memory:")
    engine = create_async_engine(url)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.drop_all)
        await connection.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.drop_all)
    await engine.dispose()


async def seed(factory: async_sessionmaker[AsyncSession]) -> tuple[str, uuid.UUID]:
    raw_session = "test-session"
    async with factory() as session:
        user = User(github_user_id=123, github_login="octocat")
        installation = GitHubInstallation(
            id=456,
            account_id=123,
            account_login="octocat",
            account_type="User",
            repository_selection="selected",
        )
        session.add_all([user, installation])
        await session.flush()
        repository = Repository(
            github_repository_id=789,
            installation_id=installation.id,
            owner="octocat",
            name="hello-world",
            full_name="octocat/hello-world",
            private=True,
            default_branch="main",
        )
        session.add_all(
            [
                UserInstallation(user_id=user.id, installation_id=installation.id),
                Session(
                    user_id=user.id,
                    token_hash=token_hash(raw_session),
                    expires_at=datetime.now(UTC) + timedelta(hours=1),
                ),
                repository,
            ]
        )
        await session.flush()
        session.add(UserRepository(user_id=user.id, repository_id=repository.id))
        await session.commit()
        return raw_session, repository.id


@pytest.mark.usefixtures("database")
async def test_repository_listing_and_durable_job_creation(
    database: async_sessionmaker[AsyncSession], tmp_path: Path
) -> None:
    raw_session, repository_id = await seed(database)
    app = create_app(
        Settings(
            app_env="test",
            ingestion_runner_enabled=False,
            snapshot_root=tmp_path / "snapshots",
        )
    )

    async def override_session() -> AsyncIterator[AsyncSession]:
        async with database() as session:
            yield session

    app.dependency_overrides[get_db_session] = override_session
    transport = ASGITransport(app=app)
    cookies = {app.state.settings.session_cookie_name: raw_session}
    async with AsyncClient(transport=transport, base_url="http://test", cookies=cookies) as client:
        repositories = await client.get("/repositories")
        created = await client.post(f"/repositories/{repository_id}/ingestions")
        duplicate = await client.post(f"/repositories/{repository_id}/ingestions")
        repositories_with_status = await client.get("/repositories")
        status_response = await client.get(f"/ingestions/{created.json()['id']}")
        models = await client.get("/embedding-models")
        invalid_model = await client.post(
            f"/repositories/{repository_id}/ingestions",
            json={"embedding_model": "not/an-embedding-model"},
        )

    assert repositories.status_code == 200
    assert repositories.json()[0]["full_name"] == "octocat/hello-world"
    assert created.status_code == 202
    assert created.json()["status"] == "queued"
    assert duplicate.json()["id"] == created.json()["id"]
    assert repositories_with_status.json()[0]["latest_ingestion"]["id"] == created.json()["id"]
    assert repositories_with_status.json()[0]["latest_ingestion"]["status"] == "queued"
    assert status_response.json()["status"] == "queued"
    assert models.json()[0]["id"] == "BAAI/bge-m3"
    assert invalid_model.status_code == 422


async def test_repository_routes_require_authentication(
    database: async_sessionmaker[AsyncSession], tmp_path: Path
) -> None:
    app = create_app(
        Settings(
            app_env="test",
            ingestion_runner_enabled=False,
            snapshot_root=tmp_path / "snapshots",
        )
    )

    async def override_session() -> AsyncIterator[AsyncSession]:
        async with database() as session:
            yield session

    app.dependency_overrides[get_db_session] = override_session
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/repositories")

    assert response.status_code == 401


class FakeSourceControl:
    def __init__(self) -> None:
        self.downloads = 0
        self.revocations = 0

    async def create_installation_token(self, installation_id: int, repository_id: int) -> str:
        return "short-lived-token"

    async def get_repository(self, token: str, repository_id: int) -> SourceRepository:
        return SourceRepository(
            id=repository_id,
            installation_id=0,
            owner="octocat",
            name="hello-world",
            full_name="octocat/hello-world",
            private=True,
            default_branch="main",
        )

    async def resolve_default_branch_sha(self, token: str, repository: SourceRepository) -> str:
        return "a" * 40

    async def download_archive(
        self,
        token: str,
        repository: SourceRepository,
        commit_sha: str,
        destination: Path,
        max_bytes: int,
    ) -> int:
        self.downloads += 1
        content = b"print('hello')\n"
        with tarfile.open(destination, "w:gz") as output:
            info = tarfile.TarInfo("octocat-hello-world-sha/app.py")
            info.size = len(content)
            output.addfile(info, io.BytesIO(content))
        return 0

    async def revoke_installation_token(self, token: str) -> None:
        self.revocations += 1


async def test_runner_processes_and_reuses_commit_snapshot(
    database: async_sessionmaker[AsyncSession], tmp_path: Path
) -> None:
    _, repository_id = await seed(database)
    settings = Settings(
        app_env="test", ingestion_runner_enabled=False, snapshot_root=tmp_path / "snapshots"
    )
    provider = FakeSourceControl()
    processor = IngestionProcessor(
        database,
        settings,
        cast(SourceControlProvider, provider),
        LocalSnapshotStore(settings),
    )
    runner = IngestionRunner(database, settings, processor)

    async with database() as session:
        user = await session.scalar(select(User).where(User.github_user_id == 123))
        assert user is not None
        first = await enqueue_ingestion(session, repository_id, user.id)
    first_id = await runner.claim_next()
    assert first_id == first.id
    await processor.process(first.id)

    async with database() as session:
        completed = await session.get(IngestionJob, first.id)
        snapshots = list(await session.scalars(select(SourceSnapshot)))
        chunking_jobs = list(await session.scalars(select(ChunkingJob)))
        assert completed is not None and completed.status == IngestionStatus.succeeded
        assert len(snapshots) == 1
        assert len(chunking_jobs) == 1
        assert chunking_jobs[0].status == ChunkingStatus.queued
        chunking_jobs[0].status = ChunkingStatus.failed
        chunking_jobs[0].attempt_count = 3
        chunking_jobs[0].error_code = "chunking_worker_error"
        chunking_jobs[0].error_message = "old worker failure"
        chunking_jobs[0].finished_at = datetime.now(UTC)
        await session.commit()
        user = await session.scalar(select(User).where(User.github_user_id == 123))
        assert user is not None
        second = await enqueue_ingestion(session, repository_id, user.id)
    second_id = await runner.claim_next()
    assert second_id == second.id
    await processor.process(second.id)

    assert provider.downloads == 1
    assert provider.revocations == 2
    async with database() as session:
        requeued = await session.scalar(select(ChunkingJob))
        assert requeued is not None
        assert requeued.status == ChunkingStatus.queued
        assert requeued.attempt_count == 0
        assert requeued.error_code is None
        assert requeued.finished_at is None


class FakeChunkingWorker:
    restart_count = 0

    async def chunk(
        self, workspace: Path, files: tuple[object, ...], profile: ChunkingProfile
    ) -> AsyncIterator[dict[str, object]]:
        if not files:
            return
        assert (workspace / "app.py").read_text() == "print('hello')\n"
        assert len(files) == 1
        text = "# app.py\n# Defines: module\n\nprint('hello')\n"
        yield {
            "protocolVersion": 3,
            "type": "file",
            "requestId": "fake",
            "path": "app.py",
            "status": "succeeded",
            "language": "python",
            "chunks": [
                {
                    "index": 0,
                    "text": "print('hello')\n",
                    "contextualizedText": text,
                    "byteRange": {"start": 0, "end": 15},
                    "lineRange": {"start": 0, "end": 0},
                    "context": {
                        "scope": [],
                        "entities": [{"name": "module", "type": "function"}],
                        "siblings": [],
                        "imports": [],
                    },
                    "tokenCount": len(text.encode()),
                    "quality": {
                        "grade": "A",
                        "score": 100,
                        "strategy": "code-chunk",
                        "metadataCompleteness": 1.0,
                        "missingMetadata": [],
                    },
                }
            ],
            "resolvedOptions": profile.options,
            "adaptiveRetries": 0,
            "durationMs": 2,
        }

    async def stop(self) -> None:
        return None


async def test_chunking_job_persists_embedding_ready_chunks(
    database: async_sessionmaker[AsyncSession], tmp_path: Path
) -> None:
    raw_session, repository_id = await seed(database)
    settings = Settings(
        app_env="test",
        ingestion_runner_enabled=False,
        chunking_runner_enabled=False,
        snapshot_root=tmp_path / "snapshots",
    )
    source = FakeSourceControl()
    store = LocalSnapshotStore(settings)
    ingestion = IngestionProcessor(database, settings, cast(SourceControlProvider, source), store)
    ingestion_runner = IngestionRunner(database, settings, ingestion)
    async with database() as session:
        user = await session.scalar(select(User).where(User.github_user_id == 123))
        assert user is not None
        ingestion_job = await enqueue_ingestion(session, repository_id, user.id)
    assert await ingestion_runner.claim_next() == ingestion_job.id
    await ingestion.process(ingestion_job.id)

    worker = cast(ChunkingWorkerClient, FakeChunkingWorker())
    processor = ChunkingProcessor(database, settings, store, worker)
    runner = ChunkingRunner(database, settings, processor)
    chunking_job_id = await runner.claim_next()
    assert chunking_job_id is not None
    await processor.process(chunking_job_id)

    async with database() as session:
        job = await session.get(ChunkingJob, chunking_job_id)
        chunks = list(await session.scalars(select(CodeChunk)))
        assert job is not None and job.status == ChunkingStatus.succeeded
        assert job.chunks_created == 1
        assert chunks[0].start_line == 1
        assert chunks[0].quality_grade == "A"
        assert chunks[0].quality_score == 100
        inputs = await list_embedding_inputs(session, job.snapshot_id, job.profile_key)
        assert inputs[0].text.startswith("# app.py")
        assert inputs[0].entities[0] == {"name": "module", "type": "function"}
        assert inputs[0].context["imports"] == []
        assert inputs[0].context["quality"] == {
            "grade": "A",
            "score": 100,
            "strategy": "code-chunk",
            "metadataCompleteness": 1.0,
            "missingMetadata": [],
        }
        vectors = await embed_inputs(MockEmbeddingProvider(), inputs)
        assert vectors[0][0] == float(len(inputs[0].text))
        source_snapshot = await session.get(SourceSnapshot, job.snapshot_id)
        assert source_snapshot is not None
        second_snapshot = SourceSnapshot(
            repository_id=source_snapshot.repository_id,
            commit_sha="b" * 40,
            archive_key=source_snapshot.archive_key,
            manifest_key=source_snapshot.manifest_key,
            file_count=source_snapshot.file_count,
            total_bytes=source_snapshot.total_bytes,
        )
        session.add(second_snapshot)
        await session.flush()
        second_job = await ensure_chunking_job(session, second_snapshot.id, job.profile_key)
        await session.commit()

    assert await runner.claim_next() == second_job.id
    await processor.process(second_job.id)
    async with database() as session:
        reused_job = await session.get(ChunkingJob, second_job.id)
        chunks = list(await session.scalars(select(CodeChunk)))
        assert reused_job is not None and reused_job.status == ChunkingStatus.succeeded
        assert reused_job.reused_files == 1
        assert len(chunks) == 1

    app = create_app(settings)

    async def override_session() -> AsyncIterator[AsyncSession]:
        async with database() as session:
            yield session

    app.dependency_overrides[get_db_session] = override_session
    cookies = {app.state.settings.session_cookie_name: raw_session}
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test", cookies=cookies
    ) as client:
        response = await client.get(f"/chunking-jobs/{chunking_job_id}")
    assert response.status_code == 200
    assert response.json()["status"] == "succeeded"
    assert response.json()["chunks_created"] == 1
