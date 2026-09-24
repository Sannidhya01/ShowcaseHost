from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.config import Settings
from app.db.models import (
    ChunkedFile,
    ChunkFileStatus,
    CodeChunk,
    EmbeddingJob,
    EmbeddingStatus,
    GitHubInstallation,
    Repository,
    SnapshotChunkFile,
    SourceSnapshot,
)
from app.integrations.embeddings.mock import MockEmbeddingProvider
from app.integrations.keyword_search.pg_search import PgSearchKeywordSearch
from app.integrations.reranking.mock import MockRerankerProvider
from app.integrations.vector_store.mock import MockVectorStore
from app.services.chat import prepare_repository_chat

pytestmark = [
    pytest.mark.pg_search_e2e,
    pytest.mark.skipif(
        os.getenv("RUN_PG_SEARCH_E2E") != "1",
        reason="set RUN_PG_SEARCH_E2E=1 with a migrated ParadeDB test database",
    ),
]


async def test_pg_search_metadata_boost_and_hybrid_rrf_use_stable_chunk_ids() -> None:
    database_url = os.environ["TEST_PG_SEARCH_DATABASE_URL"]
    engine = create_async_engine(database_url)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    unique = uuid.uuid4()
    installation_id = int(unique.hex[:12], 16)
    repository: Repository | None = None

    try:
        async with factory() as session:
            installation = GitHubInstallation(
                id=installation_id,
                account_id=installation_id,
                account_login=f"hybrid-{unique.hex}",
                account_type="User",
                repository_selection="selected",
            )
            session.add(installation)
            await session.flush()
            repository = Repository(
                github_repository_id=installation_id,
                installation_id=installation.id,
                owner="showcasehost",
                name=f"hybrid-{unique.hex}",
                full_name=f"showcasehost/hybrid-{unique.hex}",
                private=True,
                default_branch="main",
            )
            session.add(repository)
            await session.flush()
            snapshot = SourceSnapshot(
                repository_id=repository.id,
                commit_sha=unique.hex.ljust(40, "0"),
                archive_key="archive",
                manifest_key="manifest",
                file_count=3,
                total_bytes=300,
            )
            session.add(snapshot)
            await session.flush()

            chunks: dict[str, CodeChunk] = {}
            fixtures = [
                (
                    "metadata",
                    "src/telemetrybeacon/collector.py",
                    "def collect():\n    return None\n",
                    '{"path":"src/telemetrybeacon/collector.py","language":"python",'
                    '"context":{"entities":["TelemetryBeacon"]}}',
                    100,
                ),
                (
                    "raw",
                    "src/worker.py",
                    "def emit():\n    return telemetrybeacon\n",
                    '{"path":"src/worker.py","language":"python","context":{}}',
                    100,
                ),
                (
                    "dense",
                    "src/dense.py",
                    "def semantic_only():\n    return None\n",
                    '{"path":"src/dense.py","language":"python","context":{}}',
                    75,
                ),
            ]
            for index, (name, path, raw_text, metadata, quality) in enumerate(fixtures):
                chunked_file = ChunkedFile(
                    repository_id=repository.id,
                    path=path,
                    content_sha256=f"{index + 1:064x}",
                    language="python",
                    fingerprint=f"{index + 11:064x}",
                    engine_version="e2e",
                    profile_key="hybrid-profile",
                    resolved_options={},
                )
                session.add(chunked_file)
                await session.flush()
                chunk = CodeChunk(
                    id=uuid.uuid4(),
                    chunked_file_id=chunked_file.id,
                    chunk_index=0,
                    fingerprint=f"{index + 21:064x}",
                    raw_text=raw_text,
                    searchable_metadata=metadata,
                    embedding_text=raw_text,
                    byte_start=0,
                    byte_end=len(raw_text.encode()),
                    start_line=1,
                    end_line=2,
                    context={},
                    tokenizer_id="e2e",
                    token_count=8,
                    quality_grade="A" if quality == 100 else "C",
                    quality_score=quality,
                )
                session.add(chunk)
                await session.flush()
                session.add(
                    SnapshotChunkFile(
                        snapshot_id=snapshot.id,
                        profile_key="hybrid-profile",
                        path=path,
                        chunked_file_id=chunked_file.id,
                        status=ChunkFileStatus.succeeded,
                    )
                )
                chunks[name] = chunk
            session.add(
                EmbeddingJob(
                    snapshot_id=snapshot.id,
                    profile_key="hybrid-profile",
                    selection_key="auto",
                    selected_model="BAAI/bge-m3",
                    dimension=1024,
                    status=EmbeddingStatus.succeeded,
                    finished_at=datetime.now(UTC),
                )
            )
            await session.commit()

        keyword_search = PgSearchKeywordSearch(metadata_boost=2.0)
        async with factory() as session:
            keyword_results = await keyword_search.search(
                session,
                "telemetrybeacon",
                32,
                repository_id=repository.id,
                snapshot_id=snapshot.id,
                profile_key="hybrid-profile",
            )
        assert [result["id"] for result in keyword_results[:2]] == [
            str(chunks["metadata"].id),
            str(chunks["raw"].id),
        ]

        vector_store = MockVectorStore()
        vector_store.records = [
            {
                "id": str(chunks["dense"].id),
                "score": 0.99,
                "payload": {
                    "repository_id": str(repository.id),
                    "snapshot_id": str(snapshot.id),
                    "path": "src/dense.py",
                    "raw_text": chunks["dense"].raw_text,
                    "quality_score": 75,
                },
            },
            {
                "id": str(chunks["metadata"].id),
                "score": 0.90,
                "payload": {
                    "repository_id": str(repository.id),
                    "snapshot_id": str(snapshot.id),
                },
            },
        ]
        settings = Settings(app_env="test")
        async with factory() as session:
            prepared = await prepare_repository_chat(
                session,
                repository,
                "telemetrybeacon",
                [],
                settings,
                lambda model: MockEmbeddingProvider(dimension=model.dimension),
                lambda model_id: MockRerankerProvider(model_id, [1.0] * 32),
                vector_store,
                keyword_search,
            )

        assert len(prepared.sources) <= 8
        assert prepared.sources[0].chunk_id == str(chunks["metadata"].id)
        assert len({source.chunk_id for source in prepared.sources}) == len(prepared.sources)
    finally:
        if repository is not None:
            async with factory() as session:
                await session.execute(delete(Repository).where(Repository.id == repository.id))
                await session.execute(
                    delete(GitHubInstallation).where(GitHubInstallation.id == installation_id)
                )
                await session.commit()
        await engine.dispose()
