from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import Settings
from app.db.base import Base
from app.db.models import (
    EmbeddingJob,
    EmbeddingStatus,
    GitHubInstallation,
    Repository,
    Session,
    SourceSnapshot,
    User,
    UserRepository,
)
from app.db.session import get_db_session
from app.integrations.embeddings.mock import MockEmbeddingProvider
from app.integrations.generation.base import ChatMessage
from app.integrations.keyword_search.mock import MockKeywordSearch
from app.integrations.reranking.mock import MockRerankerProvider
from app.integrations.vector_store.base import VectorRecord
from app.integrations.vector_store.mock import MockVectorStore
from app.main import create_app
from app.services.auth import token_hash


class CitedGenerationProvider:
    def __init__(self) -> None:
        self.messages: list[ChatMessage] = []
        self.closed = False

    async def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        self.messages = messages
        yield "The entrypoint is `main` "
        yield "[1]."

    async def close(self) -> None:
        self.closed = True


class CapturingVectorStore(MockVectorStore):
    def __init__(self) -> None:
        super().__init__()
        self.filters: tuple[str | None, str | None] | None = None
        self.limit: int | None = None

    async def search(
        self,
        query_vector: list[float],
        limit: int,
        model_id: str = "mock/embedding",
        dimension: int = 3,
        repository_id: str | None = None,
        snapshot_id: str | None = None,
    ) -> list[VectorRecord]:
        self.filters = (repository_id, snapshot_id)
        self.limit = limit
        return await super().search(
            query_vector, limit, model_id, dimension, repository_id, snapshot_id
        )


@pytest.fixture
async def database(tmp_path: Path) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    url = os.getenv("TEST_DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'chat.db'}")
    engine = create_async_engine(url)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    await engine.dispose()


async def seed(factory: async_sessionmaker[AsyncSession]) -> tuple[str, Repository, SourceSnapshot]:
    raw_session = "chat-session"
    async with factory() as session:
        user = User(github_user_id=100, github_login="octocat")
        installation = GitHubInstallation(
            id=200,
            account_id=100,
            account_login="octocat",
            account_type="User",
            repository_selection="selected",
        )
        session.add_all([user, installation])
        await session.flush()
        repository = Repository(
            github_repository_id=300,
            installation_id=installation.id,
            owner="octocat",
            name="hello",
            full_name="octocat/hello",
            private=True,
            default_branch="main",
        )
        session.add(repository)
        await session.flush()
        snapshot = SourceSnapshot(
            repository_id=repository.id,
            commit_sha="a" * 40,
            archive_key="archive",
            manifest_key="manifest",
            file_count=1,
            total_bytes=20,
        )
        session.add(snapshot)
        await session.flush()
        session.add_all(
            [
                UserRepository(user_id=user.id, repository_id=repository.id),
                Session(
                    user_id=user.id,
                    token_hash=token_hash(raw_session),
                    expires_at=datetime.now(UTC) + timedelta(hours=1),
                ),
                EmbeddingJob(
                    snapshot_id=snapshot.id,
                    profile_key="profile",
                    selection_key="auto",
                    selected_model="BAAI/bge-m3",
                    dimension=1024,
                    status=EmbeddingStatus.succeeded,
                    finished_at=datetime.now(UTC),
                ),
            ]
        )
        await session.commit()
        return raw_session, repository, snapshot


async def test_chat_route_streams_filtered_grounded_answer(
    database: async_sessionmaker[AsyncSession], tmp_path: Path
) -> None:
    raw_session, repository, snapshot = await seed(database)
    app = create_app(
        Settings(
            app_env="test",
            chat_relevance_threshold=0.3893,
            chat_relative_relevance_threshold=0,
            ingestion_runner_enabled=False,
            chunking_runner_enabled=False,
            embedding_runner_enabled=False,
            snapshot_root=tmp_path / "snapshots",
        )
    )

    async def override_session() -> AsyncIterator[AsyncSession]:
        async with database() as session:
            yield session

    generation = CitedGenerationProvider()
    vectors = CapturingVectorStore()
    vectors.records = [
        {
            "id": "chunk-1",
            "score": 0.8,
            "vector": [1.0, 0.0, 1.0],
            "payload": {
                "repository_id": str(repository.id),
                "snapshot_id": str(snapshot.id),
                "path": "src/main.py",
                "language": "python",
                "start_line": 5,
                "end_line": 9,
                "raw_text": "def main(): pass",
            },
        }
    ]
    app.dependency_overrides[get_db_session] = override_session
    app.state.embedding_provider_factory = lambda model: MockEmbeddingProvider(
        dimension=model.dimension
    )
    app.state.reranker_provider_factory = lambda model_id: MockRerankerProvider(
        model_id, [0.9] * 32
    )
    app.state.generation_provider_factory = lambda model_id: generation
    app.state.vector_store = vectors
    keyword = MockKeywordSearch(
        [
            {
                "id": "chunk-1",
                "score": 4.0,
                "payload": {
                    "path": "src/main.py",
                    "language": "python",
                    "start_line": 5,
                    "end_line": 9,
                    "raw_text": "def main(): pass",
                },
            }
        ]
    )
    app.state.keyword_search = keyword
    cookies = {app.state.settings.session_cookie_name: raw_session}

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test", cookies=cookies
    ) as client:
        models = await client.get("/generation-models")
        response = await client.post(
            f"/repositories/{repository.id}/chat",
            json={
                "model": "qwen/qwen3.8-27b",
                "message": "Where is the entrypoint?",
                "history": [],
            },
        )

    assert next(model for model in models.json() if model["is_default"])["id"] == (
        "openai/gpt-oss-20b"
    )
    assert response.status_code == 200
    assert "event: sources" in response.text
    assert '"path":"src/main.py"' in response.text
    assert "The entrypoint is `main`" in response.text
    assert vectors.filters == (str(repository.id), str(snapshot.id))
    assert vectors.limit == 32
    assert keyword.last_search == (
        "Where is the entrypoint?",
        32,
        repository.id,
        snapshot.id,
        "profile",
    )
    assert "def main(): pass" in generation.messages[0]["content"]
    assert generation.closed is True


async def test_chat_route_rejects_auth_model_and_unready_repository(
    database: async_sessionmaker[AsyncSession], tmp_path: Path
) -> None:
    raw_session, repository, _ = await seed(database)
    app = create_app(
        Settings(
            app_env="test",
            ingestion_runner_enabled=False,
            chunking_runner_enabled=False,
            embedding_runner_enabled=False,
            snapshot_root=tmp_path / "snapshots",
        )
    )

    async def override_session() -> AsyncIterator[AsyncSession]:
        async with database() as session:
            yield session

    app.dependency_overrides[get_db_session] = override_session
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        unauthorized = await client.get("/generation-models")
    cookies = {app.state.settings.session_cookie_name: raw_session}
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test", cookies=cookies
    ) as client:
        unsupported = await client.post(
            f"/repositories/{repository.id}/chat",
            json={"model": "unknown", "message": "hello", "history": []},
        )
        missing = await client.post(
            f"/repositories/{uuid.uuid4()}/chat",
            json={"model": "qwen/qwen3.8-27b", "message": "hello", "history": []},
        )

    assert unauthorized.status_code == 401
    assert unsupported.status_code == 422
    assert missing.status_code == 404


@pytest.mark.parametrize("scores", [[], [0.3892, 0.2], [0.9, 0.3893, 0.3892]])
async def test_chat_stream_uses_only_qualifying_sources_and_abstains_when_empty(
    database: async_sessionmaker[AsyncSession], tmp_path: Path, scores: list[float]
) -> None:
    import json

    from app.services.chat import NO_RELEVANT_INFORMATION

    raw_session, repository, snapshot = await seed(database)
    app = create_app(
        Settings(
            app_env="test",
            chat_relevance_threshold=0.3893,
            chat_relative_relevance_threshold=0,
            ingestion_runner_enabled=False,
            chunking_runner_enabled=False,
            embedding_runner_enabled=False,
            snapshot_root=tmp_path / "snapshots",
        )
    )

    async def override_session() -> AsyncIterator[AsyncSession]:
        async with database() as session:
            yield session

    generation = CitedGenerationProvider()
    calls: list[str] = []

    def generation_factory(model_id: str) -> CitedGenerationProvider:
        calls.append(model_id)
        return generation

    vectors = CapturingVectorStore()
    vectors.records = [
        {
            "id": str(index),
            "score": 0.9,
            "payload": {
                "repository_id": str(repository.id),
                "snapshot_id": str(snapshot.id),
                "path": f"src/{index}.py",
                "raw_text": f"chunk_content_{index}",
            },
        }
        for index, score in enumerate(scores)
    ]
    app.dependency_overrides[get_db_session] = override_session
    app.state.embedding_provider_factory = lambda model: MockEmbeddingProvider(
        dimension=model.dimension
    )
    app.state.reranker_provider_factory = lambda model_id: MockRerankerProvider(model_id, scores)
    app.state.generation_provider_factory = generation_factory
    app.state.vector_store = vectors
    app.state.keyword_search = MockKeywordSearch(
        [
            {"id": str(index), "score": 1000.0, "payload": record["payload"]}
            for index, record in enumerate(vectors.records)
        ]
    )
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        cookies={app.state.settings.session_cookie_name: raw_session},
    ) as client:
        response = await client.post(
            f"/repositories/{repository.id}/chat",
            json={
                "model": "qwen/qwen3.8-27b",
                "message": "Where is the entrypoint?",
                "history": [
                    {"role": "user", "content": "Old question"},
                    {"role": "assistant", "content": "An unsupported old answer [1]."},
                ],
            },
        )
    assert response.status_code == 200
    events = [block.splitlines() for block in response.text.strip().split("\n\n")]
    sources = json.loads(events[0][1].removeprefix("data: "))
    expected_ids = {str(index) for index, score in enumerate(scores) if score >= 0.3893}
    assert {source["chunk_id"] for source in sources} == expected_ids
    assert [source["number"] for source in sources] == list(range(1, len(sources) + 1))
    assert events[-1][0] == "event: done"
    if expected_ids:
        assert len(calls) == 1
        assert generation.closed
        assert "chunk_content_2" not in generation.messages[0]["content"]
    else:
        assert calls == []
        assert json.loads(events[1][1].removeprefix("data: ")) == {
            "content": NO_RELEVANT_INFORMATION
        }
        assert "An unsupported old answer" not in response.text
