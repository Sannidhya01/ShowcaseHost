from __future__ import annotations

import io
import os
import tarfile
import uuid
from pathlib import Path
from typing import cast

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.chunking.client import ChunkingWorkerClient
from app.chunking.repository import list_embedding_inputs
from app.core.config import Settings
from app.db.base import Base
from app.db.models import (
    ChunkingJob,
    ChunkingStatus,
    EmbeddingJob,
    EmbeddingStatus,
    GitHubInstallation,
    IngestionJob,
    IngestionStatus,
    Repository,
    User,
)
from app.integrations.embeddings.catalog import MODEL_BY_ID, EmbeddingModel
from app.integrations.embeddings.huggingface import HuggingFaceEmbeddingProvider
from app.integrations.generation.catalog import GENERATION_MODELS
from app.integrations.generation.groq import GroqGenerationProvider
from app.integrations.keyword_search.mock import MockKeywordSearch
from app.integrations.reranking.mock import MockRerankerProvider
from app.integrations.snapshots.local import LocalSnapshotStore
from app.integrations.source_control.base import SourceControlProvider, SourceRepository
from app.integrations.vector_store.qdrant import QdrantVectorStore
from app.services.chat import prepare_repository_chat
from app.services.chunking import ChunkingProcessor, ChunkingRunner
from app.services.embeddings import EmbeddingProcessor, EmbeddingRunner
from app.services.ingestion import IngestionProcessor, IngestionRunner, enqueue_ingestion

pytestmark = [
    pytest.mark.live_e2e,
    pytest.mark.skipif(
        os.getenv("RUN_LIVE_E2E") != "1",
        reason="set RUN_LIVE_E2E=1 to exercise Hugging Face and Qdrant",
    ),
]
WORKER_PATH = Path(__file__).resolve().parents[4] / "chunker" / "dist" / "index.js"


class E2ESourceControl:
    async def create_installation_token(self, installation_id: int, repository_id: int) -> str:
        return "e2e-short-lived-token"

    async def get_repository(self, token: str, repository_id: int) -> SourceRepository:
        return SourceRepository(
            id=repository_id,
            installation_id=1,
            owner="showcasehost-e2e",
            name="embedding-pipeline",
            full_name="showcasehost-e2e/embedding-pipeline",
            private=True,
            default_branch="main",
        )

    async def resolve_default_branch_sha(self, token: str, repository: SourceRepository) -> str:
        return "e" * 40

    async def download_archive(
        self,
        token: str,
        repository: SourceRepository,
        commit_sha: str,
        destination: Path,
        max_bytes: int,
    ) -> int:
        files = {
            "src/calculator.py": (
                b"def add(left: int, right: int) -> int:\n"
                b'    """Return the sum of two integers."""\n'
                b"    return left + right\n"
            ),
            "src/format.ts": (
                b"export function formatTotal(value: number): string {\n"
                b"  return `Total: ${value}`;\n"
                b"}\n"
            ),
        }
        with tarfile.open(destination, "w:gz") as archive:
            for path, content in files.items():
                info = tarfile.TarInfo(f"repository-{commit_sha}/{path}")
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))
        return sum(len(content) for content in files.values())

    async def revoke_installation_token(self, token: str) -> None:
        return None


async def seed_repository(
    factory: async_sessionmaker[AsyncSession],
) -> tuple[uuid.UUID, uuid.UUID]:
    async with factory() as session:
        user = User(github_user_id=991_001, github_login="e2e-user")
        installation = GitHubInstallation(
            id=991_002,
            account_id=991_001,
            account_login="e2e-user",
            account_type="User",
            repository_selection="selected",
        )
        session.add_all([user, installation])
        await session.flush()
        repository = Repository(
            github_repository_id=991_003,
            installation_id=installation.id,
            owner="showcasehost-e2e",
            name="embedding-pipeline",
            full_name="showcasehost-e2e/embedding-pipeline",
            private=True,
            default_branch="main",
        )
        session.add(repository)
        await session.commit()
        return user.id, repository.id


async def test_ingestion_chunking_live_embedding_and_qdrant_storage(tmp_path: Path) -> None:
    configured = Settings()
    if configured.hf_api_token is None:
        pytest.fail("HF_API_TOKEN is required for the live E2E test")
    if configured.groq_api_key is None:
        pytest.fail("GROQ_API_KEY is required for the live E2E test")

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'metadata.db'}")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    collection_prefix = f"showcasehost-e2e-{uuid.uuid4().hex}"
    settings = Settings(
        app_env="test",
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'metadata.db'}",
        snapshot_root=tmp_path / "snapshots",
        ingestion_runner_enabled=False,
        chunking_runner_enabled=False,
        embedding_runner_enabled=False,
        embedding_batch_size=2,
        embedding_large_repository_chunks=1,
        qdrant_collection=collection_prefix,
    )
    snapshot_store = LocalSnapshotStore(settings)
    source_control = cast(SourceControlProvider, E2ESourceControl())
    vector_store = QdrantVectorStore(
        settings.qdrant_url, settings.qdrant_api_key, settings.qdrant_collection
    )
    model = MODEL_BY_ID["BAAI/bge-m3"]
    token = configured.hf_api_token.get_secret_value()

    def provider_factory(selected: EmbeddingModel) -> HuggingFaceEmbeddingProvider:
        return HuggingFaceEmbeddingProvider(
            token,
            selected.id,
            selected.dimension,
            base_url=settings.hf_inference_url,
            timeout_seconds=settings.embedding_request_timeout_seconds,
        )

    worker = ChunkingWorkerClient(settings.chunking_worker_command, WORKER_PATH)
    try:
        user_id, repository_id = await seed_repository(factory)
        ingestion_processor = IngestionProcessor(factory, settings, source_control, snapshot_store)
        ingestion_runner = IngestionRunner(factory, settings, ingestion_processor)
        async with factory() as session:
            ingestion_job = await enqueue_ingestion(session, repository_id, user_id)
        assert await ingestion_runner.claim_next() == ingestion_job.id
        await ingestion_processor.process(ingestion_job.id)

        async with factory() as session:
            ingested = await session.get(IngestionJob, ingestion_job.id)
            assert ingested is not None
            assert ingested.status == IngestionStatus.succeeded
            assert ingested.snapshot_id is not None

        chunking_processor = ChunkingProcessor(factory, settings, snapshot_store, worker)
        chunking_runner = ChunkingRunner(factory, settings, chunking_processor)
        chunking_job_id = await chunking_runner.claim_next()
        assert chunking_job_id is not None
        await chunking_processor.process(chunking_job_id)

        async with factory() as session:
            chunked = await session.get(ChunkingJob, chunking_job_id)
            embedding_job = await session.scalar(select(EmbeddingJob))
            assert chunked is not None
            assert chunked.status == ChunkingStatus.succeeded
            assert chunked.chunks_created >= 2
            assert embedding_job is not None

        embedding_processor = EmbeddingProcessor(factory, settings, provider_factory, vector_store)
        embedding_runner = EmbeddingRunner(factory, settings, embedding_processor)
        assert await embedding_runner.claim_next() == embedding_job.id
        await embedding_processor.process(embedding_job.id)

        async with factory() as session:
            embedded = await session.get(EmbeddingJob, embedding_job.id)
            assert embedded is not None
            assert embedded.status == EmbeddingStatus.succeeded
            assert embedded.selected_model == model.id
            assert embedded.dimension == model.dimension
            assert embedded.stored_chunks == embedded.total_chunks
            inputs = await list_embedding_inputs(
                session, embedded.snapshot_id, embedded.profile_key
            )

        hashes = await vector_store.existing_hashes(
            [str(item.chunk_id) for item in inputs], model.id, model.dimension
        )
        assert len(hashes) == len(inputs)
        assert all(len(content_hash) == 64 for content_hash in hashes.values())

        query_provider = provider_factory(model)
        try:
            query_vector = await query_provider.embed_query("function that adds two integers")
        finally:
            await query_provider.close()
        results = await vector_store.search(query_vector, len(inputs), model.id, model.dimension)
        assert results
        payload = results[0]["payload"]
        assert payload["model_id"] == model.id
        assert payload["dimension"] == model.dimension
        assert payload["content_hash"] in hashes.values()
        assert payload["text"]

        async with factory() as session:
            repository = await session.get(Repository, repository_id)
            assert repository is not None
            prepared = await prepare_repository_chat(
                session,
                repository,
                "Which source defines the function that adds two integers?",
                [],
                settings,
                provider_factory,
                lambda model_id: MockRerankerProvider(model_id, [1.0] * 32),
                vector_store,
                MockKeywordSearch(),
            )
        assert prepared.sources
        assert prepared.sources[0].path == "src/calculator.py"
        assert "src/calculator.py" in prepared.messages[0]["content"]

        groq_key = configured.groq_api_key.get_secret_value()
        for generation_model in GENERATION_MODELS:
            generation = GroqGenerationProvider(
                groq_key,
                generation_model.id,
                base_url=settings.groq_base_url,
                timeout_seconds=settings.generation_request_timeout_seconds,
                max_retries=settings.generation_max_retries,
                max_completion_tokens=300,
            )
            try:
                answer = "".join([part async for part in generation.stream(prepared.messages)])
            finally:
                await generation.close()
            assert answer.strip()
            assert "[1]" in answer
    finally:
        await worker.stop()
        collection_name = vector_store._collection_name(model.id, model.dimension)
        await vector_store.client.delete(
            f"{vector_store.url}/collections/{collection_name}",
            headers=vector_store._headers(),
        )
        await vector_store.close()
        await engine.dispose()
