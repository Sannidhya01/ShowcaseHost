from __future__ import annotations

import uuid
from dataclasses import dataclass

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.chunking.repository import EmbeddingInput
from app.core.config import Settings
from app.db.base import Base
from app.db.models import EmbeddingJob, EmbeddingStatus
from app.integrations.embeddings.base import EmbeddingProviderError
from app.integrations.embeddings.catalog import MODEL_BY_ID, EmbeddingModel, model_candidates
from app.integrations.vector_store.mock import MockVectorStore
from app.services.embeddings import EmbeddingProcessor, chunk_content_hash, vector_record


@dataclass(frozen=True)
class SelectionFixture:
    language: str
    token_count: int


def embedding_input(*, text: str = "def hello():\n    return 'world'\n") -> EmbeddingInput:
    return EmbeddingInput(
        chunk_id=uuid.uuid4(),
        text=text,
        raw_text=text,
        repository_id=uuid.uuid4(),
        snapshot_id=uuid.uuid4(),
        path="src/hello.py",
        language="python",
        start_line=1,
        end_line=2,
        byte_start=0,
        byte_end=len(text.encode()),
        entities=[{"name": "hello", "type": "function"}],
        scope=["module"],
        chunk_fingerprint="f" * 64,
        profile_key="profile-v1",
        tokenizer_id="utf8-byte-v1",
        token_count=len(text.encode()),
        quality_grade="A",
        quality_score=100,
    )


def test_auto_selection_prefers_long_context_for_code_and_fast_model_for_short_prose() -> None:
    code = [SelectionFixture("python", 900) for _ in range(10)]
    prose = [SelectionFixture("markdown", 120) for _ in range(10)]

    assert model_candidates(code, large_repository_chunks=5)[0].id == "BAAI/bge-m3"
    assert (
        model_candidates(prose, large_repository_chunks=5)[0].id
        == "sentence-transformers/all-MiniLM-L6-v2"
    )


def test_manual_selection_is_first_but_retains_defined_fallback_order() -> None:
    candidates = model_candidates([], manual_model="intfloat/multilingual-e5-small")

    assert [candidate.id for candidate in candidates] == [
        "intfloat/multilingual-e5-small",
        "BAAI/bge-m3",
        "sentence-transformers/all-MiniLM-L6-v2",
    ]


def test_vector_record_contains_text_model_dimension_hash_and_chunk_metadata() -> None:
    item = embedding_input()
    model = MODEL_BY_ID["BAAI/bge-m3"]
    record = vector_record(item, [0.0] * model.dimension, model)

    assert record["vector"] == [0.0] * model.dimension
    assert record["payload"]["text"] == item.text
    assert record["payload"]["entities"] == item.entities
    assert record["payload"]["context"] == item.context
    assert record["payload"]["model_id"] == model.id
    assert record["payload"]["dimension"] == model.dimension
    assert record["payload"]["content_hash"] == chunk_content_hash(item)
    assert record["payload"]["quality_grade"] == "A"
    assert record["payload"]["quality_score"] == 100


class FailingProvider:
    def __init__(self, model_id: str, dimension: int) -> None:
        self.model_id = model_id
        self.dimension = dimension

    async def validate(self) -> None:
        raise EmbeddingProviderError("model unavailable", retryable=False)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise AssertionError("failed model must not embed documents")

    async def embed_query(self, text: str) -> list[float]:
        raise AssertionError("failed model must not embed queries")

    async def close(self) -> None:
        return None


class WorkingProvider:
    def __init__(self, model_id: str, dimension: int) -> None:
        self.model_id = model_id
        self.dimension = dimension
        self.embedded = 0

    async def validate(self) -> None:
        return None

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.embedded += len(texts)
        return [[float(len(text)), *([0.0] * (self.dimension - 1))] for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_documents([text]))[0]

    async def close(self) -> None:
        return None


async def test_provider_failure_falls_back_and_reports_selected_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    item = embedding_input()
    job = EmbeddingJob(
        snapshot_id=item.snapshot_id,
        profile_key=item.profile_key,
        selection_key="auto",
        status=EmbeddingStatus.running,
        attempt_count=1,
    )
    async with factory() as session:
        session.add(job)
        await session.commit()
        job_id = job.id

    async def fake_inputs(
        session: AsyncSession, snapshot_id: uuid.UUID, profile_key: str
    ) -> list[EmbeddingInput]:
        return [item]

    monkeypatch.setattr("app.services.embeddings.list_embedding_inputs", fake_inputs)
    providers: dict[str, WorkingProvider] = {}

    def provider_factory(model: EmbeddingModel) -> FailingProvider | WorkingProvider:
        model_id = str(model.id)
        dimension = int(model.dimension)
        if model_id == "BAAI/bge-m3":
            return FailingProvider(model_id, dimension)
        provider = WorkingProvider(model_id, dimension)
        providers[model_id] = provider
        return provider

    store = MockVectorStore()
    processor = EmbeddingProcessor(
        factory,
        Settings(embedding_max_retries=0, embedding_large_repository_chunks=100),
        provider_factory,
        store,
    )
    await processor.process(job_id)

    async with factory() as session:
        completed = await session.get(EmbeddingJob, job_id)
        assert completed is not None
        assert completed.status == EmbeddingStatus.succeeded
        assert completed.selected_model == "sentence-transformers/all-MiniLM-L6-v2"
        assert completed.fallback_count == 1
    assert len(store.records) == 1
    await engine.dispose()


async def test_incremental_embedding_skips_unchanged_chunks() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    item = embedding_input()
    model = MODEL_BY_ID["BAAI/bge-m3"]
    provider = WorkingProvider(model.id, model.dimension)
    store = MockVectorStore()
    processor = EmbeddingProcessor(
        factory,
        Settings(embedding_batch_size=8, embedding_large_repository_chunks=100),
        lambda selected: provider,
        store,
    )

    assert await processor._embed_and_store(uuid.uuid4(), [item], model, provider) == (1, 0)
    assert await processor._embed_and_store(uuid.uuid4(), [item], model, provider) == (0, 1)
    assert provider.embedded == 1
    assert len(store.records) == 1
    await engine.dispose()
