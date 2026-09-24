from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import ChunkedFile, CodeChunk, SnapshotChunkFile
from app.integrations.embeddings.base import EmbeddingProvider


@dataclass(frozen=True)
class EmbeddingInput:
    chunk_id: uuid.UUID
    text: str
    raw_text: str
    repository_id: uuid.UUID
    snapshot_id: uuid.UUID
    path: str
    language: str
    start_line: int
    end_line: int
    byte_start: int
    byte_end: int
    entities: list[object]
    scope: list[object]
    chunk_fingerprint: str
    profile_key: str
    tokenizer_id: str
    token_count: int
    quality_grade: str
    quality_score: int
    context: dict[str, object] = field(default_factory=dict)


async def list_embedding_inputs(
    session: AsyncSession, snapshot_id: uuid.UUID, profile_key: str
) -> list[EmbeddingInput]:
    rows = await session.execute(
        select(CodeChunk, ChunkedFile, SnapshotChunkFile)
        .join(ChunkedFile, ChunkedFile.id == CodeChunk.chunked_file_id)
        .join(SnapshotChunkFile, SnapshotChunkFile.chunked_file_id == ChunkedFile.id)
        .where(
            SnapshotChunkFile.snapshot_id == snapshot_id,
            SnapshotChunkFile.profile_key == profile_key,
        )
        .order_by(SnapshotChunkFile.path, CodeChunk.chunk_index)
    )
    return [
        EmbeddingInput(
            chunk_id=chunk.id,
            text=chunk.embedding_text,
            raw_text=chunk.raw_text,
            repository_id=file.repository_id,
            snapshot_id=membership.snapshot_id,
            path=membership.path,
            language=file.language,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            byte_start=chunk.byte_start,
            byte_end=chunk.byte_end,
            entities=list(chunk.context.get("entities", [])),
            scope=list(chunk.context.get("scope", [])),
            chunk_fingerprint=chunk.fingerprint,
            profile_key=file.profile_key,
            tokenizer_id=chunk.tokenizer_id,
            token_count=chunk.token_count,
            quality_grade=chunk.quality_grade,
            quality_score=chunk.quality_score,
            context=dict(chunk.context),
        )
        for chunk, file, membership in rows
    ]


async def embed_inputs(
    provider: EmbeddingProvider, inputs: list[EmbeddingInput]
) -> list[list[float]]:
    """Provider-neutral proof of the chunk-to-embedding handoff."""
    return await provider.embed_documents([item.text for item in inputs])
