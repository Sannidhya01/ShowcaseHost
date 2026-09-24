from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies.auth import get_current_user
from app.db.models import (
    ChunkingJob,
    ChunkingStatus,
    EmbeddingJob,
    EmbeddingStatus,
    IngestionJob,
    IngestionStatus,
    Repository,
    SourceSnapshot,
    User,
    UserRepository,
)
from app.db.session import get_db_session
from app.integrations.embeddings.catalog import (
    EMBEDDING_MODELS,
    UnsupportedEmbeddingModel,
    require_supported_model,
)
from app.services.ingestion import enqueue_ingestion
from app.services.repositories import get_user_repository, list_user_repositories

router = APIRouter(tags=["repositories"])


class IngestionJobResponse(BaseModel):
    id: uuid.UUID
    repository_id: uuid.UUID
    status: IngestionStatus
    attempt_count: int
    commit_sha: str | None
    error_code: str | None
    error_message: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    chunking_job_id: uuid.UUID | None = None
    chunking_status: ChunkingStatus | None = None
    chunking_error_code: str | None = None
    chunking_error_message: str | None = None
    embedding_job_id: uuid.UUID | None = None
    embedding_status: EmbeddingStatus | None = None
    embedding_error_code: str | None = None
    embedding_error_message: str | None = None
    embedding_progress: float | None = None
    embedding_model: str | None = None


class RepositoryResponse(BaseModel):
    id: uuid.UUID
    github_repository_id: int
    full_name: str
    private: bool
    default_branch: str
    last_observed_sha: str | None
    chat_ready: bool = False
    latest_ingestion: IngestionJobResponse | None = None


class IngestionRequest(BaseModel):
    embedding_model: str | None = None


class EmbeddingModelResponse(BaseModel):
    id: str
    dimension: int
    context_tokens: int


class ChunkingJobResponse(BaseModel):
    id: uuid.UUID
    snapshot_id: uuid.UUID
    profile_key: str
    status: ChunkingStatus
    attempt_count: int
    discovered_files: int
    supported_files: int
    reused_files: int
    skipped_files: int
    failed_files: int
    chunks_created: int
    processed_bytes: int
    adaptive_retries: int
    has_warnings: bool
    language_timings_ms: dict[str, object]
    error_code: str | None
    error_message: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


async def job_response(job: IngestionJob, session: AsyncSession) -> IngestionJobResponse:
    response = IngestionJobResponse.model_validate(job, from_attributes=True)
    if job.snapshot_id is not None:
        chunking = await session.scalar(
            select(ChunkingJob)
            .where(ChunkingJob.snapshot_id == job.snapshot_id)
            .order_by(ChunkingJob.created_at.desc())
        )
        if chunking:
            response.chunking_job_id = chunking.id
            response.chunking_status = chunking.status
            response.chunking_error_code = chunking.error_code
            response.chunking_error_message = chunking.error_message
            embedding = await session.scalar(
                select(EmbeddingJob)
                .where(
                    EmbeddingJob.snapshot_id == job.snapshot_id,
                    EmbeddingJob.profile_key == chunking.profile_key,
                    EmbeddingJob.selection_key == (job.embedding_model_override or "auto"),
                )
                .order_by(EmbeddingJob.created_at.desc())
            )
            if embedding:
                response.embedding_job_id = embedding.id
                response.embedding_status = embedding.status
                response.embedding_error_code = embedding.error_code
                response.embedding_error_message = embedding.error_message
                response.embedding_model = embedding.selected_model or embedding.requested_model
                response.embedding_progress = (
                    embedding.processed_chunks / embedding.total_chunks
                    if embedding.total_chunks
                    else 0.0
                )
    return response


@router.get("/embedding-models", response_model=list[EmbeddingModelResponse])
async def embedding_models(user: User = Depends(get_current_user)) -> list[EmbeddingModelResponse]:
    return [
        EmbeddingModelResponse(
            id=model.id, dimension=model.dimension, context_tokens=model.context_tokens
        )
        for model in EMBEDDING_MODELS
    ]


@router.get("/repositories", response_model=list[RepositoryResponse])
async def repositories(
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> list[RepositoryResponse]:
    stored = await list_user_repositories(session, user.id)
    responses: list[RepositoryResponse] = []
    for item in stored:
        ready = await session.scalar(
            select(EmbeddingJob.id)
            .join(SourceSnapshot, SourceSnapshot.id == EmbeddingJob.snapshot_id)
            .where(
                SourceSnapshot.repository_id == item.id,
                EmbeddingJob.status == EmbeddingStatus.succeeded,
            )
            .limit(1)
        )
        response = RepositoryResponse.model_validate(item, from_attributes=True)
        response.chat_ready = ready is not None
        latest_ingestion = await session.scalar(
            select(IngestionJob)
            .where(IngestionJob.repository_id == item.id)
            .order_by(IngestionJob.created_at.desc())
        )
        if latest_ingestion is not None:
            response.latest_ingestion = await job_response(latest_ingestion, session)
        responses.append(response)
    return responses


@router.post(
    "/repositories/{repository_id}/ingestions",
    response_model=IngestionJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def start_ingestion(
    repository_id: uuid.UUID,
    request: IngestionRequest | None = None,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> IngestionJobResponse:
    repository = await get_user_repository(session, user.id, repository_id)
    if repository is None:
        raise HTTPException(status_code=404, detail="Repository not found")
    model_override = request.embedding_model if request else None
    if model_override:
        try:
            require_supported_model(model_override)
        except UnsupportedEmbeddingModel as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    job = await enqueue_ingestion(session, repository.id, user.id, model_override)
    return await job_response(job, session)


@router.get("/ingestions/{job_id}", response_model=IngestionJobResponse)
async def get_ingestion(
    job_id: uuid.UUID,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> IngestionJobResponse:
    job = await session.scalar(
        select(IngestionJob)
        .join(Repository, Repository.id == IngestionJob.repository_id)
        .join(UserRepository, UserRepository.repository_id == Repository.id)
        .where(IngestionJob.id == job_id, UserRepository.user_id == user.id)
    )
    if job is None:
        raise HTTPException(status_code=404, detail="Ingestion job not found")
    return await job_response(job, session)


@router.get("/chunking-jobs/{job_id}", response_model=ChunkingJobResponse)
async def get_chunking_job(
    job_id: uuid.UUID,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> ChunkingJobResponse:
    job = await session.scalar(
        select(ChunkingJob)
        .join(SourceSnapshot, SourceSnapshot.id == ChunkingJob.snapshot_id)
        .join(Repository, Repository.id == SourceSnapshot.repository_id)
        .join(UserRepository, UserRepository.repository_id == Repository.id)
        .where(ChunkingJob.id == job_id, UserRepository.user_id == user.id)
    )
    if job is None:
        raise HTTPException(status_code=404, detail="Chunking job not found")
    return ChunkingJobResponse.model_validate(job, from_attributes=True)
