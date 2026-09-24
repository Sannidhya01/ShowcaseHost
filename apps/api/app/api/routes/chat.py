from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncIterator
from typing import Literal, cast

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies.auth import get_current_user
from app.db.models import User
from app.db.session import get_db_session
from app.integrations.generation.base import (
    ChatMessage,
    GenerationProvider,
    GenerationProviderError,
)
from app.integrations.generation.catalog import (
    GENERATION_MODELS,
    UnsupportedGenerationModel,
    require_generation_model,
)
from app.services.chat import (
    NO_RELEVANT_INFORMATION,
    ChatPreparationError,
    EmbeddingProviderFactory,
    GenerationProviderFactory,
    RerankerProviderFactory,
    prepare_repository_chat,
)
from app.services.repositories import get_user_repository

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])


class GenerationModelResponse(BaseModel):
    id: str
    label: str
    tier: str
    provider: str
    is_default: bool


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=20_000)


class ChatRequest(BaseModel):
    model: str
    message: str = Field(min_length=1)
    history: list[HistoryMessage] = Field(default_factory=list, max_length=100)


def sse(event: str, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"


@router.get("/generation-models", response_model=list[GenerationModelResponse])
async def generation_models(
    user: User = Depends(get_current_user),
) -> list[GenerationModelResponse]:
    return [GenerationModelResponse(**model.__dict__) for model in GENERATION_MODELS]


@router.post("/repositories/{repository_id}/chat")
async def repository_chat(
    repository_id: uuid.UUID,
    body: ChatRequest,
    request: Request,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> StreamingResponse:
    settings = request.app.state.settings
    question = body.message.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Message must not be blank")
    if len(question) > settings.chat_question_max_chars:
        raise HTTPException(
            status_code=422,
            detail=f"Message must be at most {settings.chat_question_max_chars} characters",
        )
    try:
        require_generation_model(body.model)
    except UnsupportedGenerationModel as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    repository = await get_user_repository(session, user.id, repository_id)
    if repository is None:
        raise HTTPException(status_code=404, detail="Repository not found")
    history = cast(
        list[ChatMessage],
        [{"role": item.role, "content": item.content} for item in body.history],
    )
    embedding_factory = cast(EmbeddingProviderFactory, request.app.state.embedding_provider_factory)
    reranker_factory = cast(RerankerProviderFactory, request.app.state.reranker_provider_factory)
    try:
        prepared = await prepare_repository_chat(
            session,
            repository,
            question,
            history,
            settings,
            embedding_factory,
            reranker_factory,
            request.app.state.vector_store,
            request.app.state.keyword_search,
        )
    except ChatPreparationError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception(
            "Repository chat retrieval failed",
            extra={"repository_id": str(repository_id)},
        )
        raise HTTPException(
            status_code=502, detail="Repository retrieval is temporarily unavailable"
        ) from exc

    generation_factory = cast(
        GenerationProviderFactory, request.app.state.generation_provider_factory
    )

    async def events() -> AsyncIterator[str]:
        yield sse("sources", [source.public_dict() for source in prepared.sources])
        if not prepared.sources:
            yield sse("delta", {"content": NO_RELEVANT_INFORMATION})
            yield sse("done", {})
            return
        provider: GenerationProvider = generation_factory(body.model)
        try:
            async for delta in provider.stream(prepared.messages):
                yield sse("delta", {"content": delta})
            yield sse("done", {})
        except GenerationProviderError:
            logger.exception(
                "Repository chat generation failed",
                extra={"repository_id": str(repository_id), "model": body.model},
            )
            yield sse(
                "error",
                {
                    "code": "generation_unavailable",
                    "message": "Answer generation is temporarily unavailable",
                },
            )
        finally:
            await provider.close()

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
