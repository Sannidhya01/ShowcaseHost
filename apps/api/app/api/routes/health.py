from typing import Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db_session

router = APIRouter(tags=["system"])


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class ReadinessResponse(BaseModel):
    status: Literal["ready", "degraded"]
    checks: dict[str, Literal["ok", "skipped", "unavailable"]]


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", service="showcasehost-api", version="0.1.0")


@router.get("/ready", response_model=ReadinessResponse)
async def ready(session: AsyncSession = Depends(get_db_session)) -> ReadinessResponse:
    checks: dict[str, Literal["ok", "skipped", "unavailable"]] = {
        "database": "unavailable",
        "embedding_provider": "skipped",
        "generation_provider": "skipped",
        "vector_store": "skipped",
    }

    try:
        await session.execute(text("select 1"))
        checks["database"] = "ok"
    except Exception:
        checks["database"] = "unavailable"

    status: Literal["ready", "degraded"] = "ready" if checks["database"] == "ok" else "degraded"
    return ReadinessResponse(status=status, checks=checks)
