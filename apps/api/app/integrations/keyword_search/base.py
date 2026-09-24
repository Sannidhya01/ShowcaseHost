from __future__ import annotations

import uuid
from typing import Protocol, TypedDict

from sqlalchemy.ext.asyncio import AsyncSession


class SearchRecord(TypedDict, total=False):
    id: str
    score: float
    payload: dict[str, object]


class KeywordSearch(Protocol):
    async def search(
        self,
        session: AsyncSession,
        query: str,
        limit: int,
        *,
        repository_id: uuid.UUID,
        snapshot_id: uuid.UUID,
        profile_key: str,
    ) -> list[SearchRecord]:
        """Return BM25-ranked chunks scoped to one repository snapshot and profile."""
