from __future__ import annotations

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.integrations.keyword_search.base import SearchRecord


class MockKeywordSearch:
    def __init__(self, records: list[SearchRecord] | None = None) -> None:
        self.records = records or []
        self.last_search: tuple[str, int, uuid.UUID, uuid.UUID, str] | None = None

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
        self.last_search = (query, limit, repository_id, snapshot_id, profile_key)
        return self.records[:limit]
