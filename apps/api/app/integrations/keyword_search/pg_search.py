from __future__ import annotations

import uuid
from typing import cast

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.integrations.keyword_search.base import SearchRecord


class PgSearchKeywordSearch:
    def __init__(self, metadata_boost: float = 2.0) -> None:
        self.metadata_boost = metadata_boost

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
        if not query.strip() or limit <= 0:
            return []
        rows = await session.execute(
            text(
                """
                SELECT
                    code_chunks.id,
                    code_chunks.raw_text,
                    code_chunks.context,
                    code_chunks.start_line,
                    code_chunks.end_line,
                    code_chunks.byte_start,
                    code_chunks.byte_end,
                    code_chunks.quality_grade,
                    code_chunks.quality_score,
                    chunked_files.path,
                    chunked_files.language,
                    pdb.score(code_chunks.id) AS bm25_score
                FROM code_chunks
                JOIN chunked_files
                  ON chunked_files.id = code_chunks.chunked_file_id
                JOIN snapshot_chunk_files
                  ON snapshot_chunk_files.chunked_file_id = chunked_files.id
                WHERE chunked_files.repository_id = CAST(:repository_id AS uuid)
                  AND snapshot_chunk_files.snapshot_id = CAST(:snapshot_id AS uuid)
                  AND snapshot_chunk_files.profile_key = :profile_key
                  AND code_chunks.id @@@ paradedb.boolean(
                      should => ARRAY[
                          paradedb.boost(
                              :metadata_boost,
                              paradedb.match('searchable_metadata', :query)
                          ),
                          paradedb.match('raw_text', :query)
                      ]
                  )
                ORDER BY bm25_score DESC, code_chunks.quality_score DESC, code_chunks.id
                LIMIT :limit
                """
            ),
            {
                "repository_id": str(repository_id),
                "snapshot_id": str(snapshot_id),
                "profile_key": profile_key,
                "metadata_boost": self.metadata_boost,
                "query": query,
                "limit": limit,
            },
        )
        return [
            SearchRecord(
                id=str(row.id),
                score=float(row.bm25_score),
                payload={
                    "raw_text": str(row.raw_text),
                    "repository_id": str(repository_id),
                    "snapshot_id": str(snapshot_id),
                    "path": str(row.path),
                    "language": str(row.language),
                    "start_line": int(row.start_line),
                    "end_line": int(row.end_line),
                    "byte_start": int(row.byte_start),
                    "byte_end": int(row.byte_end),
                    "context": cast(dict[str, object], row.context),
                    "quality_grade": str(row.quality_grade),
                    "quality_score": int(row.quality_score),
                },
            )
            for row in rows
        ]
