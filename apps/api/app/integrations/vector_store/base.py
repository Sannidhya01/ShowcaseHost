from typing import Protocol

from app.integrations.keyword_search.base import SearchRecord


class VectorRecord(SearchRecord, total=False):
    vector: list[float]


class VectorStore(Protocol):
    async def prepare(self, model_id: str, dimension: int) -> None:
        """Create or validate model-specific vector storage."""

    async def existing_hashes(
        self, record_ids: list[str], model_id: str, dimension: int
    ) -> dict[str, str]:
        """Return stored content hashes for known record IDs."""

    async def upsert(self, records: list[VectorRecord], model_id: str, dimension: int) -> None:
        """Insert or update vector records."""

    async def search(
        self,
        query_vector: list[float],
        limit: int,
        model_id: str,
        dimension: int,
        repository_id: str | None = None,
        snapshot_id: str | None = None,
    ) -> list[VectorRecord]:
        """Return cosine-scored records, optionally scoped to a repository snapshot."""

    async def close(self) -> None:
        """Release vector-store resources."""
