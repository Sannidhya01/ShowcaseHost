from app.integrations.vector_store.base import VectorRecord, VectorStore


class MockVectorStore(VectorStore):
    def __init__(self) -> None:
        self.records: list[VectorRecord] = []

    async def prepare(self, model_id: str, dimension: int) -> None:
        return None

    async def existing_hashes(
        self, record_ids: list[str], model_id: str, dimension: int
    ) -> dict[str, str]:
        wanted = set(record_ids)
        return {
            str(record["id"]): str(record.get("payload", {}).get("content_hash"))
            for record in self.records
            if str(record.get("id")) in wanted
            and record.get("payload", {}).get("model_id") == model_id
        }

    async def upsert(
        self,
        records: list[VectorRecord],
        model_id: str = "mock/embedding",
        dimension: int = 3,
    ) -> None:
        replacements = {str(record["id"]): record for record in records}
        self.records = [
            record for record in self.records if str(record.get("id")) not in replacements
        ]
        self.records.extend(records)

    async def search(
        self,
        query_vector: list[float],
        limit: int,
        model_id: str = "mock/embedding",
        dimension: int = 3,
        repository_id: str | None = None,
        snapshot_id: str | None = None,
    ) -> list[VectorRecord]:
        return [
            record
            for record in self.records
            if (
                repository_id is None
                or record.get("payload", {}).get("repository_id") == repository_id
            )
            and (snapshot_id is None or record.get("payload", {}).get("snapshot_id") == snapshot_id)
        ][:limit]

    async def close(self) -> None:
        return None
