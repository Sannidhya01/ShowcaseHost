from __future__ import annotations

import hashlib
import re
from typing import cast

import httpx

from app.integrations.vector_store.base import VectorRecord, VectorStore


class VectorStoreError(RuntimeError):
    pass


class QdrantVectorStore(VectorStore):
    def __init__(
        self,
        url: str | None,
        api_key: str | None,
        collection: str,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.url = (url or "http://localhost:6333").rstrip("/")
        self.api_key = api_key
        self.collection = collection
        self._owns_client = client is None
        self.client = client or httpx.AsyncClient(timeout=30.0)

    async def prepare(self, model_id: str, dimension: int) -> None:
        collection = self._collection_name(model_id, dimension)
        response = await self.client.get(self._url(collection), headers=self._headers())
        if response.status_code == 404:
            response = await self.client.put(
                self._url(collection),
                headers=self._headers(),
                json={"vectors": {"size": dimension, "distance": "Cosine"}},
            )
        self._raise(response, "prepare collection")
        if response.request.method == "GET":
            try:
                size = response.json()["result"]["config"]["params"]["vectors"]["size"]
            except (KeyError, TypeError, ValueError) as exc:
                raise VectorStoreError(
                    "Qdrant returned an invalid collection configuration"
                ) from exc
            if int(size) != dimension:
                raise VectorStoreError(
                    f"Qdrant collection {collection} has dimension {size}, expected {dimension}"
                )

    async def existing_hashes(
        self, record_ids: list[str], model_id: str, dimension: int
    ) -> dict[str, str]:
        if not record_ids:
            return {}
        response = await self.client.post(
            f"{self._url(self._collection_name(model_id, dimension))}/points",
            headers=self._headers(),
            json={"ids": record_ids, "with_payload": True, "with_vector": False},
        )
        self._raise(response, "retrieve points")
        points = cast(list[dict[str, object]], response.json().get("result", []))
        found: dict[str, str] = {}
        for point in points:
            payload = point.get("payload")
            if isinstance(payload, dict) and payload.get("model_id") == model_id:
                found[str(point["id"])] = str(payload.get("content_hash", ""))
        return found

    async def upsert(self, records: list[VectorRecord], model_id: str, dimension: int) -> None:
        if not records:
            return
        response = await self.client.put(
            f"{self._url(self._collection_name(model_id, dimension))}/points",
            headers=self._headers(),
            params={"wait": "true"},
            json={"points": records},
        )
        self._raise(response, "upsert points")

    async def search(
        self,
        query_vector: list[float],
        limit: int,
        model_id: str,
        dimension: int,
        repository_id: str | None = None,
        snapshot_id: str | None = None,
    ) -> list[VectorRecord]:
        must = []
        if repository_id is not None:
            must.append({"key": "repository_id", "match": {"value": repository_id}})
        if snapshot_id is not None:
            must.append({"key": "snapshot_id", "match": {"value": snapshot_id}})
        body: dict[str, object] = {
            "vector": query_vector,
            "limit": limit,
            "with_payload": True,
        }
        if must:
            body["filter"] = {"must": must}
        response = await self.client.post(
            f"{self._url(self._collection_name(model_id, dimension))}/points/search",
            headers=self._headers(),
            json=body,
        )
        self._raise(response, "search points")
        results = cast(list[dict[str, object]], response.json().get("result", []))
        records: list[VectorRecord] = []
        for result in results:
            raw_score = result.get("score", 0.0)
            score = float(raw_score) if isinstance(raw_score, (int, float, str)) else 0.0
            records.append(
                VectorRecord(
                    id=str(result["id"]),
                    vector=cast(list[float], result.get("vector", [])),
                    score=score,
                    payload=cast(dict[str, object], result.get("payload", {})),
                )
            )
        return records

    async def close(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    def _collection_name(self, model_id: str, dimension: int) -> str:
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", model_id).strip("-").lower()[:32]
        digest = hashlib.sha256(model_id.encode()).hexdigest()[:8]
        return f"{self.collection}-{slug}-{dimension}-{digest}"

    def _url(self, collection: str) -> str:
        return f"{self.url}/collections/{collection}"

    def _headers(self) -> dict[str, str]:
        return {"api-key": self.api_key} if self.api_key else {}

    @staticmethod
    def _raise(response: httpx.Response, action: str) -> None:
        if response.is_error:
            raise VectorStoreError(
                f"Unable to {action}: Qdrant returned {response.status_code}: {response.text[:500]}"
            )
