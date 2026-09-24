from __future__ import annotations

from typing import cast

import httpx

from app.integrations.embeddings.base import EmbeddingProvider, EmbeddingProviderError


class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        token: str | None,
        model_id: str,
        dimension: int,
        *,
        base_url: str = "https://router.huggingface.co/hf-inference/models",
        timeout_seconds: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.token = token
        self.model_id = model_id
        self.dimension = dimension
        self._owns_client = client is None
        self.client = client or httpx.AsyncClient(timeout=timeout_seconds)

        model_path = "/".join(httpx.URL(model_id).path.split("/"))
        self.url = f"{base_url.rstrip('/')}/{model_path}/pipeline/feature-extraction"

    async def validate(self) -> None:
        vectors = await self.embed_documents(["ShowcaseHost embedding provider validation"])
        if len(vectors) != 1 or len(vectors[0]) != self.dimension:
            actual = len(vectors[0]) if vectors else 0
            raise EmbeddingProviderError(
                f"Model {self.model_id} returned dimension {actual}; expected {self.dimension}",
                retryable=False,
            )

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if not self.token:
            raise EmbeddingProviderError(
                "HF_API_TOKEN is required to generate embeddings", retryable=False
            )
        try:
            response = await self.client.post(
                self.url,
                headers={"Authorization": f"Bearer {self.token}"},
                json={"inputs": texts, "normalize": True, "truncate": True},
            )
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            raise EmbeddingProviderError(
                f"Hugging Face request failed: {exc}", retryable=True
            ) from exc

        if response.is_error:
            detail, estimated = self._error_detail(response)
            retry_after = self._retry_after(response, estimated)
            retryable = response.status_code in {408, 429, 502, 503, 504}
            raise EmbeddingProviderError(
                f"Hugging Face returned {response.status_code} for {self.model_id}: {detail}",
                retryable=retryable,
                retry_after=retry_after,
            )
        try:
            body = response.json()
            vectors = self._coerce_vectors(body, len(texts))
        except (TypeError, ValueError) as exc:
            raise EmbeddingProviderError(
                f"Invalid embedding response from {self.model_id}: {exc}", retryable=False
            ) from exc
        if any(len(vector) != self.dimension for vector in vectors):
            raise EmbeddingProviderError(
                f"Model {self.model_id} returned an unexpected embedding dimension",
                retryable=False,
            )
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_documents([text]))[0]

    async def close(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    @staticmethod
    def _coerce_vectors(body: object, expected: int) -> list[list[float]]:
        if not isinstance(body, list):
            raise TypeError("response body is not an array")
        if expected == 1 and body and isinstance(body[0], (int, float)):
            body = [body]
        if len(body) != expected:
            raise ValueError(f"returned {len(body)} vectors for {expected} inputs")
        vectors: list[list[float]] = []
        for item in body:
            if not isinstance(item, list) or any(
                not isinstance(value, (int, float)) for value in item
            ):
                raise TypeError("response contains token-level or non-numeric embeddings")
            vectors.append([float(value) for value in item])
        return vectors

    @staticmethod
    def _error_detail(response: httpx.Response) -> tuple[str, float | None]:
        try:
            body = cast(dict[str, object], response.json())
        except ValueError:
            return response.text[:500] or "unknown error", None
        detail = str(body.get("error") or body.get("message") or "unknown error")
        estimated = body.get("estimated_time")
        return detail[:500], float(estimated) if isinstance(estimated, (int, float)) else None

    @staticmethod
    def _retry_after(response: httpx.Response, estimated: float | None) -> float | None:
        header = response.headers.get("retry-after")
        if header:
            try:
                return float(header)
            except ValueError:
                pass
        return estimated
