from __future__ import annotations

from typing import cast

import httpx

from app.integrations.reranking.base import RerankerProvider, RerankerProviderError


class HuggingFaceRerankerProvider(RerankerProvider):
    def __init__(
        self,
        token: str | None,
        model_id: str,
        *,
        base_url: str = "https://router.huggingface.co/hf-inference/models",
        timeout_seconds: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.model_id = model_id
        self._token = token
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout_seconds)
        model_path = "/".join(httpx.URL(model_id).path.split("/"))
        self._url = f"{base_url.rstrip('/')}/{model_path}"

    async def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        if not self._token:
            raise RerankerProviderError(
                "HF_API_TOKEN is required to rerank retrieved chunks", retryable=False
            )
        try:
            response = await self._client.post(
                self._url,
                headers={"Authorization": f"Bearer {self._token}"},
                json={
                    "inputs": [
                        {"text": query, "text_pair": passage} for passage in passages
                    ]
                },
            )
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            raise RerankerProviderError(
                f"Hugging Face reranking request failed: {exc}", retryable=True
            ) from exc
        if response.is_error:
            detail, estimated = self._error_detail(response)
            retry_after = self._retry_after(response, estimated)
            raise RerankerProviderError(
                f"Hugging Face returned {response.status_code} for {self.model_id}: {detail}",
                retryable=response.status_code in {408, 429, 502, 503, 504},
                retry_after=retry_after,
            )
        try:
            return self._coerce_scores(response.json(), len(passages))
        except (TypeError, ValueError) as exc:
            raise RerankerProviderError(
                f"Invalid reranker response from {self.model_id}: {exc}", retryable=False
            ) from exc

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    @staticmethod
    def _coerce_scores(body: object, expected: int) -> list[float]:
        # HF text classification returns [[{label, score}, ...]] for a batched
        # request to the one-label BGE sequence classifier.
        if (
            isinstance(body, list)
            and len(body) == 1
            and isinstance(body[0], list)
            and len(body[0]) == expected
        ):
            body = body[0]
        if not isinstance(body, list) or len(body) != expected:
            actual = len(body) if isinstance(body, list) else "non-array"
            raise ValueError(f"returned {actual} scores for {expected} passages")
        result: list[float] = []
        for item in body:
            if isinstance(item, list) and len(item) == 1:
                item = item[0]
            if not isinstance(item, dict) or not isinstance(item.get("score"), (int, float)):
                raise TypeError("response contains a non-numeric relevance score")
            score = float(item["score"])
            if not 0 <= score <= 1:
                raise ValueError("normalized relevance score is outside [0, 1]")
            result.append(score)
        return result

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
