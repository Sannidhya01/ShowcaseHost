from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Literal, cast

import httpx

from app.integrations.generation.base import (
    ChatMessage,
    GenerationProvider,
    GenerationProviderError,
)

ProviderSort = Literal["price", "throughput", "latency"]
ReasoningEffort = Literal["none", "minimal", "low", "medium", "high"]


def _retry_delay(response: httpx.Response, attempt: int) -> float:
    retry_after = response.headers.get("retry-after")
    if retry_after:
        try:
            return max(float(retry_after), 0.0)
        except ValueError:
            pass
    return float(min(2**attempt, 8))


class OpenRouterGenerationProvider(GenerationProvider):
    """OpenRouter chat adapter with explicit lowest-price provider routing."""

    def __init__(
        self,
        api_key: str | None,
        model: str,
        *,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        max_completion_tokens: int = 4096,
        provider_sort: ProviderSort = "price",
        reasoning_effort: ReasoningEffort = "low",
        site_url: str | None = None,
        app_name: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.url = f"{base_url.rstrip('/')}/chat/completions"
        self.max_retries = max_retries
        self.max_completion_tokens = max_completion_tokens
        self.provider_sort = provider_sort
        self.reasoning_effort = reasoning_effort
        self.site_url = site_url
        self.app_name = app_name
        self._owns_client = client is None
        self.client = client or httpx.AsyncClient(timeout=timeout_seconds)

    async def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        if not self.api_key:
            raise GenerationProviderError("OPENROUTER_API_KEY is required")
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "max_completion_tokens": self.max_completion_tokens,
            "provider": {
                "sort": self.provider_sort,
                "allow_fallbacks": True,
            },
            "reasoning": {"effort": self.reasoning_effort, "exclude": True},
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.app_name:
            headers["X-OpenRouter-Title"] = self.app_name

        for attempt in range(self.max_retries + 1):
            emitted_content = False
            try:
                async with self.client.stream(
                    "POST", self.url, headers=headers, json=payload
                ) as response:
                    if response.is_error:
                        body = (await response.aread()).decode(errors="replace")[:500]
                        retryable = response.status_code in {408, 409, 429, 500, 502, 503, 504}
                        if retryable and attempt < self.max_retries:
                            await asyncio.sleep(_retry_delay(response, attempt))
                            continue
                        raise GenerationProviderError(
                            f"OpenRouter returned {response.status_code}: {body}",
                            retryable=retryable,
                        )
                    stream_error: str | None = None
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if not data or data == "[DONE]":
                            continue
                        try:
                            event = cast(dict[str, object], json.loads(data))
                        except (TypeError, ValueError) as exc:
                            raise GenerationProviderError(
                                "OpenRouter returned a malformed streaming response"
                            ) from exc
                        error = event.get("error")
                        if error is not None:
                            stream_error = str(error)[:500]
                            break
                        try:
                            choices = cast(list[dict[str, object]], event["choices"])
                            delta = cast(dict[str, object], choices[0].get("delta", {}))
                            content = delta.get("content")
                        except (KeyError, IndexError, TypeError) as exc:
                            raise GenerationProviderError(
                                "OpenRouter returned a malformed streaming response"
                            ) from exc
                        if isinstance(content, str) and content:
                            emitted_content = True
                            yield content
                    if stream_error is not None:
                        if not emitted_content and attempt < self.max_retries:
                            await asyncio.sleep(float(min(2**attempt, 8)))
                            continue
                        raise GenerationProviderError(
                            f"OpenRouter stream failed: {stream_error}", retryable=True
                        )
                    if emitted_content:
                        return
                    if attempt < self.max_retries:
                        await asyncio.sleep(float(min(2**attempt, 8)))
                        continue
                    raise GenerationProviderError(
                        "OpenRouter returned an empty response", retryable=True
                    )
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                if emitted_content or attempt >= self.max_retries:
                    raise GenerationProviderError(
                        f"OpenRouter request failed: {exc}", retryable=True
                    ) from exc
                await asyncio.sleep(float(min(2**attempt, 8)))

    async def close(self) -> None:
        if self._owns_client:
            await self.client.aclose()
