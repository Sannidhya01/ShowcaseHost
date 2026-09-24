from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator
from typing import cast

import httpx

from app.integrations.generation.base import (
    ChatMessage,
    GenerationProvider,
    GenerationProviderError,
)


def _duration_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(float(value), 0.0)
    except ValueError:
        pass
    matches = re.findall(r"([0-9]*\.?[0-9]+)(ms|s|m|h)", value.lower())
    if not matches:
        return None
    units: dict[str, float] = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}
    return sum(float(amount) * units[unit] for amount, unit in matches)


def _retry_delay(response: httpx.Response, attempt: int) -> float:
    retry_after = _duration_seconds(response.headers.get("retry-after"))
    if retry_after is not None:
        return retry_after
    reset = _duration_seconds(response.headers.get("x-ratelimit-reset-tokens"))
    if response.status_code == 429 and reset is not None:
        return reset
    return float(min(2**attempt, 4))


class GroqGenerationProvider(GenerationProvider):
    def __init__(
        self,
        api_key: str | None,
        model: str,
        *,
        base_url: str = "https://api.groq.com/openai/v1",
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        max_completion_tokens: int = 1200,
        service_tier: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.url = f"{base_url.rstrip('/')}/chat/completions"
        self.max_retries = max_retries
        self.max_completion_tokens = max_completion_tokens
        self.service_tier = service_tier
        self._owns_client = client is None
        self.client = client or httpx.AsyncClient(timeout=timeout_seconds)

    async def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        if not self.api_key:
            raise GenerationProviderError("GROQ_API_KEY is required")
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "max_completion_tokens": self.max_completion_tokens,
            # Keep chain-of-thought private while preserving the model's reasoning quality.
            "reasoning_format": "hidden",
        }
        if self.service_tier:
            payload["service_tier"] = self.service_tier
        for attempt in range(self.max_retries + 1):
            try:
                async with self.client.stream(
                    "POST",
                    self.url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                ) as response:
                    if response.is_error:
                        body = (await response.aread()).decode(errors="replace")[:500]
                        retryable = response.status_code in {408, 429, 498, 500, 502, 503, 504}
                        if retryable and attempt < self.max_retries:
                            await asyncio.sleep(_retry_delay(response, attempt))
                            continue
                        raise GenerationProviderError(
                            f"Groq returned {response.status_code}: {body}", retryable=retryable
                        )
                    emitted_content = False
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if not data or data == "[DONE]":
                            continue
                        try:
                            event = cast(dict[str, object], json.loads(data))
                            choices = cast(list[dict[str, object]], event["choices"])
                            delta = cast(dict[str, object], choices[0].get("delta", {}))
                            content = delta.get("content")
                        except (KeyError, IndexError, TypeError, ValueError) as exc:
                            raise GenerationProviderError(
                                "Groq returned a malformed streaming response"
                            ) from exc
                        if isinstance(content, str) and content:
                            emitted_content = True
                            yield content
                    if emitted_content:
                        return
                    if attempt < self.max_retries:
                        await asyncio.sleep(float(min(2**attempt, 4)))
                        continue
                    raise GenerationProviderError("Groq returned an empty response", retryable=True)
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                if attempt >= self.max_retries:
                    raise GenerationProviderError(
                        f"Groq request failed: {exc}", retryable=True
                    ) from exc
                await asyncio.sleep(min(2**attempt, 4))

    async def close(self) -> None:
        if self._owns_client:
            await self.client.aclose()
