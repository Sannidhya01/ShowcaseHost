from __future__ import annotations

import json

import httpx
import pytest

from app.integrations.generation.base import ChatMessage, GenerationProviderError
from app.integrations.generation.openrouter import OpenRouterGenerationProvider


async def collect(provider: OpenRouterGenerationProvider, messages: list[ChatMessage]) -> str:
    return "".join([part async for part in provider.stream(messages)])


async def test_openrouter_streams_from_the_lowest_price_provider() -> None:
    captured: dict[str, object] = {}
    captured_headers: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        captured_headers.update(request.headers)
        return httpx.Response(
            200,
            text='data: {"choices":[{"delta":{"reasoning":"private"}}]}\n\n'
            'data: {"choices":[{"delta":{"content":"grounded answer"}}]}\n\n'
            "data: [DONE]\n\n",
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenRouterGenerationProvider(
        "secret",
        "openai/gpt-oss-20b",
        client=client,
        max_completion_tokens=2048,
        site_url="http://localhost:3000",
        app_name="ShowcaseHost",
    )
    messages: list[ChatMessage] = [{"role": "user", "content": "Where?"}]

    assert await collect(provider, messages) == "grounded answer"
    assert captured == {
        "model": "openai/gpt-oss-20b",
        "messages": messages,
        "stream": True,
        "max_completion_tokens": 2048,
        "provider": {
            "sort": "price",
            "allow_fallbacks": True,
        },
        "reasoning": {"effort": "low", "exclude": True},
    }
    assert captured_headers["authorization"] == "Bearer secret"
    assert captured_headers["http-referer"] == "http://localhost:3000"
    assert captured_headers["x-openrouter-title"] == "ShowcaseHost"
    await client.aclose()


async def test_openrouter_requires_a_key() -> None:
    provider = OpenRouterGenerationProvider(None, "openai/gpt-oss-20b")

    with pytest.raises(GenerationProviderError, match="OPENROUTER_API_KEY"):
        await collect(provider, [{"role": "user", "content": "hello"}])
    await provider.close()


async def test_openrouter_honors_retry_after(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, headers={"retry-after": "2.5"}, text="slow down")
        return httpx.Response(
            200,
            text='data: {"choices":[{"delta":{"content":"recovered"}}]}\n\ndata: [DONE]\n\n',
        )

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("app.integrations.generation.openrouter.asyncio.sleep", fake_sleep)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenRouterGenerationProvider(
        "secret", "openai/gpt-oss-20b", client=client, max_retries=1
    )

    assert await collect(provider, [{"role": "user", "content": "hello"}]) == "recovered"
    assert sleeps == [2.5]
    await client.aclose()


async def test_openrouter_rejects_an_empty_stream() -> None:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, text="data: [DONE]\n\n"))
    )
    provider = OpenRouterGenerationProvider(
        "secret", "openai/gpt-oss-20b", client=client, max_retries=0
    )

    with pytest.raises(GenerationProviderError, match="empty response"):
        await collect(provider, [{"role": "user", "content": "hello"}])
    await client.aclose()
