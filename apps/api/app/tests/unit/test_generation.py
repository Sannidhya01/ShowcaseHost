from __future__ import annotations

import json

import httpx
import pytest

from app.integrations.generation.base import ChatMessage, GenerationProviderError
from app.integrations.generation.catalog import GENERATION_MODELS, require_generation_model
from app.integrations.generation.groq import GroqGenerationProvider


async def collect(provider: GroqGenerationProvider, messages: list[ChatMessage]) -> str:
    return "".join([part async for part in provider.stream(messages)])


async def test_groq_streams_chat_payload_and_content() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        content = (
            'data: {"choices":[{"delta":{"content":"Grounded "}}]}\n\n'
            'data: {"choices":[{"delta":{"content":"answer [1]"}}]}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(200, text=content)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GroqGenerationProvider(
        "secret",
        "qwen/qwen3.8-27b",
        client=client,
        max_completion_tokens=321,
        service_tier="auto",
    )
    messages: list[ChatMessage] = [{"role": "user", "content": "Where?"}]

    assert await collect(provider, messages) == "Grounded answer [1]"
    assert captured == {
        "model": "qwen/qwen3.8-27b",
        "messages": messages,
        "stream": True,
        "max_completion_tokens": 321,
        "reasoning_format": "hidden",
        "service_tier": "auto",
    }
    await client.aclose()


async def test_groq_rejects_missing_key_and_malformed_stream() -> None:
    missing = GroqGenerationProvider(None, "qwen/qwen3.8-27b")
    with pytest.raises(GenerationProviderError, match="GROQ_API_KEY"):
        await collect(missing, [{"role": "user", "content": "hello"}])
    await missing.close()

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, text='data: {"unexpected":true}\n\n')
        )
    )
    malformed = GroqGenerationProvider("secret", "qwen/qwen3.8-27b", client=client)
    with pytest.raises(GenerationProviderError, match="malformed"):
        await collect(malformed, [{"role": "user", "content": "hello"}])
    await client.aclose()


@pytest.mark.parametrize("status", [408, 429, 498, 500, 502, 503, 504])
async def test_groq_classifies_retryable_statuses(status: int) -> None:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(status, text="later"))
    )
    provider = GroqGenerationProvider("secret", "openai/gpt-oss-20b", client=client, max_retries=0)
    with pytest.raises(GenerationProviderError) as caught:
        await collect(provider, [{"role": "user", "content": "hello"}])
    assert caught.value.retryable is True
    await client.aclose()


async def test_groq_maps_network_failures() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GroqGenerationProvider("secret", "qwen/qwen3.8-27b", client=client, max_retries=0)
    with pytest.raises(GenerationProviderError) as caught:
        await collect(provider, [{"role": "user", "content": "hello"}])
    assert caught.value.retryable is True
    await client.aclose()


async def test_groq_honors_retry_after_header(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, headers={"retry-after": "7.5"}, text="slow down")
        return httpx.Response(
            200,
            text='data: {"choices":[{"delta":{"content":"answer"}}]}\n\ndata: [DONE]\n\n',
        )

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("app.integrations.generation.groq.asyncio.sleep", fake_sleep)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GroqGenerationProvider("secret", "qwen/qwen3.8-27b", client=client)

    assert await collect(provider, [{"role": "user", "content": "hello"}]) == "answer"
    assert attempts == 2
    assert sleeps == [7.5]
    await client.aclose()


async def test_groq_retries_an_empty_successful_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(200, text="data: [DONE]\n\n")
        return httpx.Response(
            200,
            text='data: {"choices":[{"delta":{"content":"recovered"}}]}\n\ndata: [DONE]\n\n',
        )

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("app.integrations.generation.groq.asyncio.sleep", fake_sleep)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GroqGenerationProvider("secret", "qwen/qwen3.8-27b", client=client, max_retries=1)

    assert await collect(provider, [{"role": "user", "content": "hello"}]) == "recovered"
    assert attempts == 2
    assert sleeps == [1.0]
    await client.aclose()


def test_generation_catalog_has_primary_and_faster_models() -> None:
    assert [model.id for model in GENERATION_MODELS] == [
        "qwen/qwen3.8-27b",
        "openai/gpt-oss-20b",
    ]
    assert require_generation_model("qwen/qwen3.8-27b").provider == "groq"
    assert require_generation_model("openai/gpt-oss-20b").provider == "openrouter"
    assert require_generation_model("openai/gpt-oss-20b").is_default is True
    with pytest.raises(ValueError, match="Unsupported generation model"):
        require_generation_model("unknown")
