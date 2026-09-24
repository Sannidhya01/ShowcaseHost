import httpx
import pytest

from app.integrations.reranking.base import RerankerProviderError
from app.integrations.reranking.huggingface import HuggingFaceRerankerProvider


async def test_huggingface_reranker_sends_pairs_and_parses_batched_scores() -> None:
    request_body: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        request_body.update(__import__("json").loads(request.content))
        return httpx.Response(
            200,
            json=[[{"label": "LABEL_0", "score": 0.9}, {"label": "LABEL_0", "score": 0.2}]],
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = HuggingFaceRerankerProvider("token", "BAAI/bge-reranker-v2-m3", client=client)
    try:
        assert await provider.score("query", ["first", "second"]) == [0.9, 0.2]
    finally:
        await client.aclose()
    assert request_body == {
        "inputs": [
            {"text": "query", "text_pair": "first"},
            {"text": "query", "text_pair": "second"},
        ]
    }


async def test_huggingface_reranker_reports_retryable_errors() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "loading", "estimated_time": 3})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = HuggingFaceRerankerProvider("token", "BAAI/bge-reranker-v2-m3", client=client)
    try:
        with pytest.raises(RerankerProviderError) as captured:
            await provider.score("query", ["passage"])
    finally:
        await client.aclose()
    assert captured.value.retryable is True
    assert captured.value.retry_after == 3
