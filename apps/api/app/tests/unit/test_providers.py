from app.integrations.embeddings.mock import MockEmbeddingProvider
from app.integrations.generation.base import ChatMessage
from app.integrations.generation.mock import MockGenerationProvider
from app.integrations.vector_store.mock import MockVectorStore


async def test_mock_providers_are_usable() -> None:
    embeddings = MockEmbeddingProvider()
    generation = MockGenerationProvider()
    vector_store = MockVectorStore()

    vector = await embeddings.embed_query("hello")
    await vector_store.upsert([{"id": "1", "vector": vector, "payload": {"path": "README.md"}}])
    results = await vector_store.search(vector, limit=1)
    message: ChatMessage = {"role": "user", "content": "hello"}

    assert vector == [5.0, 0.0, 1.0]
    assert results[0]["id"] == "1"
    assert await generation.generate([message]) == "hello"
