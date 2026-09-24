from app.integrations.embeddings.mock import MockEmbeddingProvider
from app.integrations.generation.mock import MockGenerationProvider
from app.integrations.vector_store.mock import MockVectorStore


def get_embedding_provider() -> MockEmbeddingProvider:
    return MockEmbeddingProvider()


def get_generation_provider() -> MockGenerationProvider:
    return MockGenerationProvider()


def get_vector_store() -> MockVectorStore:
    return MockVectorStore()
