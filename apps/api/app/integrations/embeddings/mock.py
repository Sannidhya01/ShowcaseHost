from app.integrations.embeddings.base import EmbeddingProvider


class MockEmbeddingProvider(EmbeddingProvider):
    def __init__(self, dimension: int = 3) -> None:
        self.model_id = "mock/embedding"
        self.dimension = dimension

    async def validate(self) -> None:
        return None

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 0.0, 1.0][: self.dimension] for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 0.0, 1.0][: self.dimension]

    async def close(self) -> None:
        return None
