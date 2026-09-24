from typing import Protocol


class EmbeddingProviderError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.retry_after = retry_after


class EmbeddingProvider(Protocol):
    model_id: str
    dimension: int

    async def validate(self) -> None:
        """Probe the configured model and validate its embedding shape."""

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents."""

    async def embed_query(self, text: str) -> list[float]:
        """Embed a search query."""

    async def close(self) -> None:
        """Release provider resources."""
