from typing import Protocol


class RerankerProviderError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.retry_after = retry_after


class RerankerProvider(Protocol):
    model_id: str

    async def score(self, query: str, passages: list[str]) -> list[float]:
        """Return one normalized relevance score per query-passage pair."""

    async def close(self) -> None:
        """Release provider resources."""
