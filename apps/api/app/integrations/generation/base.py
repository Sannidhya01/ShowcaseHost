from collections.abc import AsyncIterator
from typing import Literal, Protocol, TypedDict


class ChatMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


class GenerationProviderError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class GenerationProvider(Protocol):
    def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        """Stream response text from chat-style messages."""

    async def close(self) -> None:
        """Release provider resources."""
