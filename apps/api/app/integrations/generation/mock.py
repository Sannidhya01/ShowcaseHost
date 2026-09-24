from collections.abc import AsyncIterator

from app.integrations.generation.base import ChatMessage, GenerationProvider


class MockGenerationProvider(GenerationProvider):
    async def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        if messages:
            yield messages[-1]["content"]

    async def close(self) -> None:
        return None

    async def generate(self, messages: list[ChatMessage]) -> str:
        return "".join([part async for part in self.stream(messages)])
