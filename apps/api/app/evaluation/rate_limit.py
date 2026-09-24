from __future__ import annotations

import asyncio
import math
import time
from collections import deque
from collections.abc import Awaitable, Callable

Clock = Callable[[], float]
Sleeper = Callable[[float], Awaitable[None]]


class EvaluationRateLimiter:
    """Conservative rolling-window limiter shared by all local evaluation runs."""

    def __init__(
        self,
        requests_per_minute: int,
        tokens_per_minute: int,
        *,
        clock: Clock | None = None,
        sleeper: Sleeper = asyncio.sleep,
    ) -> None:
        self._requests_per_minute = requests_per_minute
        self._tokens_per_minute = tokens_per_minute
        self._clock = clock or time.monotonic
        self._sleep = sleeper
        self._window: deque[tuple[float, int]] = deque()
        self._last_request: float | None = None
        self._lock = asyncio.Lock()

    @staticmethod
    def estimate_tokens(messages: list[dict[str, str]], completion_tokens: int) -> int:
        prompt_chars = sum(len(message["content"]) for message in messages)
        # Source code often tokenizes more densely than prose; two characters per token
        # deliberately overestimates most evaluation prompts.
        return math.ceil(prompt_chars / 2) + completion_tokens

    async def acquire(self, estimated_tokens: int) -> None:
        if estimated_tokens > self._tokens_per_minute:
            raise ValueError(
                "Evaluation request exceeds the configured token-per-minute safety budget"
            )
        minimum_interval = 60 / self._requests_per_minute
        async with self._lock:
            while True:
                now = self._clock()
                while self._window and now - self._window[0][0] >= 60:
                    self._window.popleft()
                waits: list[float] = []
                if self._last_request is not None:
                    waits.append(self._last_request + minimum_interval - now)
                used_tokens = sum(tokens for _, tokens in self._window)
                if (
                    len(self._window) >= self._requests_per_minute
                    or used_tokens + estimated_tokens > self._tokens_per_minute
                ):
                    waits.append(self._window[0][0] + 60 - now)
                wait_for = max(waits, default=0.0)
                if wait_for > 0:
                    await self._sleep(wait_for)
                    continue
                timestamp = self._clock()
                self._window.append((timestamp, estimated_tokens))
                self._last_request = timestamp
                return
