from app.evaluation.rate_limit import EvaluationRateLimiter


async def test_rate_limiter_spaces_requests_and_respects_token_window() -> None:
    now = 0.0
    sleeps: list[float] = []

    def clock() -> float:
        return now

    async def sleep(seconds: float) -> None:
        nonlocal now
        sleeps.append(seconds)
        now += seconds

    limiter = EvaluationRateLimiter(20, 1_000, clock=clock, sleeper=sleep)

    await limiter.acquire(600)
    await limiter.acquire(400)
    await limiter.acquire(600)

    assert sleeps == [3.0, 57.0]


def test_rate_limiter_estimates_request_tokens() -> None:
    assert EvaluationRateLimiter.estimate_tokens([{"content": "x" * 400}], 50) == 250
