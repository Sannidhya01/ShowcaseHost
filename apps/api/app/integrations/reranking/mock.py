class MockRerankerProvider:
    def __init__(self, model_id: str = "mock/reranker", scores: list[float] | None = None) -> None:
        self.model_id = model_id
        self.scores = scores
        self.closed = False

    async def score(self, query: str, passages: list[str]) -> list[float]:
        if self.scores is not None:
            return self.scores[: len(passages)]
        query_terms = set(query.lower().split())
        return [
            len(query_terms & set(passage.lower().split())) / max(len(query_terms), 1)
            for passage in passages
        ]

    async def close(self) -> None:
        self.closed = True
