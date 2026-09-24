from app.core.config import Settings
from app.db.models import Repository
from app.integrations.generation.base import ChatMessage
from app.integrations.vector_store.base import VectorRecord
from app.services.chat import (
    ChatSource,
    apply_reranker_scores,
    bounded_history,
    build_messages,
    configured_retrieval_candidates,
    rank_retrieved_chunks,
    reciprocal_rank_fusion,
    rerank_retrieved_chunks,
    source_from_record,
)


def repository() -> Repository:
    return Repository(
        github_repository_id=1,
        installation_id=2,
        owner="octocat",
        name="hello",
        full_name="octocat/hello",
        private=False,
        default_branch="main",
    )


def test_history_keeps_newest_complete_turns_within_limits() -> None:
    history: list[ChatMessage] = [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "new question"},
        {"role": "assistant", "content": "new answer"},
        {"role": "user", "content": "incomplete"},
    ]
    assert bounded_history(history, max_messages=2, max_chars=100) == history[2:4]
    assert bounded_history(history, max_messages=12, max_chars=5) == []


def test_prompt_contains_guardrails_sources_and_citation_instructions() -> None:
    source = ChatSource(1, "chunk", "src/app.py", "python", 3, 8, "a" * 40, "print('hi')")
    prepared = build_messages(repository(), "What runs?", [], [source], Settings(app_env="test"))

    system = prepared.messages[0]["content"]
    assert "read-only" in system
    assert "untrusted data" in system
    assert "cite them inline as [1]" in system
    assert "src/app.py:3-8" in system
    assert prepared.messages[-1] == {"role": "user", "content": "What runs?"}
    assert prepared.sources == [source]


def test_source_metadata_uses_raw_text_and_one_based_lines() -> None:
    source = source_from_record(
        {
            "id": "chunk-id",
            "payload": {
                "path": "src/main.ts",
                "language": "typescript",
                "start_line": 1,
                "end_line": 4,
                "raw_text": "export const value = 1;",
            },
        },
        2,
        "b" * 40,
    )
    assert source.public_dict()["number"] == 2
    assert source.text == "export const value = 1;"
    assert source.start_line == 1


def test_chunk_quality_breaks_only_equal_relevance_scores() -> None:
    records: list[VectorRecord] = [
        {"id": "fallback-best", "score": 0.9, "payload": {"quality_score": 89}},
        {"id": "fallback-tie", "score": 0.8, "payload": {"quality_score": 75}},
        {"id": "preferred-tie", "score": 0.8, "payload": {"quality_score": 100}},
        {"id": "preferred-lower", "score": 0.7, "payload": {"quality_score": 100}},
    ]

    ranked = rank_retrieved_chunks(records)

    assert [record["id"] for record in ranked] == [
        "fallback-best",
        "preferred-tie",
        "fallback-tie",
        "preferred-lower",
    ]


def test_rrf_combines_dense_and_keyword_ranks_by_stable_chunk_id() -> None:
    dense: list[VectorRecord] = [
        {"id": "dense-only", "score": 0.99, "payload": {"quality_score": 100}},
        {"id": "shared", "score": 0.80, "payload": {"path": "shared.py"}},
    ]
    keyword: list[VectorRecord] = [
        {"id": "shared", "score": 12.0, "payload": {"raw_text": "authoritative"}},
        {"id": "keyword-only", "score": 10.0, "payload": {}},
    ]

    fused = reciprocal_rank_fusion(dense, keyword, limit=8, k=60)

    assert [record["id"] for record in fused] == [
        "shared",
        "dense-only",
        "keyword-only",
    ]
    assert fused[0]["score"] == (1 / 62) + (1 / 61)
    assert fused[0]["payload"] == {
        "path": "shared.py",
        "raw_text": "authoritative",
    }


def test_rrf_limits_results_and_uses_quality_only_for_fusion_ties() -> None:
    dense: list[VectorRecord] = [
        {"id": "fallback", "score": 2.0, "payload": {"quality_score": 75}},
        {"id": "preferred", "score": 2.0, "payload": {"quality_score": 100}},
        {"id": "third", "score": 1.0, "payload": {"quality_score": 100}},
    ]

    fused = reciprocal_rank_fusion(dense, [], limit=2)

    assert [record["id"] for record in fused] == ["preferred", "fallback"]


def test_rrf_default_weights_contribute_equally_at_the_same_rank() -> None:
    dense: list[VectorRecord] = [{"id": "vector", "score": 0.9, "payload": {}}]
    keyword: list[VectorRecord] = [{"id": "keyword", "score": 12.0, "payload": {}}]

    fused = reciprocal_rank_fusion(dense, keyword, limit=8)

    assert [record["id"] for record in fused] == ["keyword", "vector"]
    assert fused[0]["score"] == 1.0 / 61
    assert fused[1]["score"] == 1.0 / 61


def test_reranker_score_controls_cutoff_order_and_five_chunk_cap() -> None:
    candidates: list[VectorRecord] = [
        {"id": str(index), "score": 1 - index / 10, "payload": {}}
        for index in range(7)
    ]
    settings = Settings(
        app_env="test",
        chat_relevance_threshold=0.4,
        chat_relative_relevance_threshold=0,
    )
    results = apply_reranker_scores(
        candidates, [0.41, 0.99, 0.2, 0.8, 0.7, 0.6, 0.5], settings
    )
    assert [record["id"] for record in results] == ["1", "3", "4", "5", "6"]
    assert results[0]["payload"]["reranker_score"] == 0.99


def test_reranker_score_rejects_invalid_values() -> None:
    import pytest

    candidates: list[VectorRecord] = [{"id": "one", "payload": {}}]
    for score in [float("nan"), float("inf"), -0.1, 1.1]:
        with pytest.raises(ValueError):
            apply_reranker_scores(candidates, [score], Settings(app_env="test"))
    with pytest.raises(ValueError):
        apply_reranker_scores(candidates, [], Settings(app_env="test"))


def test_relative_relevance_gate_removes_weak_followers_but_keeps_close_scores() -> None:
    candidates: list[VectorRecord] = [
        {"id": "best", "payload": {}},
        {"id": "close", "payload": {}},
        {"id": "weak", "payload": {}},
    ]
    settings = Settings(
        app_env="test",
        chat_relevance_threshold=0.1,
        chat_relative_relevance_threshold=0.855,
    )
    results = apply_reranker_scores(candidates, [0.8, 0.7, 0.6], settings)
    assert [record["id"] for record in results] == ["best", "close"]


def test_chat_settings_validate_threshold_and_hard_cap() -> None:
    import pytest
    from pydantic import ValidationError

    for threshold in [-0.1, 1.1, float("nan"), float("inf")]:
        with pytest.raises(ValidationError):
            Settings(app_env="test", chat_relevance_threshold=threshold)
    for limit in [0, 6]:
        with pytest.raises(ValidationError):
            Settings(app_env="test", chat_retrieval_limit=limit)


def test_empty_context_prompt_explicitly_abstains_despite_history() -> None:
    from app.services.chat import NO_RELEVANT_INFORMATION

    history: list[ChatMessage] = [
        {"role": "user", "content": "What is the entrypoint?"},
        {"role": "assistant", "content": "It is main [1]."},
    ]
    prepared = build_messages(repository(), "What else?", history, [], Settings(app_env="test"))
    assert prepared.sources == []
    assert NO_RELEVANT_INFORMATION in prepared.messages[0]["content"]
    assert "Do not infer an answer from conversation history" in prepared.messages[0]["content"]


def test_fusion_provides_more_than_five_candidates_to_reranker() -> None:
    dense: list[VectorRecord] = [
        {"id": str(index), "score": 1 - index / 100} for index in range(24)
    ]
    results = configured_retrieval_candidates(dense, [], Settings(app_env="test"))
    assert len(results) == 20


async def test_reranker_uses_fallback_and_promotes_more_relevant_chunk() -> None:
    from app.integrations.reranking.base import RerankerProviderError
    from app.integrations.reranking.mock import MockRerankerProvider

    attempted: list[str] = []

    class FailedReranker(MockRerankerProvider):
        async def score(self, query: str, passages: list[str]) -> list[float]:
            raise RerankerProviderError("unavailable", retryable=False)

    def factory(model_id: str) -> MockRerankerProvider:
        attempted.append(model_id)
        if model_id == "BAAI/bge-reranker-v2-m3":
            return FailedReranker(model_id)
        return MockRerankerProvider(model_id, [0.2, 0.9])

    candidates: list[VectorRecord] = [
        {"id": "rrf-first", "payload": {"raw_text": "weak"}},
        {"id": "relevant", "payload": {"raw_text": "strong"}},
    ]
    results = await rerank_retrieved_chunks(
        "query",
        candidates,
        Settings(
            app_env="test",
            chat_relevance_threshold=0.1,
            chat_relative_relevance_threshold=0,
        ),
        factory,
    )
    assert attempted == ["BAAI/bge-reranker-v2-m3", "BAAI/bge-reranker-large"]
    assert [record["id"] for record in results] == ["relevant", "rrf-first"]
