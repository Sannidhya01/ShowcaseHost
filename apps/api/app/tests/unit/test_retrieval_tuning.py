from pathlib import Path
from typing import Any, cast

import pytest

from app.core.config import Settings
from scripts.tune_codequeries import evaluate, precision_recall_sort_key, recall_f1_harmonic


def test_tuning_aggregates_confusion_and_tracks_negative_abstention() -> None:
    examples = [
        {
            "row_index": 1,
            "example_type": 1,
            "labels": {"good": 1, "weak": 0},
            "dense": [{"id": "good", "score": 0.8}, {"id": "weak", "score": 0.2}],
            "keyword": [{"id": "weak", "score": 100}],
            "reranker_scores": {"good": 0.8, "weak": 0.2},
        },
        {
            "row_index": 2,
            "example_type": 0,
            "labels": {"irrelevant": 0},
            "dense": [{"id": "irrelevant", "score": 0.2}],
            "keyword": [],
            "reranker_scores": {"irrelevant": 0.2},
        },
    ]
    result = evaluate(
        examples,
        Settings(
            app_env="test",
            chat_relevance_threshold=0.4,
            chat_relative_relevance_threshold=0,
        ),
    )
    assert result["relevance_f1"] == 1
    assert result["confusion"] == {
        "true_positive": 1,
        "false_positive": 0,
        "false_negative": 0,
        "true_negative": 2,
    }
    assert result["negative_abstention"] == 1
    assert result["mean_chunks"] == 0.5
    assert result["empty_rate"] == 0.5
    assert result["details"][0]["chunk_ids"] == ["good"]
    relaxed = evaluate(
        examples,
        Settings(
            app_env="test",
            chat_relevance_threshold=0.1,
            chat_relative_relevance_threshold=0,
        ),
    )
    assert relaxed["relevance_precision"] == 1 / 3
    assert relaxed["negative_abstention"] == 0


def test_recall_f1_harmonic_balances_both_metrics() -> None:
    assert recall_f1_harmonic(1.0, 0.5) == pytest.approx(2 / 3)
    assert recall_f1_harmonic(0.0, 0.5) == 0.0


def test_precision_recall_sweep_enforces_recall_floor_before_maximizing_f1() -> None:
    high_recall = {
        "relevance_f1": 0.44,
        "relevance_recall": 0.98,
        "relevance_precision": 0.28,
        "dense_weight": 1.0,
    }
    balanced = {
        "relevance_f1": 0.46,
        "relevance_recall": 0.74,
        "relevance_precision": 0.33,
        "dense_weight": 1.0,
    }
    assert precision_recall_sort_key(high_recall) > precision_recall_sort_key(balanced)


async def test_batched_collection_matches_retriever_and_resumes_from_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import argparse
    import json

    from app.evaluation.retrieval import EvaluationHybridRetriever, codequery_chunks
    from app.integrations.embeddings.catalog import EmbeddingModel
    from app.integrations.embeddings.mock import MockEmbeddingProvider
    from app.integrations.reranking.mock import MockRerankerProvider
    from scripts import tune_codequeries

    calls: list[list[str]] = []

    class Provider(MockEmbeddingProvider):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__()

        async def embed_documents(self, texts: list[str]) -> list[list[float]]:
            calls.append(texts)
            return [[float(len(text)), 1.0, 2.0] for text in texts]

        async def embed_query(self, text: str) -> list[float]:
            return (await self.embed_documents([text]))[0]

    rows: list[dict[str, Any]] = [
        {
            "__dataset_row_index": i,
            "query_name": "Where is main?",
            "example_type": 1,
            "code_file_path": "repo/main.py",
            "answer_spans": [],
            "context_blocks": [{"content": f"def main(): return {i}", "index": 0}],
        }
        for i in range(2)
    ]
    monkeypatch.setattr(tune_codequeries, "ROOT", tmp_path)
    monkeypatch.setattr(tune_codequeries, "HuggingFaceEmbeddingProvider", Provider)
    monkeypatch.setenv("HF_API_TOKEN", "test-token")
    (tmp_path / "validation-42.json").write_text(json.dumps({"rows": rows, "sampling": {}}))
    settings = Settings(app_env="test")
    args = argparse.Namespace(split="validation", seed=42, count=2)
    await tune_codequeries.collect(args, settings)
    result = cast(dict[str, Any], json.loads((tmp_path / "validation-42-scores.json").read_text()))

    def provider_factory(model: EmbeddingModel) -> Provider:
        return Provider()

    retriever = EvaluationHybridRetriever(
        settings, provider_factory, lambda model_id: MockRerankerProvider(model_id, [1.0])
    )
    for row, actual in zip(rows, result["examples"], strict=True):
        expected = await retriever._dense_records(str(row["query_name"]), codequery_chunks(row))
        assert actual["dense"] == expected
    calls.clear()
    await tune_codequeries.collect(args, settings)
    assert calls == []
