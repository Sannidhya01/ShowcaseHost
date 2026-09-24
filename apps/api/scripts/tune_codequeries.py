"""Reproducible CodeQueries retrieval experiments; run from apps/api with PYTHONPATH=."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from app.core.config import Settings
from app.evaluation.datasets import (
    balanced_random_sample,
    codequery_group,
    sampling_metadata,
)
from app.evaluation.retrieval import bm25_records, codequery_chunks
from app.evaluation.scoring import codequery_relevance_scores, relevance_classification_scores
from app.integrations.embeddings.huggingface import HuggingFaceEmbeddingProvider
from app.integrations.keyword_search.base import SearchRecord
from app.integrations.reranking.huggingface import HuggingFaceRerankerProvider
from app.services.chat import (
    apply_reranker_scores,
    configured_retrieval_candidates,
    reranker_passage,
)

ROOT = Path(".data/retrieval-tuning")


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, separators=(",", ":")))
    temp.replace(path)


async def sample(args: argparse.Namespace, settings: Settings) -> None:
    import httpx
    import pyarrow.parquet as parquet  # type: ignore[import-not-found]

    async with httpx.AsyncClient(timeout=180, follow_redirects=True) as client:
        response = await client.get(
            f"https://huggingface.co/api/datasets/thepurpleowl/codequeries/parquet/ideal/{args.split}"
        )
        response.raise_for_status()
        urls = response.json()
        rows: list[dict[str, Any]] = []
        columns = [
            "query_name",
            "code_file_path",
            "context_blocks",
            "answer_spans",
            "supporting_fact_spans",
            "example_type",
            "single_hop",
        ]
        for index, url in enumerate(urls):
            path = ROOT / "parquet" / args.split / f"{index}.parquet"
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                temp = path.with_suffix(".download")
                async with client.stream("GET", url) as download:
                    download.raise_for_status()
                    with temp.open("wb") as handle:
                        async for block in download.aiter_bytes():
                            handle.write(block)
                temp.replace(path)
            table = parquet.read_table(path, columns=columns)
            offset = len(rows)
            for row_index, row in enumerate(table.to_pylist()):
                row["__dataset_row_index"] = offset + row_index
                rows.append(row)
            print(f"Loaded {len(rows)} {args.split} rows", flush=True)
        selected = balanced_random_sample(
            rows, codequery_group, ("negative", "positive"), args.count, args.seed
        )
        result = {
            "rows": selected,
            "sampling": sampling_metadata(len(rows), args.seed, "example_type", selected),
            "split": args.split,
            "query_counts": dict(Counter(row["query_name"] for row in selected)),
            "dataset_urls": urls,
        }
        save(ROOT / f"{args.split}-{args.seed}.json", result)
        print(
            json.dumps(
                {
                    "sampled": len(selected),
                    "split": args.split,
                    "queries": len(result["query_counts"]),
                }
            ),
            flush=True,
        )


async def collect(args: argparse.Namespace, settings: Settings) -> None:
    from app.evaluation.retrieval import _cosine
    from app.integrations.vector_store.base import VectorRecord

    sample_data = json.loads((ROOT / f"{args.split}-{args.seed}.json").read_text())
    token = settings.hf_api_token.get_secret_value() if settings.hf_api_token else None
    if not token:
        raise RuntimeError("A live HF_API_TOKEN is required; mock embeddings are not allowed")
    model_id = "BAAI/bge-m3"
    model_dimension = 1024
    inputs: dict[str, str] = {}
    vectors: dict[str, list[float]] = {}
    prepared = []

    def key_for(text: str) -> str:
        key = hashlib.sha256((model_id + text).encode()).hexdigest()
        inputs[key] = text
        return key

    selected_rows = sample_data["rows"][: args.count]
    for row in selected_rows:
        chunks = codequery_chunks(row)
        query_key = key_for(str(row["query_name"]))
        document_keys = [key_for(f"{chunk.metadata}\n{chunk.text}") for chunk in chunks]
        prepared.append((row, chunks, query_key, document_keys))
    for key in inputs:
        for folder in ["vectors", "queries"]:
            path = ROOT / folder / f"{key}.json"
            if path.exists():
                vectors[key] = json.loads(path.read_text())
                break
    missing = [key for key in inputs if key not in vectors]
    batches: list[list[str]] = []
    batch: list[str] = []
    chars = 0
    for key in missing:
        if batch and (
            len(batch) >= settings.evaluation_embedding_batch_size
            or chars + len(inputs[key]) > settings.evaluation_embedding_batch_max_chars
        ):
            batches.append(batch)
            batch = []
            chars = 0
        batch.append(key)
        chars += len(inputs[key])
    if batch:
        batches.append(batch)
    print(
        json.dumps(
            {
                "examples": len(prepared),
                "unique_inputs": len(inputs),
                "cached_inputs": len(vectors),
                "batches": len(batches),
            }
        ),
        flush=True,
    )
    slots = asyncio.Semaphore(settings.evaluation_embedding_request_concurrency)

    async def embed(keys: list[str], attempt: int = 0) -> None:
        try:
            async with slots:
                provider = HuggingFaceEmbeddingProvider(
                    token,
                    model_id,
                    model_dimension,
                    base_url=settings.hf_inference_url,
                    timeout_seconds=settings.evaluation_embedding_timeout_seconds,
                )
                try:
                    batch_vectors = await provider.embed_documents([inputs[key] for key in keys])
                finally:
                    await provider.close()
            for key, vector in zip(keys, batch_vectors, strict=True):
                vectors[key] = vector
                save(ROOT / "vectors" / f"{key}.json", vector)
            print(
                json.dumps({"embedded_inputs": len(vectors), "total_inputs": len(inputs)}),
                flush=True,
            )
        except Exception as error:
            print(
                json.dumps(
                    {"batch_size": len(keys), "attempt": attempt + 1, "error": str(error)[:300]}
                ),
                flush=True,
            )
            if attempt >= 4 or not getattr(error, "retryable", False):
                raise
            await asyncio.sleep(max(2 ** (attempt + 1), getattr(error, "retry_after", 0) or 0))
            if len(keys) > 1:
                middle = len(keys) // 2
                await asyncio.gather(
                    embed(keys[:middle], attempt + 1), embed(keys[middle:], attempt + 1)
                )
            else:
                await embed(keys, attempt + 1)

    outcomes = await asyncio.gather(*(embed(batch) for batch in batches), return_exceptions=True)
    for outcome in outcomes:
        if isinstance(outcome, BaseException):
            raise outcome
    results = []
    for row, chunks, query_key, document_keys in prepared:
        dense = [
            VectorRecord(
                id=chunk.id,
                score=_cosine(vectors[query_key], vectors[key]),
                payload=chunk.payload(),
            )
            for chunk, key in zip(chunks, document_keys, strict=True)
        ]
        results.append(
            {
                "row_index": row["__dataset_row_index"],
                "split": args.split,
                "query": row["query_name"],
                "example_type": row["example_type"],
                "labels": {c.id: c.relevance_label for c in chunks},
                "dense": dense,
                "keyword": bm25_records(
                    str(row["query_name"]),
                    chunks,
                    metadata_boost=settings.chat_keyword_metadata_boost,
                ),
            }
        )
    from app.evaluation.retrieval import evaluation_retrieval_configuration

    save(
        ROOT / f"{args.split}-{args.seed}-scores.json",
        {
            "sampling": sampling_metadata(
                int(sample_data["sampling"].get("available_examples", len(sample_data["rows"]))),
                args.seed,
                "example_type",
                selected_rows,
            ),
            "split": args.split,
            "retrieval": evaluation_retrieval_configuration(settings),
            "embedding_collection": (
                "batched identical inputs across examples; live Hugging Face BGE-M3"
            ),
            "examples": results,
        },
    )
    print(json.dumps({"scored_examples": len(results)}), flush=True)


async def rerank(args: argparse.Namespace, settings: Settings) -> None:
    path = ROOT / f"{args.split}-{args.seed}-scores.json"
    data = json.loads(path.read_text())
    token = settings.hf_api_token.get_secret_value() if settings.hf_api_token else None
    if not token:
        raise RuntimeError("A live HF_API_TOKEN is required; mock reranking is not allowed")
    slots = asyncio.Semaphore(args.concurrency or settings.evaluation_retrieval_concurrency)
    save_lock = asyncio.Lock()
    completed = 0

    async def score_model(model_id: str, query: str, passages: list[str]) -> list[float]:
        last_error: Exception | None = None
        for attempt in range(settings.reranker_max_retries + 1):
            provider = HuggingFaceRerankerProvider(
                token,
                model_id,
                base_url=settings.hf_inference_url,
                timeout_seconds=settings.reranker_request_timeout_seconds,
            )
            try:
                return await provider.score(query, passages)
            except Exception as exc:
                last_error = exc
                if not getattr(exc, "retryable", False) or attempt >= settings.reranker_max_retries:
                    break
                await asyncio.sleep(getattr(exc, "retry_after", None) or 2**attempt)
            finally:
                await provider.close()
        assert last_error is not None
        raise last_error

    async def score_batch(query: str, passages: list[str]) -> tuple[list[float], str]:
        last_error: Exception | None = None
        for model_id in (settings.reranker_model, settings.reranker_fallback_model):
            try:
                return await score_model(model_id, query, passages), model_id
            except Exception as exc:
                last_error = exc
        assert last_error is not None
        raise last_error

    async def score_example(row: dict[str, Any]) -> None:
        nonlocal completed
        records: dict[str, SearchRecord] = {}
        for record in [*row["dense"], *row["keyword"]]:
            records.setdefault(str(record["id"]), record)
        query = str(row["query"])
        scores: dict[str, float] = dict(row.get("reranker_scores", {}))
        models_used: set[str] = set(row.get("reranker_models_used", []))
        missing = [record for chunk_id, record in records.items() if chunk_id not in scores]
        batches: list[list[SearchRecord]] = []
        batch: list[SearchRecord] = []
        chars = 0
        for record in missing:
            passage = reranker_passage(record)
            if batch and (len(batch) >= 16 or chars + len(passage) > 60_000):
                batches.append(batch)
                batch, chars = [], 0
            batch.append(record)
            chars += len(passage)
        if batch:
            batches.append(batch)
        async with slots:
            for items in batches:
                passages = [reranker_passage(record) for record in items]
                values, model_id = await score_batch(query, passages)
                models_used.add(model_id)
                for record, value in zip(items, values, strict=True):
                    scores[str(record["id"])] = value
        row["reranker_scores"] = scores
        row["reranker_models_used"] = sorted(models_used)
        async with save_lock:
            completed += 1
            if completed % 10 == 0 or completed == len(data["examples"]):
                save(path, data)
                print(json.dumps({"reranked_examples": completed}), flush=True)

    await asyncio.gather(*(score_example(row) for row in data["examples"][: args.count]))
    data["retrieval"] = {
        **data["retrieval"],
        "reranker_model": settings.reranker_model,
        "reranker_fallback_model": settings.reranker_fallback_model,
    }
    save(path, data)


def evaluate(examples: list[dict[str, Any]], settings: Settings) -> dict[str, Any]:
    confusion: Counter[str] = Counter()
    empty = 0
    negative_empty = 0
    negatives = 0
    selected_count = 0
    details = []
    for row in examples:
        candidates = configured_retrieval_candidates(row["dense"], row["keyword"], settings)
        reranker_scores = row.get("reranker_scores")
        if not isinstance(reranker_scores, dict):
            raise RuntimeError("Run the rerank stage before sweeping reranker thresholds")
        selected = apply_reranker_scores(
            candidates,
            [float(reranker_scores[str(record["id"])]) for record in candidates],
            settings,
        )
        ids = [r["id"] for r in selected]
        _, counts = codequery_relevance_scores(row["labels"], ids)
        confusion.update(counts)
        empty += not ids
        selected_count += len(ids)
        negatives += row["example_type"] == 0
        negative_empty += row["example_type"] == 0 and not ids
        details.append({"row_index": row["row_index"], "confusion": counts, "chunk_ids": ids})
    return {
        **relevance_classification_scores(**confusion),
        "confusion": dict(confusion),
        "empty_rate": empty / len(examples),
        "negative_abstention": negative_empty / negatives if negatives else 0,
        "mean_chunks": selected_count / len(examples),
        "examples": len(examples),
        "details": details,
    }


def recall_f1_harmonic(recall: float, f1: float) -> float:
    """Balance retrieval recall and F1 while giving recall its own vote."""
    return 2 * recall * f1 / (recall + f1) if recall + f1 else 0.0


def precision_recall_sort_key(
    result: dict[str, Any], min_recall: float = 0.8
) -> tuple[bool, float, float, float, float]:
    """Maximize F1 among settings that retain the required retrieval recall."""
    return (
        float(result["relevance_recall"]) >= min_recall,
        float(result["relevance_f1"]),
        float(result["relevance_recall"]),
        float(result["relevance_precision"]),
        -abs(float(result["dense_weight"]) - 1.0),
    )


def sweep(args: argparse.Namespace, settings: Settings) -> None:
    data = json.loads((ROOT / f"{args.split}-{args.seed}-scores.json").read_text())
    results = []
    ratios = [float(v) for v in args.ratios.split(",")]
    relative_thresholds = (
        [float(value) for value in args.relative_thresholds.split(",")]
        if args.relative_thresholds
        else [settings.chat_relative_relevance_threshold]
    )
    observed_scores = sorted(
        {
            float(score)
            for row in data["examples"]
            for score in row.get("reranker_scores", {}).values()
        }
    )
    if args.thresholds == "exact":
        thresholds = [0.0, *observed_scores]
    elif args.thresholds == "quantiles":
        steps = min(500, max(len(observed_scores) - 1, 1))
        thresholds = sorted(
            {
                0.0,
                *(
                    observed_scores[round(index * (len(observed_scores) - 1) / steps)]
                    for index in range(steps + 1)
                ),
            }
        )
    else:
        thresholds = [float(v) for v in args.thresholds.split(",")]
    for ratio in ratios:
        for threshold in thresholds:
            for relative_threshold in relative_thresholds:
                configured = settings.model_copy(
                    update={
                        "chat_dense_weight": ratio,
                        "chat_keyword_weight": 1.0,
                        "chat_relevance_threshold": threshold,
                        "chat_relative_relevance_threshold": relative_threshold,
                    }
                )
                result = evaluate(data["examples"], configured)
                result.pop("details")
                recall = result["relevance_recall"]
                f1 = result["relevance_f1"]
                result["recall_f1_harmonic"] = recall_f1_harmonic(recall, f1)
                results.append(
                    {
                        "dense_weight": ratio,
                        "keyword_weight": 1.0,
                        "threshold": threshold,
                        "relative_threshold": relative_threshold,
                        **result,
                    }
                )
    results.sort(
        key=lambda result: precision_recall_sort_key(result, args.min_recall), reverse=True
    )
    save(
        ROOT / f"{args.split}-{args.seed}-sweep{args.suffix}.json",
        {"sampling": data["sampling"], "retrieval": data["retrieval"], "results": results},
    )
    print(json.dumps(results[:10], indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["sample", "collect", "rerank", "sweep"])
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--count", type=int, default=600)
    parser.add_argument("--concurrency", type=int, default=0)
    parser.add_argument("--ratios", default="0.01,0.1,0.25,0.5,0.75,1,1.25,1.5,2,3,4,10,100")
    parser.add_argument("--thresholds", default=",".join(str(i / 100) for i in range(0, 81, 5)))
    parser.add_argument("--relative-thresholds", default="")
    parser.add_argument("--min-recall", type=float, default=0.8)
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()
    settings = Settings()
    if args.stage == "sample":
        asyncio.run(sample(args, settings))
    elif args.stage == "collect":
        asyncio.run(collect(args, settings))
    elif args.stage == "rerank":
        asyncio.run(rerank(args, settings))
    else:
        sweep(args, settings)


if __name__ == "__main__":
    main()
