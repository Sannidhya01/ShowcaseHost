from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
import math
import random
from collections import Counter
from collections.abc import Callable
from typing import Any, cast

import httpx

from app.evaluation.catalog import BENCHMARK_BY_ID, BenchmarkId


class DatasetLoadError(RuntimeError):
    pass


class BenchmarkDatasetClient:
    def __init__(
        self,
        hf_api_url: str,
        repoqa_data_url: str,
        timeout_seconds: float,
    ) -> None:
        self._hf = httpx.AsyncClient(base_url=hf_api_url, timeout=timeout_seconds)
        self._downloads = httpx.AsyncClient(timeout=timeout_seconds, follow_redirects=True)
        self._repoqa_data_url = repoqa_data_url
        self._repoqa_cache: list[dict[str, Any]] | None = None

    async def rows(
        self,
        benchmark_id: BenchmarkId,
        config: str,
        split: str,
        offset: int,
        limit: int,
        seed: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        benchmark = BENCHMARK_BY_ID[benchmark_id]
        selected_split = next(item for item in benchmark.splits if item.id == split)
        sampling_seed = seed + offset
        if benchmark_id == "repoqa":
            rows = await self._repoqa_rows(split)
            sampled = balanced_random_sample(
                rows,
                lambda row: str(row["language"]),
                selected_split.sampling_groups,
                limit,
                sampling_seed,
            )
            return sampled, sampling_metadata(len(rows), sampling_seed, "language", sampled)
        sampled = await self._codequery_rows(
            benchmark.dataset_id,
            config,
            split,
            selected_split.examples,
            limit,
            sampling_seed,
        )
        return sampled, sampling_metadata(
            selected_split.examples, sampling_seed, "example_type", sampled
        )

    async def _codequery_rows(
        self,
        dataset: str,
        config: str,
        split: str,
        available: int,
        limit: int,
        seed: int,
    ) -> list[dict[str, Any]]:
        requested = limit
        groups = ("negative", "positive")
        targets = balanced_targets(groups, requested, seed)
        candidates: dict[str, list[dict[str, Any]]] = {group: [] for group in groups}
        seen: set[int] = set()
        rng = random.Random(seed)
        batch_size = min(100, max(32, requested * 2))
        max_batches = max(12, math.ceil(requested / batch_size) * 4)
        max_start = max(available - batch_size, 0)
        for _ in range(max_batches):
            batch_offset = rng.randint(0, max_start) if max_start else 0
            batch = await self._hf_rows(dataset, config, split, batch_offset, batch_size)
            for row in batch:
                row_index = int(row.get("__dataset_row_index", -1))
                if row_index in seen:
                    continue
                seen.add(row_index)
                group = "positive" if int(row.get("example_type", 0)) == 1 else "negative"
                candidates[group].append(row)
            if all(len(candidates[group]) >= target for group, target in targets.items()):
                break
        if not all(len(candidates[group]) >= target for group, target in targets.items()):
            raise DatasetLoadError(
                "Could not collect a balanced positive/negative CodeQueries sample"
            )
        sampled = balanced_random_sample(
            [row for group in groups for row in candidates[group]],
            codequery_group,
            groups,
            requested,
            seed,
        )
        return sampled

    async def _hf_rows(
        self, dataset: str, config: str, split: str, offset: int, limit: int
    ) -> list[dict[str, Any]]:
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                response = await self._hf.get(
                    "/rows",
                    params={
                        "dataset": dataset,
                        "config": config,
                        "split": split,
                        "offset": offset,
                        "length": limit,
                    },
                )
                response.raise_for_status()
                payload = response.json()
                if payload.get("partial"):
                    raise DatasetLoadError(
                        f"Hugging Face returned a partial response for {dataset}"
                    )
                if any(item.get("truncated_cells") for item in payload.get("rows", [])):
                    raise DatasetLoadError(f"Hugging Face truncated benchmark cells for {dataset}")
                rows: list[dict[str, Any]] = []
                for item in payload.get("rows", []):
                    row = dict(item["row"])
                    row["__dataset_row_index"] = item.get("row_idx", -1)
                    rows.append(row)
                return rows
            except DatasetLoadError:
                raise
            except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
                last_error = exc
                status = (
                    exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
                )
                retryable = status is None or status == 429 or status >= 500
                if not retryable or attempt == 3:
                    break
                await asyncio.sleep(float(2**attempt))
        raise DatasetLoadError(
            f"Could not load {dataset}/{config}/{split} from Hugging Face"
        ) from last_error

    async def _repoqa_rows(self, split: str) -> list[dict[str, Any]]:
        if self._repoqa_cache is None:
            try:
                response = await self._downloads.get(self._repoqa_data_url)
                response.raise_for_status()
                raw = await asyncio.to_thread(gzip.decompress, response.content)
                dataset = cast(dict[str, list[dict[str, Any]]], json.loads(raw))
            except (httpx.HTTPError, OSError, TypeError, ValueError) as exc:
                raise DatasetLoadError("Could not load the official RepoQA release") from exc
            flattened: list[dict[str, Any]] = []
            for language, repositories in dataset.items():
                for repository in repositories:
                    for needle in repository.get("needles", []):
                        flattened.append(
                            {"language": language, "repository": repository, "needle": needle}
                        )
            self._repoqa_cache = flattened
        if split == "all":
            return self._repoqa_cache
        selected = [row for row in self._repoqa_cache if row["language"] == split]
        if not selected:
            raise DatasetLoadError(f"Unknown RepoQA language split: {split}")
        return selected

    async def close(self) -> None:
        await self._hf.aclose()
        await self._downloads.aclose()


# Compatibility alias for code that imported the original client name.
HuggingFaceDatasetClient = BenchmarkDatasetClient


def codequery_group(row: dict[str, Any]) -> str:
    return "positive" if int(row.get("example_type", 0)) == 1 else "negative"


def balanced_targets(groups: tuple[str, ...], count: int, seed: int) -> dict[str, int]:
    order = list(groups)
    random.Random(seed).shuffle(order)
    return {
        group: count // len(groups) + int(group in order[: count % len(groups)]) for group in groups
    }


def balanced_random_sample(
    rows: list[dict[str, Any]],
    group_for: Callable[[dict[str, Any]], str],
    expected_groups: tuple[str, ...],
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {group: [] for group in expected_groups}
    for row in rows:
        group = group_for(row)
        if group in grouped:
            grouped[group].append(row)
    targets = balanced_targets(expected_groups, count, seed)
    if any(len(grouped[group]) < target for group, target in targets.items()):
        raise DatasetLoadError("The selected partition cannot provide a balanced random sample")
    rng = random.Random(seed)
    order = list(expected_groups)
    rng.shuffle(order)
    for group_rows in grouped.values():
        rng.shuffle(group_rows)
    selected = {group: grouped[group][: targets[group]] for group in expected_groups}
    result: list[dict[str, Any]] = []
    while len(result) < count:
        for group in order:
            if selected[group]:
                result.append(selected[group].pop())
    return result


def sampling_metadata(
    available: int,
    seed: int,
    dimension: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    group_for = codequery_group if dimension == "example_type" else lambda row: str(row["language"])
    example_ids = [
        str(row["__dataset_row_index"])
        if "__dataset_row_index" in row
        else json.dumps(
            [
                row.get("language"),
                row.get("repository", {}).get("repo"),
                row.get("repository", {}).get("commit_sha"),
                row.get("needle", {}).get("name"),
            ],
            separators=(",", ":"),
        )
        for row in rows
    ]
    return {
        "available_examples": available,
        "sampling_seed": str(seed),
        "sampling_strategy": f"balanced_random_by_{dimension}",
        "sampled_groups": dict(Counter(group_for(row) for row in rows)),
        "sampled_example_ids": example_ids,
        "sample_fingerprint": hashlib.sha256(
            json.dumps(example_ids, separators=(",", ":")).encode()
        ).hexdigest(),
    }
