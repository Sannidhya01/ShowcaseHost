from collections import Counter

import httpx
import pytest

from app.evaluation.datasets import BenchmarkDatasetClient, balanced_random_sample, codequery_group


def test_balanced_random_sample_is_equal_random_and_reproducible() -> None:
    rows = [{"id": f"negative-{index}", "example_type": 0} for index in range(20)] + [
        {"id": f"positive-{index}", "example_type": 1} for index in range(20)
    ]

    first = balanced_random_sample(rows, codequery_group, ("negative", "positive"), 10, 42)
    repeated = balanced_random_sample(rows, codequery_group, ("negative", "positive"), 10, 42)
    different = balanced_random_sample(rows, codequery_group, ("negative", "positive"), 10, 43)

    assert Counter(codequery_group(row) for row in first) == {"negative": 5, "positive": 5}
    assert [row["id"] for row in first] == [row["id"] for row in repeated]
    assert [row["id"] for row in first] != [row["id"] for row in different]


def test_balanced_random_sample_distributes_a_remainder_by_at_most_one() -> None:
    languages = ("cpp", "go", "java", "python", "rust", "typescript")
    rows = [
        {"id": f"{language}-{index}", "language": language}
        for language in languages
        for index in range(10)
    ]

    sample = balanced_random_sample(
        rows,
        lambda row: str(row["language"]),
        languages,
        20,
        7,
    )
    counts = Counter(str(row["language"]) for row in sample)

    assert set(counts) == set(languages)
    assert max(counts.values()) - min(counts.values()) == 1


async def test_codequeries_same_seed_and_offset_repeat_exact_ordered_sample() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        start = int(request.url.params["offset"])
        length = int(request.url.params["length"])
        return httpx.Response(
            200,
            json={
                "rows": [
                    {
                        "row_idx": index,
                        "row": {
                            "example_type": index % 2,
                            "query_name": f"query-{index}",
                            "context_blocks": [],
                        },
                    }
                    for index in range(start, start + length)
                ]
            },
        )

    client = BenchmarkDatasetClient("https://datasets.test", "https://repoqa.test", 1)
    await client._hf.aclose()
    client._hf = httpx.AsyncClient(
        base_url="https://datasets.test", transport=httpx.MockTransport(respond)
    )
    try:
        first, metadata = await client.rows("codequeries", "ideal", "test", 5, 10, 2**63 - 1)
        again, repeated = await client.rows("codequeries", "ideal", "test", 5, 10, 2**63 - 1)
        _, changed = await client.rows("codequeries", "ideal", "test", 5, 10, 42)
    finally:
        await client.close()
    assert first == again
    assert metadata == repeated
    assert metadata["sampling_seed"] == str(2**63 + 4)
    assert metadata["sampled_example_ids"] == [str(row["__dataset_row_index"]) for row in first]
    assert metadata["sample_fingerprint"] != changed["sample_fingerprint"]


async def test_codequeries_large_sample_scales_candidate_search_budget() -> None:
    requests = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        length = int(request.url.params["length"])
        example_type = 1 if requests in {2, 5, 8, 11, 13} else 0
        start = requests * length
        return httpx.Response(
            200,
            json={
                "rows": [
                    {
                        "row_idx": index,
                        "row": {
                            "example_type": example_type,
                            "query_name": f"query-{index}",
                            "context_blocks": [],
                        },
                    }
                    for index in range(start, start + length)
                ]
            },
        )

    client = BenchmarkDatasetClient("https://datasets.test", "https://repoqa.test", 1)
    await client._hf.aclose()
    client._hf = httpx.AsyncClient(
        base_url="https://datasets.test", transport=httpx.MockTransport(respond)
    )
    try:
        rows = await client._codequery_rows(
            "thepurpleowl/codequeries", "ideal", "test", 57_201, 1_000, 20_260_916
        )
    finally:
        await client.close()

    assert requests == 13
    assert Counter(codequery_group(row) for row in rows) == {"negative": 500, "positive": 500}


async def test_hugging_face_rows_retry_transient_server_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0
    sleeps: list[float] = []

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(502, text="temporary upstream failure")
        return httpx.Response(
            200,
            json={
                "rows": [
                    {
                        "row_idx": 7,
                        "row": {"example_type": 1, "query_name": "query-7"},
                    }
                ]
            },
        )

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("app.evaluation.datasets.asyncio.sleep", fake_sleep)
    client = BenchmarkDatasetClient("https://datasets.test", "https://repoqa.test", 1)
    await client._hf.aclose()
    client._hf = httpx.AsyncClient(
        base_url="https://datasets.test", transport=httpx.MockTransport(respond)
    )
    try:
        rows = await client._hf_rows("dataset", "ideal", "test", 7, 1)
    finally:
        await client.close()

    assert attempts == 2
    assert sleeps == [1.0]
    assert rows[0]["__dataset_row_index"] == 7
