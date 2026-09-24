import json
from pathlib import Path
from typing import Literal

from httpx import ASGITransport, AsyncClient

from app.core.config import Settings
from app.integrations.generation.groq import GroqGenerationProvider
from app.integrations.generation.openrouter import OpenRouterGenerationProvider
from app.main import create_app


class CapturingEvaluationManager:
    def __init__(self) -> None:
        self.request: dict[str, object] | None = None

    async def start(self, request: dict[str, object]) -> dict[str, object]:
        self.request = request
        return {"request": request}


def settings(
    tmp_path: Path, app_env: Literal["test", "production"], evaluation_enabled: bool
) -> Settings:
    return Settings(
        app_env=app_env,
        evaluation_enabled=evaluation_enabled,
        ingestion_runner_enabled=False,
        chunking_runner_enabled=False,
        embedding_runner_enabled=False,
        snapshot_root=tmp_path / "snapshots",
        evaluation_results_root=tmp_path / "evaluations",
    )


async def test_evaluation_routes_are_opt_in_and_never_mounted_in_production(
    tmp_path: Path,
) -> None:
    disabled = create_app(settings(tmp_path, "test", False))
    enabled = create_app(settings(tmp_path, "test", True))
    production = create_app(settings(tmp_path, "production", True))

    assert "/internal/evaluations/benchmarks" not in disabled.openapi()["paths"]
    assert "/internal/evaluations/benchmarks" in enabled.openapi()["paths"]
    assert "/internal/evaluations/models" in enabled.openapi()["paths"]
    assert "/internal/evaluations/runs" in enabled.openapi()["paths"]
    assert "/internal/evaluations/runs/{run_id}/cancel" in enabled.openapi()["paths"]
    assert "/internal/evaluations/benchmarks" not in production.openapi()["paths"]

    await enabled.state.evaluation_dataset_client.close()


async def test_benchmark_catalog_exposes_documentation_and_citations(tmp_path: Path) -> None:
    app = create_app(settings(tmp_path, "test", True))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/internal/evaluations/benchmarks")

    assert response.status_code == 200
    catalog = response.json()
    assert [item["id"] for item in catalog] == ["codequeries", "repoqa"]
    assert all(item["github_url"] and item["dataset_url"] for item in catalog)
    assert all(item["citation"] for item in catalog)
    codequeries = catalog[0]
    assert codequeries["total_examples"] == 171346
    assert {item["id"]: item["examples"] for item in codequeries["splits"]} == {
        "train": 102962,
        "validation": 11183,
        "test": 57201,
    }
    await app.state.evaluation_dataset_client.close()


async def test_evaluation_model_catalog_includes_openrouter_and_groq(tmp_path: Path) -> None:
    app = create_app(settings(tmp_path, "test", True))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/internal/evaluations/models")

    assert response.status_code == 200
    assert [(item["id"], item["provider"]) for item in response.json()] == [
        ("qwen/qwen3.8-27b", "groq"),
        ("openai/gpt-oss-20b", "openrouter"),
    ]
    await app.state.evaluation_dataset_client.close()


async def test_evaluation_run_requires_all_sampling_groups(tmp_path: Path) -> None:
    app = create_app(settings(tmp_path, "test", True))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/internal/evaluations/runs",
            json={
                "benchmark": "codequeries",
                "model": "openai/gpt-oss-20b",
                "split": "test",
                "limit": 1,
            },
        )

    assert response.status_code == 422
    assert response.json()["detail"] == "limit must be at least 2 to include every sampling group"
    await app.state.evaluation_dataset_client.close()


async def test_evaluation_run_persists_a_generated_sampling_seed(tmp_path: Path) -> None:
    app = create_app(settings(tmp_path, "test", True))
    capturing = CapturingEvaluationManager()
    app.state.evaluation_manager = capturing

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/internal/evaluations/runs",
            json={
                "benchmark": "codequeries",
                "model": "openai/gpt-oss-20b",
                "split": "test",
                "limit": 2,
            },
        )

    assert response.status_code == 202
    assert capturing.request is not None
    assert isinstance(capturing.request["seed"], int)
    assert response.json()["request"]["seed"] == str(capturing.request["seed"])
    await app.state.evaluation_dataset_client.close()


async def test_evaluation_seed_roundtrips_without_browser_precision_loss(tmp_path: Path) -> None:
    app = create_app(settings(tmp_path, "test", True))
    capturing = CapturingEvaluationManager()
    app.state.evaluation_manager = capturing
    seed = "9223372036854775807"
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        body = {
            "benchmark": "codequeries",
            "model": "openai/gpt-oss-20b",
            "split": "test",
            "limit": 2,
            "offset": 3,
            "seed": seed,
        }
        first = await client.post("/internal/evaluations/runs", json=body)
        repeated = await client.post("/internal/evaluations/runs", json=first.json()["request"])
        invalid = await client.post("/internal/evaluations/runs", json={**body, "seed": str(2**63)})
    assert first.status_code == repeated.status_code == 202
    assert repeated.json()["request"] == first.json()["request"]
    assert repeated.json()["request"]["seed"] == seed
    assert capturing.request is not None and capturing.request["seed"] == int(seed)
    assert invalid.status_code == 422
    await app.state.evaluation_dataset_client.close()


async def test_retrieval_configuration_reports_effective_server_tuning(tmp_path: Path) -> None:
    configured = settings(tmp_path, "test", True).model_copy(
        update={
            "chat_dense_weight": 1.25,
            "chat_keyword_weight": 0.8,
            "chat_retrieval_candidate_limit": 16,
        }
    )
    app = create_app(configured)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/internal/evaluations/retrieval")
    assert response.status_code == 200
    assert response.json()["dense_weight"] == 1.25
    assert response.json()["keyword_weight"] == 0.8
    assert response.json()["candidate_limit_per_channel"] == 16
    assert response.json()["strategy"] == "hybrid_rrf_cross_encoder_rerank"
    assert response.json()["reranker_model"] == "BAAI/bge-reranker-v2-m3"
    assert response.json()["reranker_fallback_model"] == "BAAI/bge-reranker-large"
    assert response.json()["relative_relevance_threshold"] == 0.855
    await app.state.evaluation_dataset_client.close()


async def test_saved_legacy_seed_is_exact_when_loaded_for_rerun(tmp_path: Path) -> None:
    configured = settings(tmp_path, "test", True)
    configured.evaluation_results_root.mkdir(parents=True)
    path = configured.evaluation_results_root / "legacy-run.json"
    path.write_text(
        json.dumps(
            {
                "id": "legacy-run",
                "created_at": "2026-09-01T10:00:00Z",
                "status": "succeeded",
                "request": {"benchmark": "codequeries", "seed": 2**63 - 1},
                "result": None,
            }
        )
    )
    app = create_app(configured)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        history = await client.get("/internal/evaluations/runs")
        detail = await client.get("/internal/evaluations/runs/legacy-run")
    assert history.json()[0]["request"]["seed"] == "9223372036854775807"
    assert detail.json()["request"]["seed"] == "9223372036854775807"
    assert json.loads(path.read_text())["request"]["seed"] == 2**63 - 1
    await app.state.evaluation_dataset_client.close()


async def test_recovery_route_returns_new_run_and_validates_conflicts(tmp_path: Path) -> None:
    class RecoveryManager:
        async def retry_failed(self, run_id: str) -> dict[str, object] | None:
            if run_id == "missing":
                return None
            if run_id == "changed":
                raise ValueError("Retrieval settings changed")
            return {"id": "recovered", "retry_of": run_id, "request": {"seed": 2**63 - 1}}

    app = create_app(settings(tmp_path, "test", True))
    app.state.evaluation_manager = RecoveryManager()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        recovered = await client.post("/internal/evaluations/runs/old/retry-failed")
        missing = await client.post("/internal/evaluations/runs/missing/retry-failed")
        changed = await client.post("/internal/evaluations/runs/changed/retry-failed")
    assert recovered.status_code == 202
    assert recovered.json()["retry_of"] == "old"
    assert recovered.json()["request"]["seed"] == "9223372036854775807"
    assert missing.status_code == 404
    assert changed.status_code == 409
    await app.state.evaluation_dataset_client.close()


async def test_generation_catalog_routes_models_to_separate_providers(tmp_path: Path) -> None:
    app = create_app(settings(tmp_path, "test", False))

    openrouter = app.state.generation_provider_factory("openai/gpt-oss-20b")
    groq = app.state.generation_provider_factory("qwen/qwen3.8-27b")

    assert isinstance(openrouter, OpenRouterGenerationProvider)
    assert openrouter.provider_sort == "price"
    assert isinstance(groq, GroqGenerationProvider)
    await openrouter.close()
    await groq.close()
