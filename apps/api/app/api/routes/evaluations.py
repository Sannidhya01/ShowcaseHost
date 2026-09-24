import secrets
from typing import Annotated, cast

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

from app.evaluation.catalog import BENCHMARK_BY_ID, BENCHMARKS, BenchmarkId
from app.evaluation.retrieval import evaluation_retrieval_configuration
from app.evaluation.service import EvaluationManager, RunRecord
from app.integrations.generation.catalog import (
    GENERATION_MODELS,
    UnsupportedGenerationModel,
    require_generation_model,
)

router = APIRouter(prefix="/internal/evaluations", tags=["internal-evaluations"])


class EvaluationRunRequest(BaseModel):
    benchmark: BenchmarkId
    model: str
    config: str | None = None
    split: str | None = None
    offset: Annotated[int, Field(ge=0)] = 0
    limit: Annotated[int, Field(ge=1)] = 10
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1)


def manager(request: Request) -> EvaluationManager:
    return cast(EvaluationManager, request.app.state.evaluation_manager)


def public_run(record: RunRecord) -> RunRecord:
    """Keep full-width seeds exact in JavaScript, including legacy persisted runs."""
    result = dict(record)
    if isinstance(record.get("request"), dict):
        result["request"] = dict(record["request"])
        if result["request"].get("seed") is not None:
            result["request"]["seed"] = str(result["request"]["seed"])
    return result


@router.get("/retrieval")
async def retrieval_configuration(request: Request) -> dict[str, object]:
    return evaluation_retrieval_configuration(request.app.state.settings)


@router.get("/benchmarks")
async def benchmarks(request: Request) -> list[dict[str, object]]:
    return [
        {
            **benchmark.__dict__,
            "splits": [split.__dict__ for split in benchmark.splits],
            "max_run_examples": request.app.state.settings.evaluation_max_examples,
        }
        for benchmark in BENCHMARKS
    ]


@router.get("/models")
async def models() -> list[dict[str, object]]:
    return [model.__dict__ for model in GENERATION_MODELS]


@router.post("/runs", status_code=status.HTTP_202_ACCEPTED)
async def create_run(body: EvaluationRunRequest, request: Request) -> RunRecord:
    settings = request.app.state.settings
    benchmark = BENCHMARK_BY_ID[body.benchmark]
    selected_split = next(
        (item for item in benchmark.splits if item.id == (body.split or benchmark.default_split)),
        None,
    )
    if selected_split is None or (body.config and body.config != selected_split.config):
        raise HTTPException(status_code=422, detail="Unsupported benchmark split or config")
    minimum = len(selected_split.sampling_groups)
    if body.limit < minimum:
        raise HTTPException(
            status_code=422,
            detail=f"limit must be at least {minimum} to include every sampling group",
        )
    maximum = min(settings.evaluation_max_examples, selected_split.examples)
    if body.limit > maximum:
        raise HTTPException(
            status_code=422,
            detail=f"limit must be at most {maximum} for the selected split and offset",
        )
    try:
        require_generation_model(body.model)
    except UnsupportedGenerationModel as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    run_request = body.model_dump()
    run_request["config"] = selected_split.config
    run_request["split"] = selected_split.id
    run_request["seed"] = body.seed if body.seed is not None else secrets.randbits(63)
    return public_run(await manager(request).start(run_request))


@router.get("/runs")
async def list_runs(request: Request) -> list[RunRecord]:
    return [public_run(record) for record in await manager(request).list()]


@router.get("/runs/{run_id}")
async def get_run(run_id: str, request: Request) -> RunRecord:
    result = await manager(request).get(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Evaluation run not found")
    return public_run(result)


@router.post("/runs/{run_id}/cancel")
async def cancel_run(run_id: str, request: Request) -> RunRecord:
    try:
        result = await manager(request).cancel(run_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if result is None:
        raise HTTPException(status_code=404, detail="Evaluation run not found")
    return public_run(result)


@router.post("/runs/{run_id}/retry-failed", status_code=status.HTTP_202_ACCEPTED)
async def retry_failed_run(run_id: str, request: Request) -> RunRecord:
    try:
        result = await manager(request).retry_failed(run_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if result is None:
        raise HTTPException(status_code=404, detail="Evaluation run not found")
    return public_run(result)
