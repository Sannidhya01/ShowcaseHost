from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.auth import router as auth_router
from app.api.routes.chat import router as chat_router
from app.api.routes.evaluations import router as evaluations_router
from app.api.routes.health import router as health_router
from app.api.routes.repositories import router as repositories_router
from app.chunking.client import ChunkingWorkerClient
from app.core.config import Settings, get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging
from app.db.session import configure_database, dispose_engine, get_session_factory
from app.evaluation.datasets import BenchmarkDatasetClient
from app.evaluation.service import EvaluationManager, EvaluationRunner
from app.integrations.embeddings.catalog import EmbeddingModel
from app.integrations.embeddings.huggingface import HuggingFaceEmbeddingProvider
from app.integrations.generation.base import GenerationProvider
from app.integrations.generation.catalog import require_generation_model
from app.integrations.generation.groq import GroqGenerationProvider
from app.integrations.generation.openrouter import OpenRouterGenerationProvider
from app.integrations.keyword_search.pg_search import PgSearchKeywordSearch
from app.integrations.reranking.huggingface import HuggingFaceRerankerProvider
from app.integrations.snapshots.local import LocalSnapshotStore
from app.integrations.source_control.github import GitHubSourceControlProvider
from app.integrations.vector_store.qdrant import QdrantVectorStore
from app.services.chunking import ChunkingProcessor, ChunkingRunner
from app.services.embeddings import EmbeddingProcessor, EmbeddingRunner
from app.services.ingestion import IngestionProcessor, IngestionRunner


def create_app(settings: Settings | None = None) -> FastAPI:
    active_settings = settings or get_settings()
    configure_logging(active_settings.log_level)
    configure_database(str(active_settings.database_url))

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        runner: IngestionRunner = app.state.ingestion_runner
        chunking_runner: ChunkingRunner = app.state.chunking_runner
        embedding_runner: EmbeddingRunner = app.state.embedding_runner
        if active_settings.ingestion_runner_enabled:
            runner.start()
        if active_settings.chunking_runner_enabled:
            chunking_runner.start()
        if active_settings.embedding_runner_enabled:
            embedding_runner.start()
        yield
        if active_settings.ingestion_runner_enabled:
            await runner.stop()
        if active_settings.chunking_runner_enabled:
            await chunking_runner.stop()
        if active_settings.embedding_runner_enabled:
            await embedding_runner.stop()
        if hasattr(app.state, "evaluation_manager"):
            await app.state.evaluation_manager.close()
            await app.state.evaluation_dataset_client.close()
        await dispose_engine()

    app = FastAPI(
        title="ShowcaseHost API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.settings = active_settings
    app.state.source_control = GitHubSourceControlProvider(active_settings)
    app.state.snapshot_store = LocalSnapshotStore(active_settings)
    processor = IngestionProcessor(
        get_session_factory(), active_settings, app.state.source_control, app.state.snapshot_store
    )
    app.state.ingestion_runner = IngestionRunner(get_session_factory(), active_settings, processor)
    worker_script = active_settings.chunking_worker_script
    if not worker_script.is_absolute():
        worker_script = (Path(__file__).resolve().parents[1] / worker_script).resolve()
    worker = ChunkingWorkerClient(
        active_settings.chunking_worker_command,
        worker_script,
    )
    chunking_processor = ChunkingProcessor(
        get_session_factory(), active_settings, app.state.snapshot_store, worker
    )
    app.state.chunking_runner = ChunkingRunner(
        get_session_factory(), active_settings, chunking_processor
    )
    token = (
        active_settings.hf_api_token.get_secret_value()
        if active_settings.hf_api_token is not None
        else None
    )

    def embedding_provider(model: EmbeddingModel) -> HuggingFaceEmbeddingProvider:
        return HuggingFaceEmbeddingProvider(
            token,
            model.id,
            model.dimension,
            base_url=active_settings.hf_inference_url,
            timeout_seconds=active_settings.embedding_request_timeout_seconds,
        )

    def reranker_provider(model_id: str) -> HuggingFaceRerankerProvider:
        return HuggingFaceRerankerProvider(
            token,
            model_id,
            base_url=active_settings.hf_inference_url,
            timeout_seconds=active_settings.reranker_request_timeout_seconds,
        )

    groq_key = (
        active_settings.groq_api_key.get_secret_value()
        if active_settings.groq_api_key is not None
        else None
    )
    openrouter_key = (
        active_settings.openrouter_api_key.get_secret_value()
        if active_settings.openrouter_api_key is not None
        else None
    )

    def build_generation_provider(
        model_id: str,
        *,
        max_retries: int,
        max_completion_tokens: int,
        groq_service_tier: str | None = None,
    ) -> GenerationProvider:
        model = require_generation_model(model_id)
        if model.provider == "openrouter":
            return OpenRouterGenerationProvider(
                openrouter_key,
                model_id,
                base_url=active_settings.openrouter_base_url,
                timeout_seconds=active_settings.generation_request_timeout_seconds,
                max_retries=max_retries,
                max_completion_tokens=max_completion_tokens,
                provider_sort=active_settings.openrouter_provider_sort,
                reasoning_effort=active_settings.openrouter_reasoning_effort,
                site_url=active_settings.openrouter_site_url,
                app_name=active_settings.openrouter_app_name,
            )
        return GroqGenerationProvider(
            groq_key,
            model_id,
            base_url=active_settings.groq_base_url,
            timeout_seconds=active_settings.generation_request_timeout_seconds,
            max_retries=max_retries,
            max_completion_tokens=max_completion_tokens,
            service_tier=groq_service_tier,
        )

    def generation_provider(model_id: str) -> GenerationProvider:
        return build_generation_provider(
            model_id,
            max_retries=active_settings.generation_max_retries,
            max_completion_tokens=active_settings.generation_max_completion_tokens,
        )

    app.state.embedding_provider_factory = embedding_provider
    app.state.reranker_provider_factory = reranker_provider
    app.state.generation_provider_factory = generation_provider

    app.state.vector_store = QdrantVectorStore(
        active_settings.qdrant_url,
        active_settings.qdrant_api_key,
        active_settings.qdrant_collection,
    )
    app.state.keyword_search = PgSearchKeywordSearch(
        metadata_boost=active_settings.chat_keyword_metadata_boost
    )
    embedding_processor = EmbeddingProcessor(
        get_session_factory(), active_settings, embedding_provider, app.state.vector_store
    )
    app.state.embedding_runner = EmbeddingRunner(
        get_session_factory(), active_settings, embedding_processor
    )

    if active_settings.evaluation_enabled and active_settings.app_env != "production":

        def evaluation_embedding_provider(model: EmbeddingModel) -> HuggingFaceEmbeddingProvider:
            return HuggingFaceEmbeddingProvider(
                token,
                model.id,
                model.dimension,
                base_url=active_settings.hf_inference_url,
                timeout_seconds=active_settings.evaluation_embedding_timeout_seconds,
            )

        def evaluation_generation_provider(model_id: str) -> GenerationProvider:
            return build_generation_provider(
                model_id,
                max_retries=active_settings.evaluation_max_retries,
                max_completion_tokens=active_settings.evaluation_max_completion_tokens,
                groq_service_tier=active_settings.evaluation_service_tier,
            )

        app.state.evaluation_dataset_client = BenchmarkDatasetClient(
            active_settings.evaluation_hf_api_url,
            active_settings.evaluation_repoqa_data_url,
            active_settings.evaluation_dataset_timeout_seconds,
        )
        app.state.evaluation_manager = EvaluationManager(
            active_settings.evaluation_results_root,
            EvaluationRunner(
                active_settings,
                app.state.evaluation_dataset_client,
                evaluation_generation_provider,
                evaluation_embedding_provider,
                reranker_provider,
            ),
        )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in active_settings.backend_cors_origins],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_exception_handlers(app)
    app.include_router(health_router)
    app.include_router(auth_router)
    app.include_router(chat_router)
    app.include_router(repositories_router)
    if active_settings.evaluation_enabled and active_settings.app_env != "production":
        app.include_router(evaluations_router)
    return app


app = create_app()
