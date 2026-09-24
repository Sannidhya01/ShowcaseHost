from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_env: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"
    backend_cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:3000"]
    )

    database_url: str = Field(
        default="postgresql+asyncpg://showcasehost:showcasehost@localhost:5432/showcasehost"
    )

    qdrant_url: str | None = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "code_chunks"
    vector_store_provider: str = "qdrant"

    hf_api_token: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("HF_API_TOKEN", "HF_TOKEN")
    )
    hf_inference_url: str = "https://router.huggingface.co/hf-inference/models"
    embedding_provider: str = "huggingface"
    embedding_input_token_limit: int = 8192
    embedding_runner_enabled: bool = True
    embedding_large_repository_chunks: int = 2_000
    embedding_batch_size: int = 32
    embedding_worker_concurrency: int = 4
    embedding_request_timeout_seconds: float = 60.0
    embedding_max_retries: int = 4
    embedding_backoff_base_seconds: float = 1.0
    embedding_poll_seconds: float = 1.0
    embedding_lease_seconds: int = 600
    embedding_max_attempts: int = 2

    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_fallback_model: str = "BAAI/bge-reranker-large"
    reranker_request_timeout_seconds: float = Field(default=60.0, gt=0, le=600)
    reranker_max_retries: int = Field(default=2, ge=0, le=10)

    groq_api_key: SecretStr | None = None
    groq_base_url: str = "https://api.groq.com/openai/v1"
    openrouter_api_key: SecretStr | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_provider_sort: Literal["price", "throughput", "latency"] = "price"
    openrouter_reasoning_effort: Literal["none", "minimal", "low", "medium", "high"] = "low"
    openrouter_site_url: str = "http://localhost:3000"
    openrouter_app_name: str = "ShowcaseHost"
    generation_request_timeout_seconds: float = 60.0
    generation_max_retries: int = 2
    generation_max_completion_tokens: int = 4096
    generation_provider: str = "openrouter"
    generation_model: str = "openai/gpt-oss-20b"
    chat_retrieval_limit: int = Field(default=5, ge=1, le=5)
    chat_relevance_threshold: float = Field(default=0.00017189, ge=0, le=1, allow_inf_nan=False)
    chat_relative_relevance_threshold: float = Field(
        default=0.855, ge=0, le=1, allow_inf_nan=False
    )
    chat_retrieval_candidate_limit: int = Field(default=32, ge=8, le=128)
    chat_rerank_candidate_limit: int = Field(default=20, ge=5, le=128)
    chat_rrf_k: int = Field(default=60, ge=1)
    chat_dense_weight: float = Field(default=1.0, gt=0)
    chat_keyword_weight: float = Field(default=1.0, gt=0)
    chat_keyword_metadata_boost: float = Field(default=2.0, gt=0)
    chat_question_max_chars: int = 4_000
    chat_history_max_messages: int = 12
    chat_prompt_max_chars: int = 60_000

    evaluation_enabled: bool = False
    evaluation_results_root: Path = Path(".data/evaluations")
    evaluation_hf_api_url: str = "https://datasets-server.huggingface.co"
    evaluation_dataset_timeout_seconds: float = 60.0
    evaluation_max_examples: int = 100
    evaluation_max_concurrency: int = Field(default=8, ge=1, le=64)
    evaluation_requests_per_minute: int = 100
    evaluation_tokens_per_minute: int = 50_000
    evaluation_service_tier: str = "auto"
    evaluation_prompt_max_chars: int = 16_000
    evaluation_max_completion_tokens: int = 4_096
    evaluation_max_retries: int = 8
    evaluation_example_max_retries: int = Field(default=3, ge=0, le=10)
    evaluation_example_retry_delay_seconds: float = Field(default=2.0, ge=0, le=60)
    evaluation_retrieval_concurrency: int = Field(default=2, ge=1, le=16)
    evaluation_embedding_request_concurrency: int = Field(default=4, ge=1, le=16)
    evaluation_embedding_batch_size: int = Field(default=32, ge=1, le=128)
    evaluation_embedding_batch_max_chars: int = Field(default=32_000, ge=1_000, le=250_000)
    evaluation_embedding_timeout_seconds: float = Field(default=180.0, gt=0, le=600)
    evaluation_repoqa_data_url: str = (
        "https://github.com/evalplus/repoqa_release/releases/download/2024-06-23/"
        "repoqa-2024-06-23.json.gz"
    )
    github_token: str | None = None
    github_app_id: int | None = None
    github_app_client_id: str | None = None
    github_app_client_secret: SecretStr | None = None
    github_app_slug: str | None = None
    github_app_private_key: SecretStr | None = None
    github_app_private_key_path: Path | None = None
    github_api_url: str = "https://api.github.com"
    github_web_url: str = "https://github.com"
    github_api_version: str = "2022-11-28"

    api_public_url: str = "http://localhost:8000"
    web_public_url: str = "http://localhost:3000"
    session_cookie_name: str = "showcasehost_session"
    session_ttl_seconds: int = 60 * 60 * 24 * 7
    oauth_flow_ttl_seconds: int = 600

    snapshot_root: Path = Path(".data/snapshots")
    snapshot_retention_count: int = 2
    ingestion_max_archive_bytes: int = 250 * 1024 * 1024
    ingestion_max_extracted_bytes: int = 1024 * 1024 * 1024
    ingestion_max_file_bytes: int = 20 * 1024 * 1024
    ingestion_max_files: int = 100_000
    ingestion_poll_seconds: float = 1.0
    ingestion_lease_seconds: int = 300
    ingestion_max_attempts: int = 3
    ingestion_runner_enabled: bool = True

    chunking_runner_enabled: bool = True
    chunking_profile_version: str = "code-rag-v1"
    chunking_worker_command: str = "node"
    chunking_worker_script: Path = Path("../chunker/dist/index.js")
    chunking_worker_concurrency: int = 8
    chunking_batch_bytes: int = 16 * 1024 * 1024
    chunking_streaming_file_bytes: int = 1024 * 1024
    chunking_max_file_bytes: int = 5 * 1024 * 1024
    chunking_memory_budget_bytes: int = 256 * 1024 * 1024
    chunking_target_raw_tokens: int = 1024
    chunking_overlap_ratio: float = Field(default=0.20, ge=0, le=1)
    chunking_safety_margin_ratio: float = 0.10
    chunking_poll_seconds: float = 1.0
    chunking_lease_seconds: int = 300
    chunking_max_attempts: int = 3

    @field_validator("backend_cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @field_validator(
        "github_app_id",
        "github_app_client_id",
        "github_app_client_secret",
        "github_app_slug",
        "github_app_private_key",
        "github_app_private_key_path",
        "groq_api_key",
        "openrouter_api_key",
        mode="before",
    )
    @classmethod
    def empty_github_values_are_unset(cls, value: object) -> object:
        return None if value == "" else value


@lru_cache
def get_settings() -> Settings:
    return Settings()
