from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class EmbeddingModel:
    id: str
    dimension: int
    context_tokens: int
    priority: int
    multilingual: bool = False
    fast: bool = False


# Keep this list limited to models currently deployed for feature extraction by
# Hugging Face Inference. Availability is also probed before each job starts.
EMBEDDING_MODELS: tuple[EmbeddingModel, ...] = (
    EmbeddingModel("BAAI/bge-m3", 1024, 8192, 0, multilingual=True),
    EmbeddingModel("sentence-transformers/all-MiniLM-L6-v2", 384, 256, 1, fast=True),
    EmbeddingModel("intfloat/multilingual-e5-small", 384, 512, 2, multilingual=True, fast=True),
)
MODEL_BY_ID = {model.id: model for model in EMBEDDING_MODELS}


class SelectionInput(Protocol):
    @property
    def language(self) -> str: ...

    @property
    def token_count(self) -> int: ...


class UnsupportedEmbeddingModel(ValueError):
    pass


def require_supported_model(model_id: str) -> EmbeddingModel:
    try:
        return MODEL_BY_ID[model_id]
    except KeyError as exc:
        supported = ", ".join(MODEL_BY_ID)
        raise UnsupportedEmbeddingModel(
            f"Unsupported embedding model {model_id!r}. Choose one of: {supported}"
        ) from exc


def model_candidates(
    inputs: Sequence[SelectionInput],
    *,
    manual_model: str | None = None,
    large_repository_chunks: int = 2_000,
) -> list[EmbeddingModel]:
    if manual_model:
        selected = require_supported_model(manual_model)
        return [selected, *(model for model in EMBEDDING_MODELS if model != selected)]

    if not inputs:
        return list(EMBEDDING_MODELS)

    languages = Counter(item.language.lower() for item in inputs)
    code_chunks = sum(
        count
        for language, count in languages.items()
        if language not in {"markdown", "text", "rst", "asciidoc"}
    )
    code_dominant = code_chunks / len(inputs) >= 0.60
    largest_chunk = max(item.token_count for item in inputs)

    # MiniLM materially reduces latency for high-volume, short, prose-heavy inputs.
    # Long/code chunks stay on BGE-M3 to avoid its 256-token truncation boundary.
    if (
        len(inputs) >= large_repository_chunks
        and not code_dominant
        and largest_chunk <= MODEL_BY_ID["sentence-transformers/all-MiniLM-L6-v2"].context_tokens
    ):
        order: tuple[str, ...] = (
            "sentence-transformers/all-MiniLM-L6-v2",
            "BAAI/bge-m3",
            "intfloat/multilingual-e5-small",
        )
    elif not code_dominant and len(languages) > 1:
        order = (
            "BAAI/bge-m3",
            "intfloat/multilingual-e5-small",
            "sentence-transformers/all-MiniLM-L6-v2",
        )
    else:
        order = tuple(model.id for model in EMBEDDING_MODELS)
    return [MODEL_BY_ID[model_id] for model_id in order]
