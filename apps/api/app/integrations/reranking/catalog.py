from dataclasses import dataclass


@dataclass(frozen=True)
class RerankerModel:
    id: str
    priority: int
    context_tokens: int


RERANKER_MODELS: tuple[RerankerModel, ...] = (
    RerankerModel("BAAI/bge-reranker-v2-m3", 0, 8192),
    RerankerModel("BAAI/bge-reranker-large", 1, 512),
)
RERANKER_BY_ID = {model.id: model for model in RERANKER_MODELS}


def require_reranker_model(model_id: str) -> RerankerModel:
    try:
        return RERANKER_BY_ID[model_id]
    except KeyError as exc:
        supported = ", ".join(RERANKER_BY_ID)
        raise ValueError(
            f"Unsupported reranker model {model_id!r}. Choose one of: {supported}"
        ) from exc
