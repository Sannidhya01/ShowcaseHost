from dataclasses import dataclass
from typing import Literal

GenerationProviderId = Literal["groq", "openrouter"]


@dataclass(frozen=True)
class GenerationModel:
    id: str
    label: str
    tier: str
    provider: GenerationProviderId
    is_default: bool = False


GENERATION_MODELS: tuple[GenerationModel, ...] = (
    GenerationModel("qwen/qwen3.8-27b", "Qwen 3.8 27B", "faster", "groq"),
    GenerationModel("openai/gpt-oss-20b", "GPT-OSS 20B", "primary", "openrouter", True),
)
MODEL_BY_ID = {model.id: model for model in GENERATION_MODELS}


class UnsupportedGenerationModel(ValueError):
    pass


def require_generation_model(model_id: str) -> GenerationModel:
    try:
        return MODEL_BY_ID[model_id]
    except KeyError as exc:
        supported = ", ".join(MODEL_BY_ID)
        raise UnsupportedGenerationModel(
            f"Unsupported generation model {model_id!r}. Choose one of: {supported}"
        ) from exc
