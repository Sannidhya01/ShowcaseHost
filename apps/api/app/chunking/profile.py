from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Protocol

from app.core.config import Settings


class TokenCounter(Protocol):
    id: str

    def count(self, text: str) -> int: ...


class Utf8ByteTokenCounter:
    """Conservative fallback for byte-level tokenizer families."""

    id = "utf8-bytes-v1"

    def count(self, text: str) -> int:
        return len(text.encode("utf-8"))


@dataclass(frozen=True)
class ChunkingProfile:
    version: str
    embedding_input_tokens: int
    tokenizer_id: str
    safety_margin_ratio: float
    target_raw_tokens: int
    max_chunk_size: int
    context_mode: str
    sibling_detail: str
    filter_imports: bool
    overlap_ratio: float
    max_file_bytes: int
    streaming_file_bytes: int
    batch_bytes: int
    concurrency: int
    memory_budget_bytes: int

    @classmethod
    def from_settings(cls, settings: Settings) -> ChunkingProfile:
        counter = Utf8ByteTokenCounter()
        usable = int(
            settings.embedding_input_token_limit * (1 - settings.chunking_safety_margin_ratio)
        )
        raw_target = min(settings.chunking_target_raw_tokens, int(usable * 0.70))
        return cls(
            version=settings.chunking_profile_version,
            embedding_input_tokens=settings.embedding_input_token_limit,
            tokenizer_id=counter.id,
            safety_margin_ratio=settings.chunking_safety_margin_ratio,
            target_raw_tokens=settings.chunking_target_raw_tokens,
            max_chunk_size=max(128, raw_target),
            context_mode="full",
            sibling_detail="signatures",
            filter_imports=False,
            overlap_ratio=settings.chunking_overlap_ratio,
            max_file_bytes=settings.chunking_max_file_bytes,
            streaming_file_bytes=settings.chunking_streaming_file_bytes,
            batch_bytes=settings.chunking_batch_bytes,
            concurrency=settings.chunking_worker_concurrency,
            memory_budget_bytes=settings.chunking_memory_budget_bytes,
        )

    @property
    def key(self) -> str:
        canonical = json.dumps(asdict(self), separators=(",", ":"), sort_keys=True)
        digest = hashlib.sha256(canonical.encode()).hexdigest()[:16]
        return f"{self.version}:{digest}"

    @property
    def options(self) -> dict[str, object]:
        return {
            "maxChunkSize": self.max_chunk_size,
            "contextMode": self.context_mode,
            "siblingDetail": self.sibling_detail,
            "filterImports": self.filter_imports,
            "overlapRatio": self.overlap_ratio,
        }

    @property
    def limits(self) -> dict[str, object]:
        return {
            "embeddingInputTokens": self.embedding_input_tokens,
            "safetyMarginRatio": self.safety_margin_ratio,
            "maxFileBytes": self.max_file_bytes,
            "streamingFileBytes": self.streaming_file_bytes,
            "batchBytes": self.batch_bytes,
            "concurrency": self.concurrency,
            "memoryBudgetBytes": self.memory_budget_bytes,
        }
