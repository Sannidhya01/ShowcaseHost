from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, cast

from nltk.translate.bleu_score import (  # type: ignore[import-untyped]
    SmoothingFunction,
    sentence_bleu,
)


def codequery_reference(row: dict[str, Any]) -> list[str]:
    spans = row.get("answer_spans") or []
    values = [str(item.get("span", "")).strip() for item in spans if isinstance(item, dict)]
    values = [value for value in values if value]
    return values or ["N/A"]


def relevance_classification_scores(
    true_positive: int,
    false_positive: int,
    false_negative: int,
    true_negative: int,
) -> dict[str, float]:
    """Match CodeQueries' official positive-class P/R/F1 and overall accuracy."""
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = true_positive / precision_denominator if precision_denominator else 0.0
    recall = true_positive / recall_denominator if recall_denominator else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    total = true_positive + false_positive + false_negative + true_negative
    accuracy = (true_positive + true_negative) / total if total else 0.0
    return {
        "relevance_accuracy": accuracy,
        "relevance_precision": precision,
        "relevance_recall": recall,
        "relevance_f1": f1,
    }


def codequery_relevance_scores(
    gold_labels: dict[str, int], retrieved_ids: Sequence[str]
) -> tuple[dict[str, float], dict[str, int]]:
    """Treat selected retrieval blocks as CodeQueries relevance predictions."""
    predicted_relevant = set(retrieved_ids)
    true_positive = sum(
        label == 1 and chunk_id in predicted_relevant for chunk_id, label in gold_labels.items()
    )
    false_positive = sum(
        label == 0 and chunk_id in predicted_relevant for chunk_id, label in gold_labels.items()
    )
    false_negative = sum(
        label == 1 and chunk_id not in predicted_relevant for chunk_id, label in gold_labels.items()
    )
    true_negative = sum(
        label == 0 and chunk_id not in predicted_relevant for chunk_id, label in gold_labels.items()
    )
    confusion = {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
    }
    return relevance_classification_scores(**confusion), confusion


def repoqa_target_rank(
    retrieved_chunks: Sequence[dict[str, object]],
    *,
    target_path: str,
    target_start_line: int,
    target_end_line: int,
) -> int | None:
    """Return the first rank whose chunk fully contains the target function."""
    for rank, chunk in enumerate(retrieved_chunks, start=1):
        if str(chunk.get("path", "")) != target_path:
            continue
        start = int(cast(str | int, chunk.get("start_line", 0)))
        end = int(cast(str | int, chunk.get("end_line", 0)))
        if start <= target_start_line and end >= target_end_line:
            return rank
    return None


def sanitize_repoqa_output(text: str) -> str:
    blocks = re.findall(r"^```(?:\w+)?\s*\n(.*?)(?=^```)```", text.strip(), re.S | re.M)
    return blocks[0].strip() if blocks else text.strip()


def function_similarity(candidate: str, reference: str) -> float:
    candidate_tokens = re.split(r"\s+", candidate.strip())
    reference_tokens = re.split(r"\s+", reference.strip())
    if not candidate.strip() or not reference.strip():
        return 0.0
    return float(
        sentence_bleu(
            [reference_tokens],
            candidate_tokens,
            smoothing_function=SmoothingFunction().method4,
        )
    )


def repoqa_score(
    response: str, repository: dict[str, Any], ground_truth: str, threshold: float = 0.8
) -> dict[str, Any]:
    candidate = sanitize_repoqa_output(response)
    contents = repository.get("content", {})
    best_name: str | None = None
    best_similarity = 0.0
    for needle in repository.get("needles", []):
        path = str(needle["path"])
        function = "\n".join(
            str(contents[path]).splitlines()[int(needle["start_line"]) : int(needle["end_line"])]
        )
        similarity = function_similarity(candidate, function)
        if similarity > best_similarity:
            best_name, best_similarity = str(needle["name"]), similarity
    passed = best_name == ground_truth and best_similarity >= threshold
    return {
        "pass_at_1": float(passed),
        "best_similarity": best_similarity,
        "best_target": best_name,
        "threshold": threshold,
    }
