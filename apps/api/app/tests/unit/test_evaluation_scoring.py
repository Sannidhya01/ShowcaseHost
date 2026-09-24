from app.evaluation.scoring import (
    codequery_reference,
    codequery_relevance_scores,
    relevance_classification_scores,
    repoqa_score,
)


def test_codequery_reference_is_retained_for_result_display() -> None:
    row = {"answer_spans": [{"span": " first() "}, {"span": "second()"}]}

    assert codequery_reference(row) == ["first()", "second()"]
    assert codequery_reference({"answer_spans": []}) == ["N/A"]


def test_codequery_official_relevance_classification_metrics() -> None:
    scores, confusion = codequery_relevance_scores(
        {"relevant-retrieved": 1, "relevant-missed": 1, "irrelevant-retrieved": 0, "other": 0},
        ["relevant-retrieved", "irrelevant-retrieved"],
    )

    assert confusion == {
        "true_positive": 1,
        "false_positive": 1,
        "false_negative": 1,
        "true_negative": 1,
    }
    assert scores == {
        "relevance_accuracy": 0.5,
        "relevance_precision": 0.5,
        "relevance_recall": 0.5,
        "relevance_f1": 0.5,
    }


def test_codequery_relevance_zero_denominators_match_sklearn_zero_behavior() -> None:
    assert relevance_classification_scores(0, 0, 0, 2) == {
        "relevance_accuracy": 1.0,
        "relevance_precision": 0.0,
        "relevance_recall": 0.0,
        "relevance_f1": 0.0,
    }


def test_repoqa_selects_the_closest_needle_at_threshold() -> None:
    repository = {
        "content": {"example.py": "def one():\n    return 1\n\ndef two():\n    return 2\n"},
        "needles": [
            {"name": "one", "path": "example.py", "start_line": 0, "end_line": 2},
            {"name": "two", "path": "example.py", "start_line": 3, "end_line": 5},
        ],
    }
    result = repoqa_score("```python\ndef two():\n    return 2\n```", repository, "two")

    assert result["pass_at_1"] == 1.0
    assert result["best_target"] == "two"
