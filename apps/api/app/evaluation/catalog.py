from dataclasses import dataclass
from typing import Literal

BenchmarkId = Literal["codequeries", "repoqa"]


@dataclass(frozen=True)
class BenchmarkSplit:
    id: str
    label: str
    config: str
    examples: int
    sampling_groups: tuple[str, ...]


@dataclass(frozen=True)
class Benchmark:
    id: BenchmarkId
    name: str
    dataset_id: str
    default_config: str
    default_split: str
    metric: str
    github_url: str
    dataset_url: str
    paper_url: str
    citation: str
    total_examples: int
    splits: tuple[BenchmarkSplit, ...]


BENCHMARKS: tuple[Benchmark, ...] = (
    Benchmark(
        id="codequeries",
        name="CodeQueries",
        dataset_id="thepurpleowl/codequeries",
        default_config="ideal",
        default_split="test",
        metric="Official retrieval relevance accuracy, precision, recall, and F1",
        github_url="https://github.com/thepurpleowl/codequeries-benchmark",
        dataset_url="https://huggingface.co/datasets/thepurpleowl/codequeries",
        paper_url="https://arxiv.org/abs/2209.08372",
        citation=(
            "Sahu, S. P., Mandal, M., Bharadwaj, S., Kanade, A., Maniatis, P., and Shevade, S. "
            "CodeQueries: A Dataset of Semantic Queries over Code. ISEC 2024."
        ),
        total_examples=171_346,
        splits=(
            BenchmarkSplit("train", "Train", "ideal", 102_962, ("negative", "positive")),
            BenchmarkSplit("validation", "Validation", "ideal", 11_183, ("negative", "positive")),
            BenchmarkSplit("test", "Test", "ideal", 57_201, ("negative", "positive")),
        ),
    ),
    Benchmark(
        id="repoqa",
        name="RepoQA",
        dataset_id="repoqa-2024-06-23",
        default_config="default",
        default_split="all",
        metric="Official generated-answer pass@1 at 0.8, plus retrieval MRR",
        github_url="https://github.com/evalplus/repoqa",
        dataset_url="https://evalplus.github.io/repoqa.html",
        paper_url="https://arxiv.org/abs/2406.06025",
        citation=(
            "Liu, J. et al. RepoQA: Evaluating Long Context Code Understanding. "
            "arXiv:2406.06025, 2024."
        ),
        total_examples=600,
        splits=(
            BenchmarkSplit(
                "all",
                "All languages",
                "default",
                600,
                ("cpp", "go", "java", "python", "rust", "typescript"),
            ),
            BenchmarkSplit("cpp", "C++", "default", 100, ("cpp",)),
            BenchmarkSplit("go", "Go", "default", 100, ("go",)),
            BenchmarkSplit("java", "Java", "default", 100, ("java",)),
            BenchmarkSplit("python", "Python", "default", 100, ("python",)),
            BenchmarkSplit("rust", "Rust", "default", 100, ("rust",)),
            BenchmarkSplit("typescript", "TypeScript", "default", 100, ("typescript",)),
        ),
    ),
)

BENCHMARK_BY_ID = {benchmark.id: benchmark for benchmark in BENCHMARKS}
