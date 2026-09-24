# Deterministic model evaluations

ShowcaseHost includes an opt-in API for CodeQueries and RepoQA. Both benchmarks run hybrid retrieval
using dense embeddings and metadata-boosted BM25. CodeQueries scores the retrieved blocks directly
without calling a generation model. RepoQA generates an answer using only the retrieved chunks and
then applies its official deterministic answer scorer. No evaluator LLM or judge credential is used.

Set `EVALUATION_ENABLED=true` with `APP_ENV=development` and restart the API. The router is never
mounted when `APP_ENV=production`, even if the flag is set, so its routes are also absent from the
deployed OpenAPI schema. Results are written atomically to
`.data/evaluations/<run-id>.json` by default.

## API

- `GET /internal/evaluations/benchmarks` lists datasets, protocols, links, and citations.
- `GET /internal/evaluations/retrieval` reports current server retrieval weights and budgets.
- `POST /internal/evaluations/runs` starts a background run and returns `202`.
- `GET /internal/evaluations/runs` lists persisted runs.
- `GET /internal/evaluations/runs/{run_id}` returns progress and per-example results.
- `POST /internal/evaluations/runs/{run_id}/cancel` cancels a queued or running evaluation.
- `POST /internal/evaluations/runs/{run_id}/retry-failed` creates a recovery run, preserving
  successful examples and evaluating only failed or missing indices from the original sample.

Example:

```json
{
  "benchmark": "repoqa",
  "model": "qwen/qwen3.8-27b",
  "split": "python",
  "limit": 10,
  "seed": "42"
}
```

The benchmark catalog includes the full dataset size and each available partition. CodeQueries'
`ideal` configuration has 171,346 examples: 102,962 train, 11,183 validation, and 57,201 test.
RepoQA's pinned release has 600 examples, with 100 in each of C++, Go, Java, Python, Rust, and
TypeScript; runs may select one language or all languages. The UI provides these as dropdowns and
caps each run at both the selected partition size and `EVALUATION_MAX_EXAMPLES`. RepoQA is
downloaded lazily and cached in memory, while CodeQueries uses Hugging Face's datasets-server API.
The counts follow the datasets-server [size API](https://huggingface.co/docs/dataset-viewer/en/quick_start).

## Production-aligned hybrid retrieval

CodeQueries context blocks form its retrieval corpus. For RepoQA, every file in the sampled
repository is split into overlapping, line-aligned chunks. All chunks are embedded and ranked before
the top fused results are placed in the generation prompt with the function description.

Dense retrieval and metadata-boosted BM25 run concurrently. Each channel contributes at most
`CHAT_RETRIEVAL_CANDIDATE_LIMIT` candidates, and the harness calls the same Reciprocal Rank Fusion
implementation used by repository chat with `CHAT_RRF_K`, `CHAT_DENSE_WEIGHT`, and
`CHAT_KEYWORD_WEIGHT`. The top `CHAT_RERANK_CANDIDATE_LIMIT` fused candidates are rescored by the
same cross-encoder provider as chat. Only chunks meeting `CHAT_RELEVANCE_THRESHOLD` (default
normalized reranker score 0.00017189) and `CHAT_RELATIVE_RELEVANCE_THRESHOLD` (default 0.855 of the
query's best score) are retained, up to `CHAT_RETRIEVAL_LIMIT` (hard maximum five). The thresholds,
models, and metric are recorded in evaluation results for calibration. CodeQueries scores
them directly; RepoQA supplies them to answer generation. The current defaults are 32 candidates per
channel, 20 rerank candidates, `k=60`, vector weight `1.0`, keyword weight `1.0`, and five final chunks.
The selected sweep and held-out results are documented in
[CodeQueries retrieval tuning](codequeries-retrieval-tuning.md).
Chat and evaluations apply these settings through the same configured fusion function. Tests
exercise CodeQueries with conflicting dense and keyword rankings and verify that changing the
weights changes the ranked evidence under evaluation.

The CodeQueries benchmark corpus is ephemeral and is not inserted into the application's PostgreSQL
or Qdrant collections. The harness therefore computes dense and BM25 rankings over those ephemeral
chunks while sharing the production fusion implementation and every retrieval tuning setting. Document
embeddings are cached by stable corpus hash in the API process, so examples from the same RepoQA
repository reuse the already embedded corpus. Each persisted run and example records
the effective retrieval strategy, embedding model, weights, limits, metadata boost, and selected
chunk IDs, making comparisons reproducible and preventing silent fallback to dataset-provided
context order. Run metadata explicitly labels the in-memory cosine and BM25 backends: these
benchmarks test hybrid ranking and generation, not PostgreSQL/pg_search or Qdrant integration.

## Example failure recovery

The runner completes one fast attempt for every example before starting any example-level retry.
After that first pass, each failed example gets up to three additional attempts
(`EVALUATION_EXAMPLE_MAX_RETRIES=3`) with exponential backoff starting at two seconds
(`EVALUATION_EXAMPLE_RETRY_DELAY_SECONDS=2`). Provider `Retry-After` delays are honored. Only failed
examples are repeated; successful examples retain their responses and scores. Nested retrieval
errors are unwrapped and saved with the attempt history, and the UI displays retrying and failed
counts. Cancelling stops outstanding attempts. Retrieval is separately bounded to two concurrent
examples (`EVALUATION_RETRIEVAL_CONCURRENCY=2`). Up to four embedding requests run concurrently
across those examples (`EVALUATION_EMBEDDING_REQUEST_CONCURRENCY=4`). Each request contains at most 32 inputs
(`EVALUATION_EMBEDDING_BATCH_SIZE=32`) and approximately 32,000 source characters
(`EVALUATION_EMBEDDING_BATCH_MAX_CHARS=32000`), with a 180-second request timeout
(`EVALUATION_EMBEDDING_TIMEOUT_SECONDS=180`). The dual limit packs many small CodeQueries blocks
efficiently while keeping CodeQueries requests within the Hugging Face inference service's practical
request envelope; these evaluation-specific settings do not alter chat retrieval. After a
transient document-embedding failure, that corpus's retry
uses half the batch size and resumes after any already embedded batches. Thus adaptive sizing does
not repeat successful embedding work.

The run record is persisted as `running` immediately. As soon as dataset sampling finishes, the API
also persists the total example count and an empty initial result.

The run waits for every example to succeed or exhaust its retries before reporting final scores.
If failures remain, the status is `completed_with_errors`, and metrics explicitly cover successful
examples only. The UI reports score coverage; such a run is not a complete benchmark evaluation.

Use **Retry failed examples** on a saved failed/partial run to recover missing examples without
repeating successful generations. Recovery creates a new run linked by `retry_of`, verifies the
saved sample fingerprint, and requires matching retrieval settings. The original record remains
available. Use **Rerun same examples** to evaluate the full sample after changing weights instead.

## Balanced random subsets

Every run uses a random subset rather than the first `limit` rows. The generated sampling seed is
stored in the run request and can be supplied explicitly as `seed` to reproduce the same subset.
The UI accepts an optional sampling seed and shows each saved run's seed and retrieval weights.
After tuning `CHAT_DENSE_WEIGHT` / `CHAT_KEYWORD_WEIGHT` and restarting the API, select **Rerun same
examples** on a completed, failed, or cancelled run. It preserves that run's benchmark, config,
split, offset, example count, seed, and model while using the current server retrieval settings.
Leave the seed field blank on a new run to draw a fresh sample.

Seeds are returned as decimal strings so the browser preserves all 63 bits; the API also accepts
integer seeds. Legacy saved integer seeds are converted to strings on read. Runs without a seed
cannot be repeated exactly through this button. Samples record ordered example IDs and a sample
fingerprint for comparing runs. Reproducibility assumes unchanged upstream dataset contents and
sampling code; a seed controls sampling, not model generation randomness. The effective sampling
seed is the request seed plus its offset for both benchmarks.
CodeQueries is sampled equally across its documented negative (`example_type=0`) and positive
(`example_type=1`) examples. RepoQA's `all` split is sampled equally across its six languages; a
single-language split is shuffled within that language. If `limit` is not evenly divisible by the
number of groups, group counts differ by at most one. A run must request at least one example per
group: two for CodeQueries and six for RepoQA `all`.

## Scoring protocols

- **CodeQueries retrieval:** the retrieved blocks are treated as positive relevance predictions;
  every unselected corpus block is a negative prediction. Gold relevance is derived from the
  official answer and supporting-fact annotations and their source intervals. The run reports only
  the official relevance evaluator's overall accuracy and positive-class precision, recall, and F1.
  Recall@5, MRR, span-generation scores, and generated answers are not calculated. This path never
  invokes a generation provider.
- **RepoQA Search Needle Function:** the repository is chunked, embedded, and retrieved before the
  model generates from the selected chunks. The official answer metric extracts the first fenced code block,
  computes the benchmark's smoothed token BLEU against every needle function in that repository,
  and passes only when the gold function is the closest match and similarity is at least the
  official default `0.8`. Runs report official pass@1 and retrieval MRR. MRR is the reciprocal rank
  of the first retrieved chunk that fully contains the target function, averaged over recovered
  targets; Hit@5 is not calculated. Keep RepoQA runs measured because answer
  generation still consumes the configured model provider's quota. Cancelling preserves all results
  completed before cancellation.

## High-throughput, rate-limit-safe runs

Evaluation requests share `EVALUATION_MAX_CONCURRENCY` generation slots across all active runs;
the default is eight. CodeQueries retrieval concurrency, embedding batch size, and embedding request
timeout are controlled independently by `EVALUATION_RETRIEVAL_CONCURRENCY`,
`EVALUATION_EMBEDDING_REQUEST_CONCURRENCY`, `EVALUATION_EMBEDDING_BATCH_SIZE`,
`EVALUATION_EMBEDDING_BATCH_MAX_CHARS`, and
`EVALUATION_EMBEDDING_TIMEOUT_SECONDS`. The default GPT-OSS
model routes through OpenRouter with providers sorted by
price. OpenRouter does not publish a stable RPM/TPM ceiling for paid requests routed across
providers, and its current-key API marks the returned `rate_limit` field as deprecated. OpenRouter
runs therefore use all available local slots and treat upstream HTTP 429 plus `Retry-After` as the
authoritative backpressure signal. This avoids the former 50,000 estimated-token local bottleneck.

Groq remains on the existing provider-neutral rolling-window scheduler: 100 requests/minute and
50,000 estimated tokens/minute by default. The estimate conservatively treats two characters as
one token because source code is denser than prose. All providers retain the 16,000 prompt-character
and 4,096 completion-token request bounds. Relevant settings are:

- `EVALUATION_MAX_CONCURRENCY`
- `EVALUATION_REQUESTS_PER_MINUTE`
- `EVALUATION_TOKENS_PER_MINUTE`
- `EVALUATION_SERVICE_TIER` (Groq models only)
- `EVALUATION_PROMPT_MAX_CHARS`
- `EVALUATION_MAX_COMPLETION_TOKENS`
- `EVALUATION_MAX_RETRIES`

On HTTP 429, both provider adapters honor retry guidance instead of immediately failing. Each
completed example and the current aggregate are persisted, so progress remains
visible throughout a longer run. Start with 50 examples, then repeat with 100 if the score needs a
wider sample. Successful RepoQA records retain both `candidate_answer` and `raw_response`; empty
model streams are retried and fail explicitly rather than being scored as blank output.
CodeQueries records contain retrieval evidence only.

## Documentation and citations

### CodeQueries

CodeQueries provides semantic code queries with exact answer and supporting-fact spans. See the
[benchmark repository](https://github.com/thepurpleowl/codequeries-benchmark),
[dataset card](https://huggingface.co/datasets/thepurpleowl/codequeries), and
[paper](https://arxiv.org/abs/2209.08372).

> Sahu, S. P., Mandal, M., Bharadwaj, S., Kanade, A., Maniatis, P., and Shevade, S. CodeQueries: A
> Dataset of Semantic Queries over Code. ISEC 2024.

### RepoQA

The pinned RepoQA release contains 600 Search Needle Function tests spanning six languages. Its
official evaluator selects the syntactically closest function and applies a configurable
similarity threshold whose default is `0.8`. See the
[benchmark repository](https://github.com/evalplus/repoqa),
[project page](https://evalplus.github.io/repoqa.html), and
[paper](https://arxiv.org/abs/2406.06025).

> Liu, J. et al. RepoQA: Evaluating Long Context Code Understanding. arXiv:2406.06025, 2024.
