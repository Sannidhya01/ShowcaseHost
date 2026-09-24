# Repository Chat

Repository chat is a read-only retrieval-augmented generation flow. It is available only after a
repository snapshot has completed embedding.

1. The API authorizes the repository against the signed-in user.
2. It selects the latest successful embedding job and reuses that job's exact embedding model and
   vector dimension for the query.
3. Dense Qdrant retrieval and PostgreSQL `pg_search` BM25 retrieval run concurrently. Both are
   filtered to the same repository, snapshot, and chunking profile, and both identify results with
   the persisted `code_chunks.id` UUID.
4. Each channel retrieves 32 candidates before fusion. Keyword
   retrieval searches both the chunk's metadata document and raw source text, with a `2.0` metadata
   boost so paths, language, entities, imports, and structural context are primary lexical signals.
5. Reciprocal Rank Fusion merges the two ranked lists with `k = 60`, a `1.0` dense/vector weight,
   and a `1.0` keyword weight.
   A chunk found by both channels is emitted once. The 20 highest fused results enter a
   provider-neutral cross-encoder reranking stage. `BAAI/bge-reranker-v2-m3` scores every
   query-passage pair by default, with `BAAI/bge-reranker-large` configured as the fallback.
   The reranker score sets the final order and relevance cutoff. Up to five survivors enter the prompt;
   persisted chunk quality breaks exact retrieval/fusion ties, preferring complete
   preferred-strategy chunks over fallback chunks.
6. Retrieved chunks are numbered and supplied as untrusted source material with file and one-based
   line metadata.
7. A provider-neutral generation interface streams a cited answer from the model's configured
   provider. The public stream
   emits `sources`, `delta`, `done`, and sanitized `error` server-sent events.

The model catalog uses OpenRouter-hosted `openai/gpt-oss-20b` as the primary default. OpenRouter
receives `provider.sort = "price"`, so it selects the lowest-priced currently viable inference
endpoint and can fall back to the next provider. The existing Groq-hosted `qwen/qwen3.8-27b` option
remains available. The browser sends recent conversation turns with each request; the server
retains only the newest complete turns within configured message and prompt budgets. Chat history
is never written to the database.

The candidate depth, fusion constant, channel weights, and metadata boost are configuration-driven
through `CHAT_RETRIEVAL_CANDIDATE_LIMIT`, `CHAT_RRF_K`, `CHAT_DENSE_WEIGHT`,
`CHAT_KEYWORD_WEIGHT`, `CHAT_KEYWORD_METADATA_BOOST`, and `CHAT_RERANK_CANDIDATE_LIMIT`.
`RERANKER_MODEL` and `RERANKER_FALLBACK_MODEL` select the cross-encoders. The default final
retrieval limit remains `CHAT_RETRIEVAL_LIMIT=5`.

The development evaluation harness consumes these same settings and the same RRF implementation.
Its persisted output includes the effective weights and limits so a benchmark run can be tied to
the retrieval tuning under test.

Source text is data, not instructions. The system prompt denies code mutation claims, limits the
assistant to codebase intelligence, requires inline numbered citations, and directs the model to
admit when retrieved evidence is insufficient.

### Relevance gate

`CHAT_RELEVANCE_THRESHOLD=0.00017189` requires each returned chunk to have at least that normalized
cross-encoder score for the current query. `CHAT_RELATIVE_RELEVANCE_THRESHOLD=0.855` also requires
each chunk to score at least 85.5% as highly as the best candidate for that query. This relative
gate removes weak followers while allowing the absolute score scale to vary across queries. The
score is useful for ordering and calibration but is
not a probability or a validated relevance guarantee. RRF determines which candidates are sent to
the reranker; the reranker then promotes the most relevant passages and decides which ones pass.
This lets strong keyword-only candidates qualify while preserving an absolute relevance cutoff.

Filtering happens across the fused candidate pool before selecting up to
`CHAT_RETRIEVAL_LIMIT` chunks (default and hard maximum: five). The result can be
zero through five chunks; weak matches never fill unused slots. Citations are
numbered after filtering. When no source survives retrieval or the prompt budget,
the chat stream returns an empty sources event, an explicit message that no
relevant information was found, and a done event without invoking generation.
Conversation history cannot override this response.

Chat and the evaluation harness share the gate. Evaluation results record the
absolute and relative thresholds, reranker models, and normalized-score metric. Calibrate them with answerable queries and
unrelated/no-answer queries for each embedding model: inspect accepted chunks,
precision, recall, and empty-result frequency while trying nearby values.
Raise it to reject weak context; lower it if useful evidence is being excluded. BGE reranker scores
are concentrated near zero, so thresholds should be calibrated from observed values rather than
treated as percentages.
