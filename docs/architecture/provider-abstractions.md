# Provider Abstractions

ShowcaseHost uses internal protocols for embeddings, reranking, generation, vector storage, and lexical
retrieval:

- `EmbeddingProvider`
- `RerankerProvider`
- `GenerationProvider`
- `VectorStore`
- `KeywordSearch`

Hugging Face feature extraction, Hugging Face cross-encoder reranking, and Qdrant storage are
production adapters. Generation has separate
OpenRouter and Groq adapters selected per catalog model. Vendor clients remain isolated from domain
code. `PgSearchKeywordSearch` is the PostgreSQL/`pg_search` adapter; repository chat depends only on
the provider-neutral `KeywordSearch` protocol.
The reranking adapter accepts query-passage pairs and returns normalized scores; chat and evaluation
depend only on `RerankerProvider`. The default model is `BAAI/bge-reranker-v2-m3`, followed by
`BAAI/bge-reranker-large` when the first provider fails.

Repository ingestion adds two implemented boundaries:

- `SourceControlProvider`, implemented for GitHub App authentication and commit-pinned archives.
- `SnapshotStore`, implemented by private local archive/manifest storage.

Application services depend on these protocols. A future Git provider or S3-compatible snapshot
store must be added as another implementation rather than imported into ingestion services.

Chunking adds a versioned local worker protocol and `EmbeddingInput`. Domain code consumes the
embedding-ready contract rather than importing `code-chunk`; the vendor package remains isolated in
`apps/chunker`. `TokenCounter` allows exact model tokenizers to replace the conservative UTF-8 byte
counter without changing chunk persistence.

Embedding jobs are durable and start automatically after chunking. Auto-selection uses the chunk
language mix, repository volume, and candidate context windows. The fixed, currently HF-hosted list
is `BAAI/bge-m3` (primary), `sentence-transformers/all-MiniLM-L6-v2` (fast/short-context), and
`intfloat/multilingual-e5-small` (multilingual alternative). Each model is probed before work starts;
transient errors use bounded exponential backoff before the next model is attempted.

Qdrant collections are model- and dimension-specific. Point payloads include source text, complete
chunk location/context metadata, model ID, dimension, and a SHA-256 content hash. Stable chunk IDs
plus that hash allow unchanged points to be skipped on later snapshots.

The same stable `CodeChunk.id` UUID is the Qdrant point ID and the `pg_search` index key. Hybrid
retrieval therefore fuses results without path-derived aliases or a second identity mapping.

Set `HF_API_TOKEN` with an Inference Providers-capable Hugging Face token. `HF_TOKEN` remains a
backward-compatible alias. Batch size, worker concurrency, large-repository threshold, timeouts,
retry policy, and runner leases are configuration-driven. Tests use mocks or dependency overrides.
