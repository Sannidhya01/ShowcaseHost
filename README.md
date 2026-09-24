# ShowcaseHost

**Ask a question about a codebase. Get an answer grounded in the code, with file and line citations.**

ShowcaseHost is an AI-powered codebase intelligence project built for questions such as “Where is authentication enforced?” or “How does repository ingestion handle failures?” It parses repositories with source-code structure in mind, retrieves relevant code through semantic and keyword search, and uses the retrieved evidence to produce a cited answer. When the search finds no sufficiently relevant code, the chat says so instead of inventing an explanation.

## Why this project stands out

- **Code-aware indexing:** Repository files are classified by language and split around structural or logical boundaries. The chunking service supports `code-chunk`, offline Tree-sitter grammars, and fallbacks for other text files.
- **Retrieval tuned for source code:** Repository-scoped vector search and BM25 keyword search run in parallel. Reciprocal rank fusion combines the candidates, a reranker orders them, and a relevance threshold limits the evidence sent to the answer model.
- **Verifiable answers:** Streaming responses cite file paths and line ranges from the indexed repository. Queries without qualifying sources return an explicit no-evidence response.
- **Measured retrieval quality:** On a 998-example CodeQueries relevance evaluation, the reported results were **91.3% recall, 87.8% accuracy, and 62.3% F1** (47.3% precision). These measure retrieval relevance, not the factual correctness of generated answers. See the [evaluation guide](docs/development/evaluations.md) and [retrieval tuning notes](docs/development/codequeries-retrieval-tuning.md).
- **Full-stack engineering:** A Next.js and React frontend connects to a FastAPI backend with durable ingestion jobs, database migrations, configurable provider interfaces, automated tests, and a local Docker stack.

## How it works

1. A GitHub App ingestion job fetches a repository snapshot and creates source-aware chunks with file and line metadata.
2. Embeddings and searchable text are stored for repository-scoped semantic and keyword retrieval.
3. A question triggers both searches; fusion, reranking, and relevance filtering select up to five evidence chunks.
4. The chat streams an answer with citations, or reports that it found no relevant evidence.

The implementation is organized as a [web app](apps/web), [API](apps/api), [chunking service](apps/chunker), and [shared contracts](packages/contracts). Architectural choices and provider boundaries are documented in [docs/architecture](docs/architecture). Repository visualizations are a future milestone.

## Tech stack

| Layer           | Technologies                                      |
| --------------- | ------------------------------------------------- |
| Frontend        | Next.js, React, TypeScript, Tailwind CSS          |
| API and jobs    | Python, FastAPI, Pydantic, SQLAlchemy, Alembic    |
| Parsing         | `code-chunk`, Tree-sitter, GitHub Linguist        |
| Search and data | Qdrant, ParadeDB/PostgreSQL with `pg_search`      |
| Quality         | Vitest, React Testing Library, pytest, Ruff, mypy |

## Run locally

Prerequisites: Node.js 22, pnpm 10, Python 3.12, `uv`, and Docker.

```bash
make install
cp .env.example .env
cp apps/web/.env.local.example apps/web/.env.local
cp apps/api/.env.example apps/api/.env
make infra-up
make migrate
```

Start the API and web app in separate terminals:

```bash
make api
make web
```

Open `http://localhost:3000`. The API runs at `http://localhost:8000`. See the [development docs](docs/development) for GitHub App setup, provider configuration, and evaluation commands. Cloud credentials are optional until the corresponding feature is invoked; repository chat requires a completed embedding job and a configured model provider.

Run the full local check with `make check`. This repository contains the implementation and local test setup; it does not include a hosted demo or a claim that every external provider has been tested live.
