# Local Setup

1. Install Node.js 22, pnpm 10, Python 3.12, uv, and Docker.
2. Run `make install`.
3. Copy environment examples:

```bash
cp .env.example .env
cp apps/web/.env.local.example apps/web/.env.local
cp apps/api/.env.example apps/api/.env
```

4. Start local infrastructure with `make infra-up`.
5. Run the API with `make api`. This first builds the Node chunking companion.
6. Run the frontend with `make web`.

Local PostgreSQL and Qdrant use Docker volumes so data persists between restarts.

## GitHub App setup

Repository ingestion remains dormant when GitHub settings are absent. To exercise it, create a
GitHub App with read-only Repository Contents and Metadata permissions. Configure:

- Homepage URL: `http://localhost:3000`
- Callback URL: `http://localhost:8000/auth/github/callback`
- Setup URL: `http://localhost:8000/github/install/setup`

Set `GITHUB_APP_ID`, `GITHUB_APP_CLIENT_ID`, `GITHUB_APP_CLIENT_SECRET`, `GITHUB_APP_SLUG`, and
`GITHUB_APP_PRIVATE_KEY_PATH` in `apps/api/.env`. The private key file must remain outside version
control. Run `make migrate` before starting the API.

Live public/private ingestion is unverified until a locally configured GitHub App completes a smoke
test. The automated suite uses mocked GitHub responses and does not require credentials.

## Generation provider setup

Set `OPENROUTER_API_KEY` in `apps/api/.env` to use the default `openai/gpt-oss-20b` model.
`OPENROUTER_PROVIDER_SORT=price` explicitly selects OpenRouter's least-expensive viable provider at
request time; provider fallback remains enabled. `OPENROUTER_REASONING_EFFORT=low` limits paid
reasoning tokens for interactive chat and deterministic evaluations. `GROQ_API_KEY` and the Groq
adapter remain independently available for Groq-backed catalog models. Never commit either key.
See the [official GPT-OSS 20B model documentation](https://developers.openai.com/api/docs/models/gpt-oss-20b)
and [OpenRouter provider-routing documentation](https://openrouter.ai/docs/guides/routing/provider-selection).

## AST chunking

`make install` installs the pinned `code-chunk`, Tree-sitter runtime, offline grammar bundle, and
language metadata packages. Chunking runs automatically after a
snapshot succeeds. Run `make migrate` after upgrading an existing database. Operational settings,
including the embedding input limit and worker size/concurrency bounds, are documented in
`apps/api/.env.example`.

The automated Node fixtures verify the six supported language grammars. A production-image smoke
test can be run with `docker build -f apps/api/Dockerfile .`; live embedding and vector-store calls
remain outside this stage.
