# ShowcaseHost Agent Notes

ShowcaseHost is a flagship personal project for AI-powered codebase intelligence. It will ingest repositories, parse source structurally, index AST-aware chunks, and answer questions with file and line citations. The current repository is only the production-style foundation.

## Structure

- `apps/web`: Next.js, React, TypeScript frontend.
- `apps/api`: FastAPI modular monolith backend.
- `packages/contracts`: generated or shared API contracts for frontend use.
- `infrastructure/docker`: optional production-like Docker assets.
- `docs`: architecture and development documentation.

## Commands

- Install: `make install`
- Frontend dev: `make web`
- Backend dev: `make api`
- Local infrastructure: `make infra-up` / `make infra-down`
- Migrations: `make migrate`
- Test: `make test`
- Lint: `make lint`
- Format: `make format`
- Type check: `make typecheck`
- Full check: `make check`

## Boundaries

- Provider implementations must remain replaceable.
- Domain and service code must not import vendor SDKs directly.
- Hugging Face, Groq, Qdrant, Neon, and model IDs are configuration choices, not architecture.
- Cloud credentials must stay optional until the related feature is invoked.
- No secrets may be committed.
- New behavior requires tests.
- Avoid unnecessary frameworks, orchestration layers, and abstractions.
- Update documentation when architecture or setup changes.
- Run `make check` before considering a task complete.
- Never claim that an external integration works unless it has been tested or clearly marked as unverified.
