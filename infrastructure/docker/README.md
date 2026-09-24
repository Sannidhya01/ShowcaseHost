# Docker Infrastructure

The root `docker-compose.yml` starts ParadeDB's PostgreSQL 16 image (including the `pg_search`
extension) and Qdrant. The frontend and backend are run directly during active development so
reload and tooling stay fast. Existing plain-PostgreSQL volumes should be backed up before switching
images. Compose starts PostgreSQL with `pg_search` in `shared_preload_libraries`; `make migrate`
enables the extension and creates the chunk BM25 index.
