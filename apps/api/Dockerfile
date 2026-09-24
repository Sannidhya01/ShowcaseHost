FROM node:22-bookworm-slim AS chunker-build

WORKDIR /workspace
RUN corepack enable
COPY package.json pnpm-lock.yaml pnpm-workspace.yaml ./
COPY apps/chunker/package.json ./apps/chunker/package.json
RUN pnpm install --frozen-lockfile --filter @showcasehost/chunker...
COPY apps/chunker ./apps/chunker
RUN pnpm --filter @showcasehost/chunker build

FROM node:22-bookworm-slim AS node-runtime

FROM python:3.12-slim AS runtime

WORKDIR /app/api
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CHUNKING_WORKER_SCRIPT=/workspace/apps/chunker/dist/index.js

RUN pip install --no-cache-dir uv
COPY --from=node-runtime /usr/local/bin/node /usr/local/bin/node
COPY --from=chunker-build /workspace/node_modules/.pnpm /workspace/node_modules/.pnpm
COPY --from=chunker-build /workspace/apps/chunker/node_modules /workspace/apps/chunker/node_modules
COPY --from=chunker-build /workspace/apps/chunker/dist /workspace/apps/chunker/dist
COPY apps/api/pyproject.toml ./
RUN uv sync --no-dev

COPY apps/api/app ./app
COPY apps/api/alembic ./alembic
COPY apps/api/alembic.ini ./

CMD ["uv", "run", "uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
