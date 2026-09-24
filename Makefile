PNPM ?= corepack pnpm
UV ?= python3 -m uv
export UV_CACHE_DIR ?= $(CURDIR)/apps/api/.uv-cache

.PHONY: install dev web api chunker-build infra-up infra-down migrate test lint format typecheck check clean

install:
	$(PNPM) install
	cd apps/api && $(UV) sync --all-extras --dev

dev:
	$(MAKE) infra-up
	@printf "Run 'make api' and 'make web' in separate terminals.\n"

web:
	$(PNPM) --filter @showcasehost/web dev

api: chunker-build
	cd apps/api && $(UV) run uvicorn app.main:create_app --factory --reload --host 0.0.0.0 --port 8000

chunker-build:
	$(PNPM) --filter @showcasehost/chunker build

infra-up:
	docker compose up -d postgres qdrant

infra-down:
	docker compose down

migrate:
	cd apps/api && $(UV) run alembic upgrade head

test:
	$(PNPM) test
	cd apps/api && $(UV) run pytest

lint:
	$(PNPM) lint
	cd apps/api && $(UV) run ruff check .

format:
	$(PNPM) format
	cd apps/api && $(UV) run ruff format .

typecheck:
	$(PNPM) typecheck
	cd apps/api && $(UV) run mypy app

check:
	$(PNPM) check
	cd apps/api && $(UV) run ruff check . && $(UV) run mypy app && $(UV) run pytest

clean:
	rm -rf node_modules apps/web/.next apps/web/coverage apps/api/.venv apps/api/.uv-cache apps/api/.pytest_cache apps/api/.ruff_cache apps/api/.mypy_cache
