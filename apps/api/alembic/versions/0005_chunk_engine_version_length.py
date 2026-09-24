"""Allow composite chunk-engine version fingerprints.

Revision ID: 0005_chunk_engine_version_length
Revises: 0004_embeddings
Create Date: 2026-08-26
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0005_chunk_engine_version_length"
down_revision: str | None = "0004_embeddings"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.alter_column(
        "chunked_files",
        "engine_version",
        existing_type=sa.String(32),
        type_=sa.String(255),
        existing_nullable=False,
    )


def downgrade() -> None:
    op.alter_column(
        "chunked_files",
        "engine_version",
        existing_type=sa.String(255),
        type_=sa.String(32),
        existing_nullable=False,
    )
