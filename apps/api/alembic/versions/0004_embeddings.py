"""Embedding generation and storage jobs.

Revision ID: 0004_embeddings
Revises: 0003_ast_chunking
Create Date: 2026-08-05
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0004_embeddings"
down_revision: str | None = "0003_ast_chunking"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

embedding_status = sa.Enum("queued", "running", "succeeded", "failed", name="embedding_status")


def upgrade() -> None:
    op.add_column("ingestion_jobs", sa.Column("embedding_model_override", sa.String(255)))
    embedding_status.create(op.get_bind())
    op.create_table(
        "embedding_jobs",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "snapshot_id",
            sa.Uuid(),
            sa.ForeignKey("source_snapshots.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("profile_key", sa.String(128), nullable=False),
        sa.Column("selection_key", sa.String(255), nullable=False),
        sa.Column("requested_model", sa.String(255)),
        sa.Column("selected_model", sa.String(255)),
        sa.Column("dimension", sa.Integer()),
        sa.Column(
            "status", postgresql.ENUM(name="embedding_status", create_type=False), nullable=False
        ),
        sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("next_attempt_at", sa.DateTime(timezone=True)),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True)),
        sa.Column("total_chunks", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("processed_chunks", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("stored_chunks", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("skipped_chunks", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("fallback_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error_code", sa.String(64)),
        sa.Column("error_message", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.UniqueConstraint("snapshot_id", "profile_key", "selection_key"),
    )
    op.create_index("ix_embedding_jobs_snapshot_id", "embedding_jobs", ["snapshot_id"])
    op.create_index("ix_embedding_jobs_status", "embedding_jobs", ["status"])
    op.create_index(
        "ix_embedding_jobs_claim",
        "embedding_jobs",
        ["status", "next_attempt_at", "lease_expires_at"],
    )


def downgrade() -> None:
    op.drop_table("embedding_jobs")
    embedding_status.drop(op.get_bind())
    op.drop_column("ingestion_jobs", "embedding_model_override")
