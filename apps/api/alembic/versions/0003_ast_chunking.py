"""AST-aware chunking schema.

Revision ID: 0003_ast_chunking
Revises: 0002_repository_ingestion
Create Date: 2026-08-01
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0003_ast_chunking"
down_revision: str | None = "0002_repository_ingestion"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

chunking_status = sa.Enum("queued", "running", "succeeded", "failed", name="chunking_status")
chunk_file_status = sa.Enum("succeeded", "skipped", "failed", name="chunk_file_status")


def upgrade() -> None:
    chunking_status.create(op.get_bind())
    chunk_file_status.create(op.get_bind())
    op.create_table(
        "chunking_jobs",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "snapshot_id",
            sa.Uuid(),
            sa.ForeignKey("source_snapshots.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("profile_key", sa.String(128), nullable=False),
        sa.Column(
            "status", postgresql.ENUM(name="chunking_status", create_type=False), nullable=False
        ),
        sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("next_attempt_at", sa.DateTime(timezone=True)),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True)),
        sa.Column("error_code", sa.String(64)),
        sa.Column("error_message", sa.Text()),
        sa.Column("discovered_files", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("supported_files", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("reused_files", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("skipped_files", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("failed_files", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("chunks_created", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("processed_bytes", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("adaptive_retries", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("has_warnings", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("language_timings_ms", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.UniqueConstraint("snapshot_id", "profile_key"),
    )
    op.create_index("ix_chunking_jobs_snapshot_id", "chunking_jobs", ["snapshot_id"])
    op.create_index("ix_chunking_jobs_status", "chunking_jobs", ["status"])
    op.create_index(
        "ix_chunking_jobs_claim",
        "chunking_jobs",
        ["status", "next_attempt_at", "lease_expires_at"],
    )
    op.create_table(
        "chunked_files",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "repository_id",
            sa.Uuid(),
            sa.ForeignKey("repositories.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("path", sa.Text(), nullable=False),
        sa.Column("content_sha256", sa.String(64), nullable=False),
        sa.Column("language", sa.String(32), nullable=False),
        sa.Column("fingerprint", sa.String(64), nullable=False, unique=True),
        sa.Column("engine_version", sa.String(32), nullable=False),
        sa.Column("profile_key", sa.String(128), nullable=False),
        sa.Column("resolved_options", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_chunked_files_repository_id", "chunked_files", ["repository_id"])
    op.create_index("ix_chunked_files_fingerprint", "chunked_files", ["fingerprint"])
    op.create_index("ix_chunked_files_profile_key", "chunked_files", ["profile_key"])
    op.create_table(
        "code_chunks",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "chunked_file_id",
            sa.Uuid(),
            sa.ForeignKey("chunked_files.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("fingerprint", sa.String(64), nullable=False, unique=True),
        sa.Column("raw_text", sa.Text(), nullable=False),
        sa.Column("embedding_text", sa.Text(), nullable=False),
        sa.Column("byte_start", sa.Integer(), nullable=False),
        sa.Column("byte_end", sa.Integer(), nullable=False),
        sa.Column("start_line", sa.Integer(), nullable=False),
        sa.Column("end_line", sa.Integer(), nullable=False),
        sa.Column("context", sa.JSON(), nullable=False),
        sa.Column("tokenizer_id", sa.String(128), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.UniqueConstraint("chunked_file_id", "chunk_index"),
    )
    op.create_index("ix_code_chunks_chunked_file_id", "code_chunks", ["chunked_file_id"])
    op.create_index("ix_code_chunks_fingerprint", "code_chunks", ["fingerprint"])
    op.create_table(
        "snapshot_chunk_files",
        sa.Column(
            "snapshot_id",
            sa.Uuid(),
            sa.ForeignKey("source_snapshots.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("profile_key", sa.String(128), primary_key=True),
        sa.Column("path", sa.Text(), primary_key=True),
        sa.Column(
            "chunked_file_id",
            sa.Uuid(),
            sa.ForeignKey("chunked_files.id", ondelete="RESTRICT"),
        ),
        sa.Column(
            "status", postgresql.ENUM(name="chunk_file_status", create_type=False), nullable=False
        ),
        sa.Column("reason", sa.String(128)),
        sa.Column("error_message", sa.Text()),
    )
    op.create_index(
        "ix_snapshot_chunk_files_chunked_file_id", "snapshot_chunk_files", ["chunked_file_id"]
    )


def downgrade() -> None:
    op.drop_table("snapshot_chunk_files")
    op.drop_table("code_chunks")
    op.drop_table("chunked_files")
    op.drop_table("chunking_jobs")
    chunk_file_status.drop(op.get_bind())
    chunking_status.drop(op.get_bind())
