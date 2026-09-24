"""Add pg_search BM25 indexing for hybrid retrieval.

Revision ID: 0007_pg_search_hybrid_retrieval
Revises: 0006_chunk_quality
Create Date: 2026-09-02
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0007_pg_search_hybrid_retrieval"
down_revision: str | None = "0006_chunk_quality"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_search")
    op.add_column("code_chunks", sa.Column("searchable_metadata", sa.Text(), nullable=True))
    op.execute(
        """
        UPDATE code_chunks
        SET searchable_metadata = jsonb_build_object(
            'path', chunked_files.path,
            'language', chunked_files.language,
            'context', code_chunks.context
        )::text
        FROM chunked_files
        WHERE chunked_files.id = code_chunks.chunked_file_id
        """
    )
    op.alter_column("code_chunks", "searchable_metadata", nullable=False)
    op.execute(
        """
        CREATE INDEX code_chunks_bm25_idx
        ON code_chunks
        USING paradedb (
            id,
            (searchable_metadata::pdb.simple),
            (raw_text::pdb.simple),
            quality_score
        )
        WITH (key_field = 'id')
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS code_chunks_bm25_idx")
    op.drop_column("code_chunks", "searchable_metadata")
