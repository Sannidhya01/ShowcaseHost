"""Persist chunk quality grades and ratings.

Revision ID: 0006_chunk_quality
Revises: 0005_chunk_engine_version_length
Create Date: 2026-09-02
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0006_chunk_quality"
down_revision: str | None = "0005_chunk_engine_version_length"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("code_chunks", sa.Column("quality_grade", sa.String(1), nullable=True))
    op.add_column("code_chunks", sa.Column("quality_score", sa.Integer(), nullable=True))
    op.execute(
        """
        UPDATE code_chunks
        SET quality_grade = CASE context->'parser'->>'name'
                WHEN 'tree-sitter' THEN 'B'
                WHEN 'logical-boundary' THEN 'C'
                WHEN 'plain-text' THEN 'D'
                ELSE 'A'
            END,
            quality_score = CASE context->'parser'->>'name'
                WHEN 'tree-sitter' THEN 75
                WHEN 'logical-boundary' THEN 55
                WHEN 'plain-text' THEN 35
                ELSE 90
            END
        """
    )
    op.alter_column("code_chunks", "quality_grade", nullable=False)
    op.alter_column("code_chunks", "quality_score", nullable=False)


def downgrade() -> None:
    op.drop_column("code_chunks", "quality_score")
    op.drop_column("code_chunks", "quality_grade")
