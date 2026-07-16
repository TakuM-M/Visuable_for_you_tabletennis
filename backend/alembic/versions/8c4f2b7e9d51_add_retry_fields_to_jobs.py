"""add retry fields to jobs

Revision ID: 8c4f2b7e9d51
Revises: d373a4952a37
Create Date: 2026-05-18 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "8c4f2b7e9d51"
down_revision: Union[str, None] = "d373a4952a37"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "jobs",
        sa.Column("retry_count", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "jobs",
        sa.Column("next_retry_at", postgresql.TIMESTAMP(timezone=True), nullable=True),
    )
    op.add_column(
        "jobs",
        sa.Column(
            "updated_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index("ix_jobs_status_next_retry_at", "jobs", ["status", "next_retry_at"])
    op.create_index("ix_jobs_status_started_at", "jobs", ["status", "started_at"])


def downgrade() -> None:
    op.drop_index("ix_jobs_status_started_at", table_name="jobs")
    op.drop_index("ix_jobs_status_next_retry_at", table_name="jobs")
    op.drop_column("jobs", "updated_at")
    op.drop_column("jobs", "next_retry_at")
    op.drop_column("jobs", "retry_count")
