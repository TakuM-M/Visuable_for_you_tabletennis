"""add videos created_at index

Revision ID: a1f3c7d5e2b9
Revises: 8c4f2b7e9d51
Create Date: 2026-05-18 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = 'a1f3c7d5e2b9'
down_revision: Union[str, None] = '8c4f2b7e9d51'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index('ix_videos_created_at', 'videos', ['created_at'])


def downgrade() -> None:
    op.drop_index('ix_videos_created_at', table_name='videos')
