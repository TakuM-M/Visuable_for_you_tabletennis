"""add thumbnail_path to videos

Revision ID: e5a1c9b3f7d2
Revises: ce16a51855ec
Create Date: 2026-07-28 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "e5a1c9b3f7d2"
down_revision: Union[str, None] = "ce16a51855ec"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 既存動画には遡って生成しないため nullable=True。フロントは None を
    # プレースホルダ表示にフォールバックする
    op.add_column("videos", sa.Column("thumbnail_path", sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column("videos", "thumbnail_path")
