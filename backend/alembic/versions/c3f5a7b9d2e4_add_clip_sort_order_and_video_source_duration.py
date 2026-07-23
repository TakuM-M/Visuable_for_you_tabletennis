"""add clip sort_order and video source_duration

Revision ID: c3f5a7b9d2e4
Revises: b2e4f6a8c1d3
Create Date: 2026-06-08 00:00:01.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "c3f5a7b9d2e4"
down_revision: Union[str, None] = "b2e4f6a8c1d3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 連結時の並び順。既存行は 0 で初期化する。
    op.add_column(
        "clips",
        sa.Column("sort_order", sa.Integer(), nullable=False, server_default="0"),
    )
    # 元動画の再生時間（書き出し済み output の duration とは別管理）。
    op.add_column(
        "videos",
        sa.Column("source_duration", sa.Float(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("videos", "source_duration")
    op.drop_column("clips", "sort_order")
