"""add ready to videostatus enum

Revision ID: b2e4f6a8c1d3
Revises: a1f3c7d5e2b9
Create Date: 2026-06-08 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "b2e4f6a8c1d3"
down_revision: Union[str, None] = "a1f3c7d5e2b9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ML 解析完了・編集待ち状態を表す ready を videostatus enum に追加する。
    # ALTER TYPE ... ADD VALUE はトランザクション内で実行できないため
    # autocommit_block でラップする。
    with op.get_context().autocommit_block():
        op.execute(
            "ALTER TYPE videostatus ADD VALUE IF NOT EXISTS 'ready' AFTER 'processing'"
        )


def downgrade() -> None:
    # PostgreSQL は enum 値の削除を直接サポートしないため、downgrade は非対応とする。
    pass
