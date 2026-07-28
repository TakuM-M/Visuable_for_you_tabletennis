"""add runpod_job_id to jobs

Revision ID: ce16a51855ec
Revises: c3f5a7b9d2e4
Create Date: 2026-07-28 02:55:56.515987

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'ce16a51855ec'
down_revision: Union[str, None] = 'c3f5a7b9d2e4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # NOTE: autogenerate は ix_jobs_status_started_at / ix_jobs_status_next_retry_at /
    # ix_videos_created_at の削除も出力するが、これらは reaper のクエリ用に別途
    # 作成したインデックスでモデル定義に現れないだけなので消してはいけない。
    op.add_column('jobs', sa.Column('runpod_job_id', sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column('jobs', 'runpod_job_id')
