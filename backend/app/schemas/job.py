import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.models.job import JobStatus


class JobResponse(BaseModel):
    """ジョブ情報レスポンス（ジョブはシステム側で作成するためCreateスキーマは不要）"""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    video_id: uuid.UUID
    status: JobStatus
    started_at: datetime | None
    completed_at: datetime | None
    error_message: str | None
    retry_count: int
    next_retry_at: datetime | None
    created_at: datetime
    updated_at: datetime


class ClipData(BaseModel):
    """MLサービスが返す1シーンの区間（秒）"""

    start_time: float
    end_time: float


class JobCompleteRequest(BaseModel):
    """MLサービスからの処理完了コールバックのリクエストボディ。

    job_id はパスパラメータで受け取るためボディには持たない。
    """

    clips: list[ClipData]


class JobFailRequest(BaseModel):
    """MLサービスからの処理失敗コールバックのリクエストボディ。

    job_id はパスパラメータで受け取るためボディには持たない。
    """

    error: str
