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
    created_at: datetime
