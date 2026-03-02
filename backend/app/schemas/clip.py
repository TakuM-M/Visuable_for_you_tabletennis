import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class ClipResponse(BaseModel):
    """切り抜き動画レスポンス（クリップはMLサービスが生成するためCreateスキーマは不要）"""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    video_id: uuid.UUID
    job_id: uuid.UUID
    start_time: float
    end_time: float
    storage_path: str
    created_at: datetime
