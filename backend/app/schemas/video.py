import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.models.video import VideoStatus


class VideoCreate(BaseModel):
    """動画アップロードリクエスト（ファイル本体は別途 multipart で受け取る）"""

    title: str


class VideoResponse(BaseModel):
    """動画情報レスポンス"""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    user_id: uuid.UUID
    title: str
    storage_path: str
    duration: float | None
    status: VideoStatus
    created_at: datetime
    updated_at: datetime
