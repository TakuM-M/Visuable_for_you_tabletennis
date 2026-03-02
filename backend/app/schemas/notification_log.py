import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.models.notification_log import NotificationStatus


class NotificationLogResponse(BaseModel):
    """メール通知履歴レスポンス"""

    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: uuid.UUID
    job_id: uuid.UUID
    email: str
    status: NotificationStatus
    sent_at: datetime | None
    created_at: datetime
