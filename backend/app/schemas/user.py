import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr


class UserCreate(BaseModel):
    """ユーザー登録リクエスト"""

    email: EmailStr
    password: str
    display_name: str


class UserResponse(BaseModel):
    """ユーザー情報レスポンス（password_hash は返さない）"""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    email: str
    display_name: str
    email_verified: bool
    created_at: datetime
