import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ClipResponse(BaseModel):
    """切り抜き区間レスポンス"""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    video_id: uuid.UUID
    job_id: uuid.UUID
    start_time: float
    end_time: float
    sort_order: int
    storage_path: str
    created_at: datetime


class ClipInput(BaseModel):
    """ユーザー編集で送られる 1 区間。連結順は配列内の並び順で決まる。"""

    start_time: float = Field(ge=0)
    end_time: float = Field(gt=0)

    @model_validator(mode="after")
    def _check_range(self) -> "ClipInput":
        if self.end_time <= self.start_time:
            raise ValueError("end_time は start_time より大きくする必要があります")
        return self


class ClipsReplaceRequest(BaseModel):
    """切り抜き一括置換リクエスト（PUT /videos/{id}/clips）。

    送られた配列で動画の切り抜きをまるごと置き換える。新規・編集・削除・
    並べ替えをこの 1 リクエストで表現する。
    """

    clips: list[ClipInput] = Field(max_length=200)
