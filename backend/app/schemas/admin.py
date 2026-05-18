from pydantic import BaseModel


class StorageMetricsResponse(BaseModel):
    """R2 / DB のストレージ統計レスポンス"""

    r2_total_bytes: int
    r2_object_count: int
    db_video_count: int
    videos_per_user: dict[str, int]
