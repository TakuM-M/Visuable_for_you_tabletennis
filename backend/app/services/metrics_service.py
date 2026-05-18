from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.models.video import Video
from app.schemas.admin import StorageMetricsResponse
from app.services import storage_service

logger = get_logger(__name__)


def collect_storage_metrics(db: Session) -> StorageMetricsResponse:
    """R2 と DB を集計してストレージ統計を返す"""
    total_videos = db.query(func.count(Video.id)).scalar() or 0
    videos_per_user_rows = (
        db.query(Video.user_id, func.count(Video.id)).group_by(Video.user_id).all()
    )

    total_bytes = 0
    object_count = 0
    try:
        client = storage_service._get_client()
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=storage_service.R2_BUCKET_NAME):
            for obj in page.get("Contents", []):
                total_bytes += obj["Size"]
                object_count += 1
    except Exception as e:
        # R2 接続失敗時もメトリクス取得自体は継続させる（DB 側は返す）
        logger.warning("R2 使用量集計失敗: %s", e)

    return StorageMetricsResponse(
        r2_total_bytes=total_bytes,
        r2_object_count=object_count,
        db_video_count=total_videos,
        videos_per_user={str(uid): cnt for uid, cnt in videos_per_user_rows},
    )
