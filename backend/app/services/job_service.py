import os
import uuid
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.models.job import JobStatus
from app.models.notification_log import NotificationStatus
from app.models.video import VideoStatus
from app.repositories import clip as clip_repo
from app.repositories import job as job_repo
from app.repositories import notification_log as notification_log_repo
from app.repositories import user as user_repo
from app.repositories import video as video_repo
from app.services.email_service import send_clip_completion_email

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")


def complete_job(
    db: Session,
    job_id: uuid.UUID,
    clips: list[dict],   # {"start_time": float, "end_time": float} のリスト
) -> None:
    job = job_repo.get_by_id(db, job_id)
    if job is None:
        return

    # 1. クリップを保存
    for clip_data in clips:
        clip_repo.create(
            db=db,
            video_id=job.video_id,
            job_id=job_id,
            start_time=clip_data["start_time"],
            end_time=clip_data["end_time"],
            storage_path="",
        )

    # 2. Jobをcompletedに更新
    job_repo.update_status(
        db=db,
        job_id=job_id,
        status=JobStatus.completed,
        completed_at=datetime.now(timezone.utc),
    )

    # 3. Videoをcompletedに更新
    video = video_repo.update_status(db, job.video_id, VideoStatus.completed)

    # 4. メール送信
    if video is None:
        return
    user = user_repo.get_by_id(db, video.user_id)  # video.user_id でユーザーを取得
    if user is None:
        return

    video_url = f"{FRONTEND_URL}/videos/{video.id}"

    # 通知ログを pending で作成
    log = notification_log_repo.create(
        db=db,
        user_id=user.id,
        job_id=job_id,
        email=user.email,
    )

    # メール送信
    success = send_clip_completion_email(
        to_email=user.email,
        video_title=video.title,
        clip_count=len(clips),
        video_url=video_url,
    )

    # 送信結果に応じてログを更新
    notification_log_repo.update_status(
        db=db,
        log_id=log.id,
        status=NotificationStatus.sent if success else NotificationStatus.failed,
        sent_at=datetime.now(timezone.utc) if success else None,
    )
