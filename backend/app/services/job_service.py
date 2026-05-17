import os
import tempfile
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
from app.services import storage_service
from app.services.email_service import send_clip_completion_email
from app.services.video_clip_service import clip_video

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")


def complete_job(
    db: Session,
    job_id: uuid.UUID,
    clips: list[dict],   # {"start_time": float, "end_time": float} のリスト
) -> None:
    job = job_repo.get_by_id(db, job_id)
    if job is None:
        return

    video = video_repo.get_by_id(db, job.video_id)
    if video is None:
        return

    output_r2_key = ""

    # 1. FFmpeg でシーンをカット・結合
    if clips:
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                # R2から元動画をローカル一時ファイルにダウンロード
                local_input = os.path.join(tmpdir, "input.mp4")
                presigned_url = storage_service.generate_presigned_url(video.storage_path)

                import httpx
                with httpx.Client(timeout=600.0) as client:
                    with client.stream("GET", presigned_url) as response:
                        response.raise_for_status()
                        with open(local_input, "wb") as f:
                            for chunk in response.iter_bytes(chunk_size=65536):
                                f.write(chunk)

                # FFmpegで処理（ローカル一時ファイル）
                local_output = os.path.join(tmpdir, "play_scenes.mp4")
                clip_video(local_input, clips, local_output)

                # 処理済み動画をR2にアップロード
                output_r2_key = f"outputs/{job_id}/play_scenes.mp4"
                storage_service.upload_file(local_output, output_r2_key)

        except Exception as e:
            print(f"FFmpegクリップ失敗 job_id={job_id}: {e}")
            output_r2_key = ""

    # 2. クリップを保存
    for clip_data in clips:
        clip_repo.create(
            db=db,
            video_id=job.video_id,
            job_id=job_id,
            start_time=clip_data["start_time"],
            end_time=clip_data["end_time"],
            storage_path="",
        )

    # 3. Jobをcompletedに更新
    job_repo.update_status(
        db=db,
        job_id=job_id,
        status=JobStatus.completed,
        completed_at=datetime.now(timezone.utc),
    )

    # 4. VideoにoutputPathを保存してcompletedに更新
    video_repo.update_output_path(db, job.video_id, output_r2_key)
    video = video_repo.update_status(db, job.video_id, VideoStatus.completed)

    # 5. メール送信
    if video is None:
        return
    user = user_repo.get_by_id(db, video.user_id)
    if user is None:
        return

    video_url = f"{FRONTEND_URL}/videos/{video.id}"

    log = notification_log_repo.create(
        db=db,
        user_id=user.id,
        job_id=job_id,
        email=user.email,
    )

    success = send_clip_completion_email(
        to_email=user.email,
        video_title=video.title,
        clip_count=len(clips),
        video_url=video_url,
    )

    notification_log_repo.update_status(
        db=db,
        log_id=log.id,
        status=NotificationStatus.sent if success else NotificationStatus.failed,
        sent_at=datetime.now(timezone.utc) if success else None,
    )
