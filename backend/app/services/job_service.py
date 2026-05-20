import os
import tempfile
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import BackgroundTasks, HTTPException
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging import get_logger
from app.models.job import JobStatus
from app.models.notification_log import NotificationStatus
from app.models.user import User
from app.models.video import VideoStatus
from app.repositories import clip as clip_repo
from app.repositories import job as job_repo
from app.repositories import notification_log as notification_log_repo
from app.repositories import user as user_repo
from app.repositories import video as video_repo
from app.services import storage_service
from app.services.email_service import (
    send_clip_completion_email,
    send_clip_failure_email,
)
from app.services.video_clip_service import clip_video

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")

logger = get_logger(__name__)


def _video_url(video_id: uuid.UUID) -> str:
    return f"{FRONTEND_URL}/videos/{video_id}"


def _compute_next_retry_at(retry_count: int) -> datetime:
    """retry_count 回目の失敗後に次回リトライをいつ行うかを計算する"""
    backoff = settings.job_retry_backoff_seconds
    if not backoff:
        return datetime.now(timezone.utc) + timedelta(seconds=60)
    index = min(retry_count, len(backoff) - 1)
    return datetime.now(timezone.utc) + timedelta(seconds=backoff[index])


def _send_failure_notification(
    db: Session, job_id: uuid.UUID, video_id: uuid.UUID, error_message: str
) -> None:
    """最終失敗時にユーザーへメール通知し、結果を notification_logs に記録する"""
    video = video_repo.get_by_id(db, video_id)
    if video is None:
        return
    user = user_repo.get_by_id(db, video.user_id)
    if user is None:
        return

    log = notification_log_repo.create(
        db=db,
        user_id=user.id,
        job_id=job_id,
        email=user.email,
    )
    success = send_clip_failure_email(
        to_email=user.email,
        video_title=video.title,
        video_url=_video_url(video.id),
        error_message=error_message,
    )
    notification_log_repo.update_status(
        db=db,
        log_id=log.id,
        status=NotificationStatus.sent if success else NotificationStatus.failed,
        sent_at=datetime.now(timezone.utc) if success else None,
    )


def handle_ml_failure(
    db: Session, job_id: uuid.UUID, error_message: str
) -> None:
    """ML 処理関連の失敗を共通ハンドリングする。

    リトライ枠が残っていれば next_retry_at を設定して failed に遷移。
    リトライ枠を使い切っていれば最終 failed として動画ステータスとメール通知も行う。
    """
    job = job_repo.get_by_id(db, job_id)
    if job is None:
        logger.warning("失敗ハンドリング対象のジョブが見つかりません job_id=%s", job_id)
        return

    if job.retry_count < settings.job_max_retries:
        next_retry_at = _compute_next_retry_at(job.retry_count)
        job_repo.mark_failed(db, job_id, error_message, next_retry_at)
        logger.info(
            "ジョブ失敗 job_id=%s retry_count=%s next_retry_at=%s",
            job_id,
            job.retry_count,
            next_retry_at.isoformat(),
        )
        return

    # 最終失敗
    job_repo.mark_failed(db, job_id, error_message, None)
    video_repo.update_status(db, job.video_id, VideoStatus.failed)
    _send_failure_notification(db, job_id, job.video_id, error_message)
    logger.warning(
        "ジョブ最終失敗 job_id=%s retry_count=%s error=%s",
        job_id,
        job.retry_count,
        error_message,
    )


def retry_job(
    db: Session,
    job_id: uuid.UUID,
    current_user: User,
    background_tasks: BackgroundTasks,
) -> None:
    """ユーザー操作による手動再実行。retry_count を 0 にリセットして再キックする"""
    # 循環インポート回避のため遅延 import
    from app.services.video_service import call_ml_service

    job = job_repo.get_by_id(db, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="ジョブが見つかりません")

    video = video_repo.get_by_id(db, job.video_id)
    if video is None:
        raise HTTPException(status_code=404, detail="動画が見つかりません")

    if video.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="このジョブへの権限がありません")

    if job.status != JobStatus.failed:
        raise HTTPException(
            status_code=409, detail="失敗したジョブのみ再実行できます"
        )

    job_repo.reset_for_manual_retry(db, job_id)
    video_repo.update_status(db, video.id, VideoStatus.queued)
    background_tasks.add_task(
        call_ml_service, video.storage_path, str(job.id), str(video.id)
    )


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
            logger.exception("FFmpegクリップ失敗 job_id=%s: %s", job_id, e)
            handle_ml_failure(db, job_id, f"クリップ生成失敗: {e}")
            return

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

    # 出力動画の再生時間を取得・保存
    if output_r2_key:
        try:
            from app.services import video_service
            output_url = storage_service.generate_presigned_url(output_r2_key, expires_in=7200)
            output_duration = video_service._extract_duration(output_url)
            if output_duration is not None:
                video_repo.update_duration(db, job.video_id, output_duration)
        except Exception as e:
            logger.warning("出力動画の再生時間取得に失敗しました job_id=%s: %s", job_id, e)

    video = video_repo.update_status(db, job.video_id, VideoStatus.completed)

    # 5. メール送信
    if video is None:
        return
    user = user_repo.get_by_id(db, video.user_id)
    if user is None:
        return

    video_url = _video_url(video.id)

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
