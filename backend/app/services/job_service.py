import os
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import BackgroundTasks, HTTPException
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging import get_logger
from app.db.session import SessionLocal
from app.models.job import JobStatus
from app.models.notification_log import NotificationStatus
from app.models.user import User
from app.models.video import VideoStatus
from app.repositories import clip as clip_repo
from app.repositories import job as job_repo
from app.repositories import notification_log as notification_log_repo
from app.repositories import user as user_repo
from app.repositories import video as video_repo
from app.services import runpod_service
from app.services.email_service import (
    send_analysis_complete_email,
    send_clip_failure_email,
)

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


def handle_ml_failure(db: Session, job_id: uuid.UUID, error_message: str) -> None:
    """ML 処理関連の失敗を共通ハンドリングする。

    リトライ枠が残っていれば next_retry_at を設定して failed に遷移。
    リトライ枠を使い切っていれば最終 failed として動画ステータスとメール通知も行う。

    ML 失敗の入口（呼び出し失敗・完了処理失敗・タイムアウト・ML からの失敗通知・
    RunPod 状態の突き合わせ）はすべてここに集約されているため、GPU の停止もここで
    まとめて行う。
    """
    job = job_repo.get_by_id(db, job_id)
    if job is None:
        logger.warning("失敗ハンドリング対象のジョブが見つかりません job_id=%s", job_id)
        return

    # 失敗を記録する前に GPU を止める。ここを飛ばして failed にすると RunPod の
    # ワーカーが走り続けて課金が止まらず、さらにリトライで GPU が並列に増えてしまう。
    # 既に終了しているジョブへの cancel は無害なので状態を問わず投げる。
    if job.runpod_job_id:
        runpod_service.cancel_job(job.runpod_job_id)

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
        raise HTTPException(status_code=409, detail="失敗したジョブのみ再実行できます")

    job_repo.reset_for_manual_retry(db, job_id)
    video_repo.update_status(db, video.id, VideoStatus.queued)
    background_tasks.add_task(
        call_ml_service, video.storage_path, str(job.id), str(video.id)
    )


def process_complete_job(
    job_id: uuid.UUID,
    clips: list[dict],
) -> None:
    """ML コールバックの背景タスクエントリポイント。

    ルーターは即座に 202 を返し、本関数が自前 DB セッションを開いて
    重い結合処理を担当する。例外は handle_ml_failure に委譲する。
    """
    with SessionLocal() as db:
        try:
            complete_job(db=db, job_id=job_id, clips=clips)
        except Exception as e:
            logger.exception("complete_job 失敗 job_id=%s: %s", job_id, e)
            handle_ml_failure(db, job_id, f"完了処理失敗: {e}")


def process_fail_job(job_id: uuid.UUID, error: str) -> None:
    """ML 失敗コールバックの背景タスクエントリポイント。

    ML 側が自分の失敗（動画ダウンロード失敗・推論エラー等）を自覚できた場合に
    呼ばれる。これが無いと job は processing のまま残り、job_reaper の
    タイムアウト（job_timeout_hours）まで失敗が確定しない。
    リトライ判定・通知は handle_ml_failure に委譲する。
    """
    with SessionLocal() as db:
        handle_ml_failure(db, job_id, f"ML処理失敗: {error}")


def complete_job(
    db: Session,
    job_id: uuid.UUID,
    clips: list[dict],  # {"start_time": float, "end_time": float} のリスト
) -> None:
    """ML 解析完了コールバックの本処理。

    出力動画はここでは生成しない（ユーザーが編集画面で書き出し操作をしたときに
    video_service.export_video 経由で生成する）。ここでは検出された区間を clip と
    して保存し、動画を ready（編集可能）に遷移させ、編集できる旨をメール通知する。
    """
    job = job_repo.get_by_id(db, job_id)
    if job is None:
        return

    video = video_repo.get_by_id(db, job.video_id)
    if video is None:
        return

    # 1. クリップ区間を保存（並び順は検出順）
    for i, clip_data in enumerate(clips):
        clip_repo.create(
            db=db,
            video_id=job.video_id,
            job_id=job_id,
            start_time=clip_data["start_time"],
            end_time=clip_data["end_time"],
            storage_path="",
            sort_order=i,
        )

    # 2. Job を completed に更新
    job_repo.update_status(
        db=db,
        job_id=job_id,
        status=JobStatus.completed,
        completed_at=datetime.now(timezone.utc),
    )

    # 3. Video を ready（解析完了・編集可能・未書き出し）に更新
    video = video_repo.update_status(db, job.video_id, VideoStatus.ready)

    # 4. 編集できるようになったことをメール通知
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

    success = send_analysis_complete_email(
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
