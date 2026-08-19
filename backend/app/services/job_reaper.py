import shutil
import time
from datetime import datetime, timedelta, timezone

from app.core.config import settings
from app.core.logging import get_logger
from app.db.session import SessionLocal
from app.repositories.protocols import JobRepository, VideoRepository
from app.repositories.job import job_repository
from app.repositories.video import video_repository
from app.services import job_service, metrics_service, runpod_service, video_service
from app.services.video_service import LOCAL_TMP_DIR, call_ml_service

logger = get_logger(__name__)


def reap_timeouts(*, job_repo: JobRepository = job_repository) -> None:
    """started_at が job_timeout_hours より古い queued / processing ジョブを失敗扱いにする"""
    threshold = datetime.now(timezone.utc) - timedelta(hours=settings.job_timeout_hours)
    with SessionLocal() as db:
        jobs = job_repo.get_timed_out_jobs(db, threshold)
        for job in jobs:
            logger.warning(
                "タイムアウト検知 job_id=%s started_at=%s", job.id, job.started_at
            )
            job_service.handle_ml_failure(db, job.id, "ML タイムアウト")


def reconcile_runpod_jobs(*, job_repo: JobRepository = job_repository) -> None:
    """processing のジョブを RunPod 側の実状態と突き合わせ、終了済みなら失敗扱いにする。

    GPU 側が自分の失敗を通知できないケース（OOM・ワーカークラッシュ・コールバック
    送信自体の失敗）は /internal/jobs/{id}/fail が呼ばれないため、backend からは
    「processing のまま音信不通」にしか見えない。RunPod に問い合わせることで
    job_timeout_hours（既定 24 時間）を待たずに検知する。
    """
    if not video_service.USE_RUNPOD:
        return

    with SessionLocal() as db:
        # セッションを跨いで使うので必要な値だけコピーする
        targets = [
            (job.id, job.runpod_job_id)
            for job in job_repo.get_running_runpod_jobs(db)
            if job.runpod_job_id is not None
        ]

    for job_id, runpod_job_id in targets:
        status = runpod_service.get_job_status(runpod_job_id)

        # None（問い合わせ失敗）はまだ生きている可能性があるので触らない。
        # 一時的な通信断で稼働中のジョブを殺さないための判断。
        if status is None or status in runpod_service.ACTIVE_STATUSES:
            continue

        if status == "COMPLETED":
            # 推論は完了しているのにコールバックが届いていない。原因は backend 側
            # （502・APIキー不一致）やネットワークであることが多く、リトライしても
            # 同じ結果になりうるため、ログとエラーメッセージで区別できるようにする
            message = "RunPod は COMPLETED だがコールバック未達"
        elif status in runpod_service.DEAD_STATUSES:
            message = f"RunPod ジョブ異常終了: {status}"
        else:
            # 未知の状態は判断を保留する（誤検知で生きているジョブを殺さない）。
            # 本当に固まっていれば reap_timeouts が最終的に拾う
            logger.warning(
                "RunPod の未知の状態を検知 job_id=%s status=%s", job_id, status
            )
            continue

        logger.warning(
            "RunPod 状態の不一致を検知 job_id=%s runpod_status=%s", job_id, status
        )
        with SessionLocal() as db:
            job_service.handle_ml_failure(db, job_id, message)


def dispatch_retries(*, job_repo: JobRepository = job_repository) -> None:
    """next_retry_at に達した failed ジョブを再キックする"""
    now = datetime.now(timezone.utc)
    with SessionLocal() as db:
        jobs = job_repo.get_jobs_ready_for_retry(
            db=db, now=now, max_retries=settings.job_max_retries
        )
        # 再キック対象を確定してからセッションを閉じるため、必要情報をコピー
        targets = [
            (str(job.id), str(job.video_id), job.video.storage_path) for job in jobs
        ]
        for job in jobs:
            job_repo.prepare_for_auto_retry(db, job.id)

    for job_id, video_id, storage_path in targets:
        logger.info("自動リトライ実行 job_id=%s", job_id)
        # call_ml_service は同期 HTTP。失敗時は内部で handle_ml_failure を呼ぶ
        call_ml_service(storage_path, job_id, video_id)


def cleanup_expired_videos(*, video_repo: VideoRepository = video_repository) -> None:
    """video_retention_days を超えた動画を delete_video() で削除する"""
    threshold = datetime.now(timezone.utc) - timedelta(
        days=settings.video_retention_days
    )
    with SessionLocal() as db:
        expired = video_repo.get_expired(db, threshold)
        # セッションを跨いで使うので id だけコピー
        ids = [v.id for v in expired]

    removed = 0
    for video_id in ids:
        try:
            with SessionLocal() as db:
                if video_service.delete_video(db, video_id):
                    removed += 1
        except Exception as e:
            logger.warning("保持期限切れ動画の削除失敗 id=%s: %s", video_id, e)

    if removed > 0:
        logger.info("保持期限切れ動画 削除数=%s", removed)


def log_storage_metrics() -> None:
    """R2 / DB の現在のストレージ統計を INFO ログに出力する"""
    try:
        with SessionLocal() as db:
            metrics = metrics_service.collect_storage_metrics(db)
    except Exception as e:
        logger.warning("ストレージ統計取得失敗: %s", e)
        return
    logger.info(
        "ストレージ統計 r2_bytes=%s r2_objects=%s db_videos=%s",
        metrics.r2_total_bytes,
        metrics.r2_object_count,
        metrics.db_video_count,
    )


def clean_tmp_dir() -> None:
    """LOCAL_TMP_DIR 配下で mtime が tmp_retention_hours を超えた項目を削除する"""
    if not LOCAL_TMP_DIR.exists():
        return

    cutoff = time.time() - settings.tmp_retention_hours * 3600
    removed = 0
    for entry in LOCAL_TMP_DIR.iterdir():
        try:
            if entry.stat().st_mtime >= cutoff:
                continue
            if entry.is_dir():
                shutil.rmtree(entry, ignore_errors=True)
            else:
                entry.unlink(missing_ok=True)
            removed += 1
        except Exception as e:
            logger.warning("tmp 削除失敗 path=%s: %s", entry, e)
    if removed > 0:
        logger.info("tmp クリーンアップ完了 削除数=%s", removed)
