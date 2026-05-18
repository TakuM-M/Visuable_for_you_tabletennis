import shutil
import time
from datetime import datetime, timedelta, timezone

from app.core.config import settings
from app.core.logging import get_logger
from app.db.session import SessionLocal
from app.repositories import job as job_repo
from app.services import job_service
from app.services.video_service import LOCAL_TMP_DIR, call_ml_service

logger = get_logger(__name__)


def reap_timeouts() -> None:
    """started_at が job_timeout_hours より古い queued / processing ジョブを失敗扱いにする"""
    threshold = datetime.now(timezone.utc) - timedelta(
        hours=settings.job_timeout_hours
    )
    with SessionLocal() as db:
        jobs = job_repo.get_timed_out_jobs(db, threshold)
        for job in jobs:
            logger.warning(
                "タイムアウト検知 job_id=%s started_at=%s", job.id, job.started_at
            )
            job_service.handle_ml_failure(db, job.id, "ML タイムアウト")


def dispatch_retries() -> None:
    """next_retry_at に達した failed ジョブを再キックする"""
    now = datetime.now(timezone.utc)
    with SessionLocal() as db:
        jobs = job_repo.get_jobs_ready_for_retry(
            db=db, now=now, max_retries=settings.job_max_retries
        )
        # 再キック対象を確定してからセッションを閉じるため、必要情報をコピー
        targets = [
            (str(job.id), str(job.video_id), job.video.storage_path)
            for job in jobs
        ]
        for job in jobs:
            job_repo.prepare_for_auto_retry(db, job.id)

    for job_id, video_id, storage_path in targets:
        logger.info("自動リトライ実行 job_id=%s", job_id)
        # call_ml_service は同期 HTTP。失敗時は内部で handle_ml_failure を呼ぶ
        call_ml_service(storage_path, job_id, video_id)


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
