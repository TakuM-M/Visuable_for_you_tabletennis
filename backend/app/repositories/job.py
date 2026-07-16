import uuid
from datetime import datetime

from sqlalchemy.orm import Session

from app.models.job import Job, JobStatus


def create(db: Session, video_id: uuid.UUID) -> Job:
    job = Job(video_id=video_id)
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def get_by_id(db: Session, job_id: uuid.UUID) -> Job | None:
    return db.query(Job).filter(Job.id == job_id).first()


def get_by_video_id(db: Session, video_id: uuid.UUID) -> list[Job]:
    return db.query(Job).filter(Job.video_id == video_id).all()


def get_latest_by_video_id(db: Session, video_id: uuid.UUID) -> Job | None:
    """動画に紐づく最新のジョブを返す。

    ユーザーが編集で新規作成した clip に流用する job_id を取得するために使う。
    """
    return (
        db.query(Job)
        .filter(Job.video_id == video_id)
        .order_by(Job.created_at.desc())
        .first()
    )


def update_status(
    db: Session,
    job_id: uuid.UUID,
    status: JobStatus,
    started_at: datetime | None = None,
    completed_at: datetime | None = None,
    error_message: str | None = None,
) -> Job | None:
    job = get_by_id(db, job_id)
    if job is None:
        return None
    job.status = status
    if started_at is not None:
        job.started_at = started_at
    if completed_at is not None:
        job.completed_at = completed_at
    if error_message is not None:
        job.error_message = error_message
    db.commit()
    db.refresh(job)
    return job


def mark_failed(
    db: Session,
    job_id: uuid.UUID,
    error_message: str,
    next_retry_at: datetime | None,
) -> Job | None:
    """ジョブを失敗状態に遷移させる。next_retry_at を渡せば自動リトライ対象となる"""
    job = get_by_id(db, job_id)
    if job is None:
        return None
    job.status = JobStatus.failed
    job.error_message = error_message
    job.next_retry_at = next_retry_at
    db.commit()
    db.refresh(job)
    return job


def get_timed_out_jobs(db: Session, threshold: datetime) -> list[Job]:
    """started_at が threshold より古い queued / processing ジョブを取得する"""
    return (
        db.query(Job)
        .filter(Job.status.in_([JobStatus.queued, JobStatus.processing]))
        .filter(Job.started_at.is_not(None))
        .filter(Job.started_at < threshold)
        .with_for_update(skip_locked=True)
        .all()
    )
    
    
def get_queued_started_null_jobs(db: Session) -> list[Job]:
    """status : queued started_at : null のジョブを取得する"""
    return (
        db.query(Job)
        .filter(Job.status == JobStatus.queued)
        .filter(Job.started_at.is_(None))
        .with_for_update(skip_locked=True)
        .all()
    )


def get_jobs_ready_for_retry(
    db: Session, now: datetime, max_retries: int
) -> list[Job]:
    """next_retry_at が now を過ぎた failed ジョブで、まだ自動リトライ枠が残っているものを取得"""
    return (
        db.query(Job)
        .filter(Job.status == JobStatus.failed)
        .filter(Job.retry_count < max_retries)
        .filter(Job.next_retry_at.is_not(None))
        .filter(Job.next_retry_at <= now)
        .with_for_update(skip_locked=True)
        .all()
    )


def prepare_for_auto_retry(db: Session, job_id: uuid.UUID) -> Job | None:
    """自動リトライ実行直前: retry_count をインクリメントして queued に戻す"""
    job = get_by_id(db, job_id)
    if job is None:
        return None
    job.status = JobStatus.queued
    job.retry_count = job.retry_count + 1
    job.started_at = None
    job.completed_at = None
    job.next_retry_at = None
    job.error_message = None
    db.commit()
    db.refresh(job)
    return job


def reset_for_manual_retry(db: Session, job_id: uuid.UUID) -> Job | None:
    """手動再実行: retry_count を 0 に戻し、自動リトライ枠を作り直す"""
    job = get_by_id(db, job_id)
    if job is None:
        return None
    job.status = JobStatus.queued
    job.retry_count = 0
    job.started_at = None
    job.completed_at = None
    job.next_retry_at = None
    job.error_message = None
    db.commit()
    db.refresh(job)
    return job


def delete_by_video_id(db: Session, video_id: uuid.UUID) -> int:
    jobs = get_by_video_id(db, video_id)
    count = len(jobs)
    for job in jobs:
        db.delete(job)
    db.commit()
    return count
