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

def delete_by_video_id(db: Session, video_id: uuid.UUID) -> int:
    jobs = get_by_video_id(db, video_id)
    count = len(jobs)
    for job in jobs:
        db.delete(job)
    db.commit()
    return count
