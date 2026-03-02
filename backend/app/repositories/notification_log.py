import uuid
from datetime import datetime

from sqlalchemy.orm import Session

from app.models.notification_log import NotificationLog, NotificationStatus


def create(
    db: Session,
    user_id: uuid.UUID,
    job_id: uuid.UUID,
    email: str,
) -> NotificationLog:
    log = NotificationLog(
        user_id=user_id,
        job_id=job_id,
        email=email,
    )
    db.add(log)
    db.commit()
    db.refresh(log)
    return log


def get_by_job_id(db: Session, job_id: uuid.UUID) -> list[NotificationLog]:
    return db.query(NotificationLog).filter(NotificationLog.job_id == job_id).all()


def update_status(
    db: Session,
    log_id: int,
    status: NotificationStatus,
    sent_at: datetime | None = None,
) -> NotificationLog | None:
    log = db.query(NotificationLog).filter(NotificationLog.id == log_id).first()
    if log is None:
        return None
    log.status = status
    if sent_at is not None:
        log.sent_at = sent_at
    db.commit()
    db.refresh(log)
    return log
