import uuid
from datetime import datetime

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models.job import Job, JobStatus
from app.models.video import Video, VideoStatus


def create(
    db: Session,
    user_id: uuid.UUID,
    title: str,
    storage_path: str,
    duration: float | None = None,
    source_duration: float | None = None,
) -> Video:
    video = Video(
        user_id=user_id,
        title=title,
        storage_path=storage_path,
        duration=duration,
        source_duration=source_duration,
    )
    db.add(video)
    db.commit()
    db.refresh(video)
    return video


def get_by_id(db: Session, video_id: uuid.UUID) -> Video | None:
    return db.query(Video).filter(Video.id == video_id).first()


def get_by_user_id(db: Session, user_id: uuid.UUID) -> list[Video]:
    return db.query(Video).filter(Video.user_id == user_id).all()


def count_by_user_id(db: Session, user_id: uuid.UUID) -> int:
    return (
        db.query(func.count(Video.id)).filter(Video.user_id == user_id).scalar() or 0
    )


def get_expired(db: Session, threshold: datetime) -> list[Video]:
    """created_at が threshold より古い動画を返す"""
    return db.query(Video).filter(Video.created_at < threshold).all()


def get_processing_without_running_job(db: Session) -> list[Video]:
    """実行中（queued / processing）の job を持たない processing 状態の動画を返す。

    ML 解析中の動画は必ず実行中 job を伴うのに対し、書き出し（export）は
    job レコードを作らずプロセス内の背景タスクだけで動く。そのため再起動で
    背景タスクが失われると processing のまま取り残される。その検出に使う。
    """
    running_video_ids = db.query(Job.video_id).filter(
        Job.status.in_([JobStatus.queued, JobStatus.processing])
    )
    return (
        db.query(Video)
        .filter(
            Video.status == VideoStatus.processing,
            ~Video.id.in_(running_video_ids),
        )
        .all()
    )


def update_status(
    db: Session,
    video_id: uuid.UUID,
    status: VideoStatus,
) -> Video | None:
    video = get_by_id(db, video_id)
    if video is None:
        return None
    video.status = status
    db.commit()
    db.refresh(video)
    return video

def update_output_path(
    db: Session,
    video_id: uuid.UUID,
    output_path: str,
) -> Video | None:
    video = get_by_id(db, video_id)
    if video is None:
        return None
    video.output_path = output_path
    db.commit()
    db.refresh(video)
    return video

def update_duration(
    db: Session,
    video_id: uuid.UUID,
    duration: float,
) -> Video | None:
    video = get_by_id(db, video_id)
    if video is None:
        return None
    video.duration = duration
    db.commit()
    db.refresh(video)
    return video

def delete(db: Session, video_id: uuid.UUID) -> bool:
    video = get_by_id(db, video_id)
    if video is None:
        return False
    db.delete(video)
    db.commit()
    return True