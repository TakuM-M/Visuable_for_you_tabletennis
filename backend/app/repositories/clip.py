import uuid

from sqlalchemy.orm import Session

from app.models.clip import Clip


def create(
    db: Session,
    video_id: uuid.UUID,
    job_id: uuid.UUID,
    start_time: float,
    end_time: float,
    storage_path: str,
) -> Clip:
    clip = Clip(
        video_id=video_id,
        job_id=job_id,
        start_time=start_time,
        end_time=end_time,
        storage_path=storage_path,
    )
    db.add(clip)
    db.commit()
    db.refresh(clip)
    return clip


def get_by_video_id(db: Session, video_id: uuid.UUID) -> list[Clip]:
    return db.query(Clip).filter(Clip.video_id == video_id).all()


def get_by_job_id(db: Session, job_id: uuid.UUID) -> list[Clip]:
    return db.query(Clip).filter(Clip.job_id == job_id).all()
