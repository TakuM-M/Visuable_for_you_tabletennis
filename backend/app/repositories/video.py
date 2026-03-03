import uuid

from sqlalchemy.orm import Session

from app.models.video import Video, VideoStatus


def create(
    db: Session,
    user_id: uuid.UUID,
    title: str,
    storage_path: str,
) -> Video:
    video = Video(
        user_id=user_id,
        title=title,
        storage_path=storage_path,
    )
    db.add(video)
    db.commit()
    db.refresh(video)
    return video


def get_by_id(db: Session, video_id: uuid.UUID) -> Video | None:
    return db.query(Video).filter(Video.id == video_id).first()


def get_by_user_id(db: Session, user_id: uuid.UUID) -> list[Video]:
    return db.query(Video).filter(Video.user_id == user_id).all()


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

def delete(db: Session, video_id: uuid.UUID) -> bool:
    video = get_by_id(db, video_id)
    if video is None:
        return False
    db.delete(video)
    db.commit()
    return True