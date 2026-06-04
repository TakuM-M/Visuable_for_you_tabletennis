from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.deps import get_owned_video
from app.db.session import get_db
from app.models.video import Video
from app.repositories import clip as clip_repo
from app.schemas.clip import ClipResponse

router = APIRouter(tags=["clips"])


@router.get("/videos/{video_id}/clips", response_model=list[ClipResponse])
def list_clips_by_video(
    video: Video = Depends(get_owned_video),
    db: Session = Depends(get_db),
) -> list[ClipResponse]:
    """動画に紐づく切り抜き一覧取得"""
    return clip_repo.get_by_video_id(db, video.id)
