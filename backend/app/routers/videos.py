import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.core.deps import get_current_user
from app.db.session import get_db
from app.models.user import User
from app.repositories import video as video_repo
from app.schemas.video import VideoResponse

from app.services import video_service

router = APIRouter(prefix="/videos", tags=["videos"])

@router.post("", response_model=VideoResponse, status_code=201)
def upload_video(
    title: str = Form(...),
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    background_tasks: BackgroundTasks = BackgroundTasks(),
) -> VideoResponse:
    """動画アップロード"""
    video = video_service.upload_video(
        db=db,
        user_id=current_user.id,
        title=title,
        file=file,
        background_tasks=background_tasks,
    )
    
    return video


@router.get("", response_model=list[VideoResponse])
def list_videos(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[VideoResponse]:
    """ログインユーザーの動画一覧取得"""
    return video_repo.get_by_user_id(db, current_user.id)


@router.get("/{video_id}", response_model=VideoResponse)
def get_video(
    video_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> VideoResponse:
    """動画詳細取得"""
    video = video_repo.get_by_id(db, video_id)
    if video is None:
        raise HTTPException(status_code=404, detail="動画が見つかりません")
    return video