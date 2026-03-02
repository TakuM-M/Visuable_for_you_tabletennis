import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.repositories import video as video_repo
from app.schemas.video import VideoCreate, VideoResponse

router = APIRouter(prefix="/videos", tags=["videos"])

UPLOAD_DIR = Path("/app/uploads/videos")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("", response_model=VideoResponse, status_code=201)
def upload_video(
    title: str = Form(...),
    file: UploadFile = File(...),
    # TODO: 認証実装後は現在のログインユーザーから取得する
    user_id: uuid.UUID = Form(...),
    db: Session = Depends(get_db),
) -> VideoResponse:
    """動画アップロード"""
    save_path = UPLOAD_DIR / f"{uuid.uuid4()}_{file.filename}"
    with save_path.open("wb") as f:
        f.write(file.file.read())

    video = video_repo.create(
        db=db,
        user_id=user_id,
        title=title,
        storage_path=str(save_path),
    )
    return video


@router.get("", response_model=list[VideoResponse])
def list_videos(
    # TODO: 認証実装後は現在のログインユーザーから取得する
    user_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> list[VideoResponse]:
    """動画一覧取得"""
    return video_repo.get_by_user_id(db, user_id)


@router.get("/{video_id}", response_model=VideoResponse)
def get_video(video_id: uuid.UUID, db: Session = Depends(get_db)) -> VideoResponse:
    """動画詳細取得"""
    video = video_repo.get_by_id(db, video_id)
    if video is None:
        raise HTTPException(status_code=404, detail="動画が見つかりません")
    return video
