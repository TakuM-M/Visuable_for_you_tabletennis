import os
import uuid
from pathlib import Path

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.core.deps import get_current_user
from app.db.session import get_db
from app.models.user import User
from app.models.video import VideoStatus
from app.repositories import job as job_repo
from app.repositories import video as video_repo
from app.schemas.video import VideoResponse

router = APIRouter(prefix="/videos", tags=["videos"])

UPLOAD_DIR = Path("/app/uploads/videos")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ML_SERVICE_URL = os.getenv("ML_SERVICE_URL", "http://ml-mock:8001")
BACKEND_INTERNAL_URL = os.getenv("BACKEND_INTERNAL_URL", "http://backend:8000")


def call_ml_service(video_path: str, job_id: str) -> None:
    """MLサービスに処理を依頼する（バックグラウンドで実行）"""
    callback_url = f"{BACKEND_INTERNAL_URL}/internal/jobs/{job_id}/complete"
    try:
        with httpx.Client(timeout=5.0) as client:
            client.post(
                f"{ML_SERVICE_URL}/process",
                json={
                    "job_id": job_id,
                    "video_path": video_path,
                    "callback_url": callback_url,
                },
            )
        print(f"MLサービス呼び出し成功 job_id={job_id}")
    except Exception as e:
        print(f"MLサービス呼び出し失敗 job_id={job_id}: {e}")
        # 失敗してもアップロード自体は成功扱い（Jobはqueued状態のまま）


@router.post("", response_model=VideoResponse, status_code=201)
def upload_video(
    title: str = Form(...),
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    background_tasks: BackgroundTasks = BackgroundTasks(),
) -> VideoResponse:
    """動画アップロード"""
    save_path = UPLOAD_DIR / f"{uuid.uuid4()}_{file.filename}"
    with save_path.open("wb") as f:
        f.write(file.file.read())

    video = video_repo.create(
        db=db,
        user_id=current_user.id,
        title=title,
        storage_path=str(save_path),
    )
    
    video_repo.update_status(db, video.id, VideoStatus.queued)

    # Jobを作成してMLサービスに処理を依頼
    job = job_repo.create(db=db, video_id=video.id)
    background_tasks.add_task(call_ml_service, str(save_path), str(job.id))

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