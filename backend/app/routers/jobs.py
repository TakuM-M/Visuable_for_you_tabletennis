import os
import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.deps import get_current_user
from app.db.session import get_db
from app.models.user import User
from app.repositories import job as job_repo
from app.repositories import video as video_repo
from app.schemas.job import JobResponse
from app.services import job_service

router = APIRouter(tags=["jobs"])


@router.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(
    job_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> JobResponse:
    """ジョブ詳細取得"""
    job = job_repo.get_by_id(db, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="ジョブが見つかりません")
    return job


@router.get("/videos/{video_id}/jobs", response_model=list[JobResponse])
def list_jobs_by_video(
    video_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[JobResponse]:
    """動画に紐づくジョブ一覧取得"""
    return job_repo.get_by_video_id(db, video_id)

class ClipData(BaseModel):
    start_time: float
    end_time: float


class JobCompleteRequest(BaseModel):
    job_id: str
    clips: list[ClipData]

@router.get("/internal/videos/{video_id}/raw")
def download_raw_video(
    video_id: uuid.UUID,
    token: str,
    db: Session = Depends(get_db),
) -> FileResponse:
    """RunPod ワーカーが動画をダウンロードするための内部エンドポイント"""
    internal_api_key = os.getenv("INTERNAL_API_KEY", "")
    if not internal_api_key or token != internal_api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")
    video = video_repo.get_by_id(db, video_id)
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(video.storage_path, media_type="video/mp4")


@router.post("/internal/jobs/{job_id}/complete")
def complete_job(
    job_id: uuid.UUID,
    request: JobCompleteRequest,
    db: Session = Depends(get_db),
) -> dict:
    """MLサービスからの処理完了コールバック"""
    clips = [{"start_time": c.start_time, "end_time": c.end_time} for c in request.clips]
    job_service.complete_job(db=db, job_id=job_id, clips=clips)
    print(f"ジョブ完了 job_id={job_id} clips={len(clips)}件")
    return {"message": "完了"}