import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.deps import get_current_user, get_owned_video, require_internal_api_key
from app.core.logging import get_logger
from app.db.session import get_db
from app.models.user import User
from app.models.video import Video
from app.repositories import job as job_repo
from app.schemas.job import JobCompleteRequest, JobResponse
from app.services import job_service

logger = get_logger(__name__)

router = APIRouter(tags=["jobs"])


@router.get("/videos/{video_id}/jobs", response_model=list[JobResponse])
def list_jobs_by_video(
    video: Video = Depends(get_owned_video),
    db: Session = Depends(get_db),
) -> list[JobResponse]:
    """動画に紐づくジョブ一覧取得"""
    return job_repo.get_by_video_id(db, video.id)


@router.post("/jobs/{job_id}/retry", response_model=JobResponse)
def retry_job(
    job_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> JobResponse:
    """失敗したジョブの手動再実行"""
    job_service.retry_job(
        db=db,
        job_id=job_id,
        current_user=current_user,
        background_tasks=background_tasks,
    )
    job = job_repo.get_by_id(db, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="ジョブが見つかりません")
    return job


@router.post(
    "/internal/jobs/{job_id}/complete",
    dependencies=[Depends(require_internal_api_key)],
    status_code=202,
)
def complete_job(
    job_id: uuid.UUID,
    request: JobCompleteRequest,
    background_tasks: BackgroundTasks,
) -> dict:
    """MLサービスからの処理完了コールバック。

    重い結合処理は背景タスクへ委譲し、ML 側の read timeout を回避するため即 202 を返す。
    """
    clips = [
        {"start_time": c.start_time, "end_time": c.end_time} for c in request.clips
    ]
    background_tasks.add_task(job_service.process_complete_job, job_id, clips)
    logger.info("ジョブ完了コールバック受信 job_id=%s clips=%s件", job_id, len(clips))
    return {"message": "受付完了"}
