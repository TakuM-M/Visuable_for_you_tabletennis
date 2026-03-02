import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.deps import get_current_user
from app.db.session import get_db
from app.models.job import JobStatus
from app.models.user import User
from app.repositories import clip as clip_repo
from app.repositories import job as job_repo
from app.schemas.job import JobResponse

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


# =====================================================
# 内部エンドポイント（MLサービスからのコールバック用）
# =====================================================

class ClipData(BaseModel):
    start_time: float
    end_time: float


class JobCompleteRequest(BaseModel):
    job_id: str
    clips: list[ClipData]


@router.post("/internal/jobs/{job_id}/complete")
def complete_job(
    job_id: uuid.UUID,
    request: JobCompleteRequest,
    db: Session = Depends(get_db),
) -> dict:
    """MLサービスからの処理完了コールバック"""
    job = job_repo.get_by_id(db, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="ジョブが見つかりません")

    # 検出されたシーンをClipとして保存
    for clip_data in request.clips:
        clip_repo.create(
            db=db,
            video_id=job.video_id,
            job_id=job_id,
            start_time=clip_data.start_time,
            end_time=clip_data.end_time,
            storage_path="",  # 模擬サービスではファイルなし
        )

    # Jobのステータスをcompletedに更新
    job_repo.update_status(
        db=db,
        job_id=job_id,
        status=JobStatus.completed,
        completed_at=datetime.now(timezone.utc),
    )

    print(f"ジョブ完了 job_id={job_id} clips={len(request.clips)}件")
    return {"message": "完了"}
