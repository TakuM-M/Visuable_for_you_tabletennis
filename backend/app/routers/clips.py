from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.deps import get_owned_video
from app.db.session import get_db
from app.models.clip import Clip
from app.models.video import Video
from app.repositories.clip import clip_repository as clip_repo
from app.schemas.clip import ClipResponse, ClipsReplaceRequest
from app.services import video_service

router = APIRouter(tags=["clips"])


def _to_response(clip: Clip) -> ClipResponse:
    """Clip モデルを API レスポンスへ変換する"""
    return ClipResponse.model_validate(clip)


@router.get("/videos/{video_id}/clips", response_model=list[ClipResponse])
def list_clips_by_video(
    video: Video = Depends(get_owned_video),
    db: Session = Depends(get_db),
) -> list[ClipResponse]:
    """動画に紐づく切り抜き一覧取得"""
    return [_to_response(clip) for clip in clip_repo.get_by_video_id(db, video.id)]


@router.put("/videos/{video_id}/clips", response_model=list[ClipResponse])
def replace_clips_by_video(
    body: ClipsReplaceRequest,
    video: Video = Depends(get_owned_video),
    db: Session = Depends(get_db),
) -> list[ClipResponse]:
    """切り抜きを一括置換する（新規・編集・削除・並べ替えをまとめて反映）。

    出力動画はこの時点では再生成しない。編集を反映するには
    POST /videos/{id}/export で書き出す。
    """
    return [
        _to_response(clip)
        for clip in video_service.replace_clips(db, video, body.clips)
    ]
