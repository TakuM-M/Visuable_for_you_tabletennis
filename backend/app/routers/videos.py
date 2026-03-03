import os
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
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

@router.get("/{video_id}/output")
def get_output_video(
    video_id: uuid.UUID,
    request: Request,
    db: Session = Depends(get_db),
) -> StreamingResponse:
    """連結済み動画ファイルを返す（Range リクエスト対応でシーク可能）"""
    video = video_repo.get_by_id(db, video_id)
    if video is None:
        raise HTTPException(status_code=404, detail="動画が見つかりません")
    if not video.output_path:
        raise HTTPException(status_code=404, detail="連結動画がまだ生成されていません")

    file_size = os.path.getsize(video.output_path)
    range_header = request.headers.get("Range")

    if range_header:
        # Range: bytes=0-999 のような形式をパース
        range_value = range_header.strip().replace("bytes=", "")
        start_str, _, end_str = range_value.partition("-")
        start = int(start_str) if start_str else 0
        end = int(end_str) if end_str else file_size - 1
        end = min(end, file_size - 1)
        chunk_size = end - start + 1

        def iterfile():
            with open(video.output_path, "rb") as f:
                f.seek(start)
                remaining = chunk_size
                while remaining > 0:
                    data = f.read(min(65536, remaining))
                    if not data:
                        break
                    remaining -= len(data)
                    yield data

        headers = {
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(chunk_size),
        }
        return StreamingResponse(iterfile(), status_code=206, headers=headers, media_type="video/mp4")

    # Range ヘッダーなし（最初のリクエスト）
    def iterfile():
        with open(video.output_path, "rb") as f:
            while chunk := f.read(65536):
                yield chunk

    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(file_size),
    }
    return StreamingResponse(iterfile(), headers=headers, media_type="video/mp4")


@router.delete("/{video_id}", status_code=204)
def delete_video(
    video_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> None:
    """動画削除"""
    video = video_repo.get_by_id(db, video_id)
    if video is None:
        raise HTTPException(status_code=404, detail="動画が見つかりません")
    video_service.delete_video(db, video_id)
    return None