from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.core.deps import get_current_user, get_owned_video
from app.db.session import get_db
from app.models.user import User
from app.models.video import Video
from app.repositories import video as video_repo
from app.schemas.video import (
    ChunkUploadInitRequest,
    ChunkUploadInitResponse,
    VideoOutputResponse,
    VideoResponse,
)

from app.services import video_service
from app.services import storage_service

router = APIRouter(prefix="/videos", tags=["videos"])

@router.post("", response_model=VideoResponse, status_code=201)
def upload_video(
    title: str = Form(...),
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    background_tasks: BackgroundTasks = BackgroundTasks(),
) -> VideoResponse:
    """動画アップロード（単一リクエスト）

    NOTE: 現在フロントエンドからは未使用。フロントは大容量対応のためチャンク
    アップロード（/videos/upload/init|chunk|complete、frontend/src/lib/chunkedUpload.ts）
    を使用している。小容量の直アップロード・動作確認・将来用途のために保持している。
    自分の current_user.id で作成するため所有者チェックは不要。
    """
    try:
        video = video_service.upload_video(
            db=db,
            user_id=current_user.id,
            title=title,
            file=file,
            background_tasks=background_tasks,
        )
    except video_service.QuotaExceededError as e:
        raise HTTPException(status_code=409, detail=str(e))

    return video


@router.post("/upload/init", response_model=ChunkUploadInitResponse)
def chunk_upload_init(
    body: ChunkUploadInitRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ChunkUploadInitResponse:
    """チャンクアップロード初期化"""
    try:
        upload_id = video_service.init_chunk_upload(
            db=db,
            user_id=current_user.id,
            title=body.title,
            filename=body.filename,
            total_chunks=body.total_chunks,
        )
    except video_service.QuotaExceededError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return ChunkUploadInitResponse(upload_id=upload_id)


@router.post("/upload/{upload_id}/chunk", status_code=204)
def chunk_upload(
    upload_id: str,
    index: int,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
) -> None:
    """チャンクデータ受信"""
    try:
        video_service.save_chunk(upload_id, index, file)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Upload not found")


@router.post("/upload/{upload_id}/complete", response_model=VideoResponse, status_code=201)
def chunk_upload_complete(
    upload_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    background_tasks: BackgroundTasks = BackgroundTasks(),
) -> VideoResponse:
    """チャンクアップロード完了・動画結合"""
    try:
        video = video_service.complete_chunk_upload(
            db=db,
            user_id=current_user.id,
            upload_id=upload_id,
            background_tasks=background_tasks,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except video_service.QuotaExceededError as e:
        raise HTTPException(status_code=409, detail=str(e))
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
    video: Video = Depends(get_owned_video),
) -> VideoResponse:
    """動画詳細取得"""
    return video


@router.get("/{video_id}/output", response_model=VideoOutputResponse)
def get_output_video(
    video: Video = Depends(get_owned_video),
) -> VideoOutputResponse:
    """連結済み動画の presigned URL を返す。

    認可（JWT＋所有者）は get_owned_video が担い、バイト本体は払い出した
    presigned URL でフロントが R2 から直接取得する。
    """
    if not video.output_path:
        raise HTTPException(status_code=404, detail="連結動画がまだ生成されていません")

    url = storage_service.generate_presigned_url(video.output_path)
    return VideoOutputResponse(url=url)


@router.delete("/{video_id}", status_code=204)
def delete_video(
    video: Video = Depends(get_owned_video),
    db: Session = Depends(get_db),
) -> None:
    """動画削除"""
    video_service.delete_video(db, video.id)
    return None