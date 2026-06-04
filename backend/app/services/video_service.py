import json
import os
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx
from fastapi import BackgroundTasks, UploadFile
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging import get_logger
from app.db.session import SessionLocal
from app.models.job import JobStatus
from app.models.video import Video, VideoStatus
from app.repositories import job as job_repo
from app.repositories import video as video_repo
from app.repositories import clip as clip_repo
from app.repositories import notification_log as notification_log_repo
from app.services import storage_service

logger = get_logger(__name__)


class QuotaExceededError(Exception):
    """ユーザーごとの動画本数上限に到達した際に raise する"""


def _ensure_under_quota(db: Session, user_id: uuid.UUID) -> None:
    """user_video_quota を超えていれば QuotaExceededError を raise する"""
    if video_repo.count_by_user_id(db, user_id) >= settings.user_video_quota:
        raise QuotaExceededError(
            f"動画本数上限 {settings.user_video_quota} 本に到達しました"
        )


def _extract_duration(local_path: Path) -> float | None:
    """ffprobe でローカル動画ファイルの再生時間（秒）を取得する。
    失敗した場合は None を返す（アップロード処理は継続）。
    """
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(local_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return float(result.stdout.strip())
    except Exception as e:
        logger.warning("ffprobe による再生時間取得に失敗しました: %s", e)
        return None

# ローカル一時ディレクトリ（チャンク結合・FFmpeg処理用）
LOCAL_TMP_DIR = Path("/app/uploads/tmp")
LOCAL_TMP_DIR.mkdir(parents=True, exist_ok=True)

ML_SERVICE_URL = os.getenv("ML_SERVICE_URL", "http://ml-mock:8001")
BACKEND_INTERNAL_URL = os.getenv("BACKEND_INTERNAL_URL", "http://backend:8000")
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "")
RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID", "")
INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY", "")
USE_RUNPOD = os.getenv("USE_RUNPOD", "false").lower() == "true"


def call_ml_service(r2_key: str, job_id: str, video_id: str) -> None:
    """MLサービスに処理を依頼する（RunPod or ローカル Mock）。

    BackgroundTasks / APScheduler の両方から呼ばれる前提で、内部で独自に DB セッションを開く。
    開始時に status=processing / started_at=now をセットし、ディスパッチ失敗時は
    job_service.handle_ml_failure に委譲して自動リトライ・通知を一元化する。
    """
    job_uuid = uuid.UUID(job_id)
    callback_url = f"{BACKEND_INTERNAL_URL}/internal/jobs/{job_id}/complete"

    # 処理開始マーク（タイムアウト判定の起点になる）
    with SessionLocal() as db:
        job_repo.update_status(
            db=db,
            job_id=job_uuid,
            status=JobStatus.processing,
            started_at=datetime.now(timezone.utc),
        )
        video_repo.update_status(db, uuid.UUID(video_id), VideoStatus.processing)

    try:
        video_download_url = storage_service.generate_presigned_url(r2_key, expires_in=7200)
        if USE_RUNPOD:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(
                    f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/run",
                    headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
                    json={
                        "input": {
                            "video_download_url": video_download_url,
                            "job_id": job_id,
                            "callback_url": callback_url,
                        }
                    },
                )
                response.raise_for_status()
            logger.info(
                "RunPod ジョブ送信成功 job_id=%s runpod_id=%s",
                job_id,
                response.json().get("id"),
            )
        else:
            with httpx.Client(timeout=5.0) as client:
                response = client.post(
                    f"{ML_SERVICE_URL}/process",
                    json={
                        "job_id": job_id,
                        "video_path": video_download_url,
                        "callback_url": callback_url,
                    },
                )
                response.raise_for_status()
            logger.info("MLサービス呼び出し成功 job_id=%s", job_id)
    except Exception as e:
        # 循環インポート回避のため遅延 import
        from app.services import job_service

        logger.exception("MLサービス呼び出し失敗 job_id=%s: %s", job_id, e)
        with SessionLocal() as db:
            job_service.handle_ml_failure(db, job_uuid, f"ML呼び出し失敗: {e}")


def _register_video_and_start_ml(
    db: Session,
    user_id: uuid.UUID,
    title: str,
    r2_key: str,
    background_tasks: BackgroundTasks,
) -> Video:
    """動画をDBに登録し、MLサービスを呼び出す共通処理"""
    video = video_repo.create(
        db=db,
        user_id=user_id,
        title=title,
        storage_path=r2_key,
    )
    video_repo.update_status(db, video.id, VideoStatus.queued)
    job = job_repo.create(db=db, video_id=video.id)
    background_tasks.add_task(call_ml_service, r2_key, str(job.id), str(video.id))
    return video


def upload_video(
    db: Session,
    user_id: uuid.UUID,
    title: str,
    file: UploadFile,
    background_tasks: BackgroundTasks,
) -> Video:
    """動画アップロード（単一リクエスト）

    NOTE: 現在フロントエンドからは未使用。フロントは大容量対応のため
    チャンクアップロード（init_chunk_upload / save_chunk / complete_chunk_upload）
    を使用している。小容量の直アップロード・動作確認・将来用途のために保持している。
    """
    _ensure_under_quota(db, user_id)
    file_id = uuid.uuid4()
    local_path = LOCAL_TMP_DIR / f"{file_id}_{file.filename}"
    r2_key = f"videos/{file_id}.mp4"

    # ローカルに一時保存 → R2にアップロード → ローカル削除
    with local_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    storage_service.upload_file(str(local_path), r2_key)
    local_path.unlink(missing_ok=True)

    return _register_video_and_start_ml(db, user_id, title, r2_key, background_tasks)


def init_chunk_upload(
    db: Session,
    user_id: uuid.UUID,
    title: str,
    filename: str,
    total_chunks: int,
) -> str:
    """チャンクアップロードを初期化し、upload_idを返す"""
    _ensure_under_quota(db, user_id)
    upload_id = str(uuid.uuid4())
    upload_dir = LOCAL_TMP_DIR / upload_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    # メタデータを保存
    meta_path = upload_dir / "meta.json"
    meta_path.write_text(json.dumps({
        "title": title,
        "filename": filename,
        "total_chunks": total_chunks,
    }))
    return upload_id


def save_chunk(upload_id: str, index: int, file: UploadFile) -> None:
    """チャンクデータを一時ディレクトリに保存"""
    upload_dir = LOCAL_TMP_DIR / upload_id
    if not upload_dir.exists():
        raise FileNotFoundError(f"Upload {upload_id} not found")
    chunk_path = upload_dir / str(index)
    with chunk_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)


def complete_chunk_upload(
    db: Session,
    user_id: uuid.UUID,
    upload_id: str,
    background_tasks: BackgroundTasks,
) -> Video:
    """全チャンクを結合して動画を登録し、ML処理を開始する"""
    _ensure_under_quota(db, user_id)
    upload_dir = LOCAL_TMP_DIR / upload_id
    if not upload_dir.exists():
        raise FileNotFoundError(f"Upload {upload_id} not found")

    meta = json.loads((upload_dir / "meta.json").read_text())
    title = meta["title"]
    total_chunks = meta["total_chunks"]

    # 全チャンクが揃っているか確認
    for i in range(total_chunks):
        if not (upload_dir / str(i)).exists():
            raise FileNotFoundError(f"Chunk {i} is missing")

    # チャンクを結合（ローカル一時ファイル）
    file_id = uuid.uuid4()
    r2_key = f"videos/{file_id}.mp4"
    merged_path = LOCAL_TMP_DIR / f"{file_id}_merged.mp4"

    with merged_path.open("wb") as out_f:
        for i in range(total_chunks):
            chunk_path = upload_dir / str(i)
            with chunk_path.open("rb") as chunk_f:
                shutil.copyfileobj(chunk_f, out_f)

    # R2にアップロード → ローカル一時ファイルを削除
    storage_service.upload_file(str(merged_path), r2_key)
    merged_path.unlink(missing_ok=True)
    shutil.rmtree(upload_dir)

    return _register_video_and_start_ml(db, user_id, title, r2_key, background_tasks)


def delete_video(db: Session, video_id: uuid.UUID) -> bool:
    video = video_repo.get_by_id(db, video_id)
    if video is None:
        return False
    storage_r2_key = video.storage_path
    output_r2_key = video.output_path if video.output_path else None

    jobs = job_repo.get_by_video_id(db, video_id)
    for job in jobs:
        notification_log_repo.delete_by_job_id(db, job.id)
    clip_repo.delete_by_video_id(db, video_id)
    job_repo.delete_by_video_id(db, video_id)
    video_repo.delete(db, video_id)

    # R2からファイルを削除
    try:
        storage_service.delete_file(storage_r2_key)
    except Exception as e:
        logger.warning("R2 元動画削除失敗: %s", e)
    if output_r2_key:
        try:
            storage_service.delete_file(output_r2_key)
        except Exception as e:
            logger.warning("R2 output削除失敗: %s", e)

    return True
