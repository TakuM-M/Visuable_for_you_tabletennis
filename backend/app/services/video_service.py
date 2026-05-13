import os
import shutil
import uuid
from pathlib import Path

import httpx
from fastapi import BackgroundTasks, UploadFile
from sqlalchemy.orm import Session

from app.models.video import Video, VideoStatus
from app.repositories import job as job_repo
from app.repositories import video as video_repo
from app.repositories import clip as clip_repo
from app.repositories import notification_log as notification_log_repo

UPLOAD_DIR = Path("/app/uploads/videos")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_TMP_DIR = Path("/app/uploads/tmp")
CHUNK_TMP_DIR.mkdir(parents=True, exist_ok=True)

ML_SERVICE_URL = os.getenv("ML_SERVICE_URL", "http://ml-mock:8001")
BACKEND_INTERNAL_URL = os.getenv("BACKEND_INTERNAL_URL", "http://backend:8000")
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "")
RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID", "")
INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY", "")
USE_RUNPOD = os.getenv("USE_RUNPOD", "false").lower() == "true"


def call_ml_service(video_path: str, job_id: str, video_id: str) -> None:
    """MLサービスに処理を依頼する（RunPod or ローカル Mock）"""
    callback_url = f"{BACKEND_INTERNAL_URL}/internal/jobs/{job_id}/complete"

    if USE_RUNPOD:
        video_download_url = (
            f"{BACKEND_INTERNAL_URL}/internal/videos/{video_id}/raw"
            f"?token={INTERNAL_API_KEY}"
        )
        try:
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
            print(f"RunPod ジョブ送信成功 job_id={job_id} runpod_id={response.json().get('id')}")
        except Exception as e:
            print(f"RunPod 呼び出し失敗 job_id={job_id}: {e}")
    else:
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

def _register_video_and_start_ml(
    db: Session,
    user_id: uuid.UUID,
    title: str,
    save_path: Path,
    background_tasks: BackgroundTasks,
) -> Video:
    """動画をDBに登録し、MLサービスを呼び出す共通処理"""
    video = video_repo.create(
        db=db,
        user_id=user_id,
        title=title,
        storage_path=str(save_path),
    )
    video_repo.update_status(db, video.id, VideoStatus.queued)
    job = job_repo.create(db=db, video_id=video.id)
    background_tasks.add_task(call_ml_service, str(save_path), str(job.id), str(video.id))
    return video


def upload_video(
    db: Session,
    user_id: uuid.UUID,
    title: str,
    file: UploadFile,
    background_tasks: BackgroundTasks,
) -> Video:
    """動画アップロード（単一リクエスト）"""
    save_path = UPLOAD_DIR / f"{uuid.uuid4()}_{file.filename}"
    with save_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return _register_video_and_start_ml(db, user_id, title, save_path, background_tasks)


def init_chunk_upload(title: str, filename: str, total_chunks: int) -> str:
    """チャンクアップロードを初期化し、upload_idを返す"""
    upload_id = str(uuid.uuid4())
    upload_dir = CHUNK_TMP_DIR / upload_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    # メタデータを保存
    meta_path = upload_dir / "meta.json"
    import json
    meta_path.write_text(json.dumps({
        "title": title,
        "filename": filename,
        "total_chunks": total_chunks,
    }))
    return upload_id


def save_chunk(upload_id: str, index: int, file: UploadFile) -> None:
    """チャンクデータを一時ディレクトリに保存"""
    upload_dir = CHUNK_TMP_DIR / upload_id
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
    import json

    upload_dir = CHUNK_TMP_DIR / upload_id
    if not upload_dir.exists():
        raise FileNotFoundError(f"Upload {upload_id} not found")

    meta = json.loads((upload_dir / "meta.json").read_text())
    title = meta["title"]
    filename = meta["filename"]
    total_chunks = meta["total_chunks"]

    # 全チャンクが揃っているか確認
    for i in range(total_chunks):
        if not (upload_dir / str(i)).exists():
            raise FileNotFoundError(f"Chunk {i} is missing")

    # チャンクを結合
    save_path = UPLOAD_DIR / f"{uuid.uuid4()}_{filename}"
    with save_path.open("wb") as out_f:
        for i in range(total_chunks):
            chunk_path = upload_dir / str(i)
            with chunk_path.open("rb") as chunk_f:
                shutil.copyfileobj(chunk_f, out_f)

    # 一時ディレクトリを削除
    shutil.rmtree(upload_dir)

    return _register_video_and_start_ml(db, user_id, title, save_path, background_tasks)


def delete_video(db: Session, video_id: uuid.UUID) -> bool:
    # ファイルパスを先に取得（DB削除前に）
    video = video_repo.get_by_id(db, video_id)
    if video is None:
        return False
    storage_path = Path(video.storage_path)
    output_path = Path(video.output_path) if video.output_path else None

    jobs = job_repo.get_by_video_id(db, video_id)
    for job in jobs:
        notification_log_repo.delete_by_job_id(db, job.id)
    clip_repo.delete_by_video_id(db, video_id)
    job_repo.delete_by_video_id(db, video_id)
    video_repo.delete(db, video_id)

    # ファイル削除（存在しなくてもエラーにしない）
    storage_path.unlink(missing_ok=True)
    if output_path:
        output_path.unlink(missing_ok=True)

    return True

    