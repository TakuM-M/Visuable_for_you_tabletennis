import json
import os
import shutil
import subprocess
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx
from fastapi import BackgroundTasks, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging import get_logger
from app.db.session import SessionLocal
from app.models.clip import Clip
from app.models.job import JobStatus
from app.models.video import Video, VideoStatus
from app.repositories import job as job_repo
from app.repositories import video as video_repo
from app.repositories import clip as clip_repo
from app.repositories import notification_log as notification_log_repo
from app.schemas.clip import ClipInput
from app.services import storage_service
from app.services.video_clip_service import clip_video

logger = get_logger(__name__)


class QuotaExceededError(Exception):
    """ユーザーごとの動画本数上限に到達した際に raise する"""


def _ensure_under_quota(db: Session, user_id: uuid.UUID) -> None:
    """user_video_quota 1ユーザーあたりの動画本数上限を超えていれば QuotaExceededError を raise する"""
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
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
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


# サムネイルの横幅（px）。高さはアスペクト比維持で自動決定する。
# 一覧の小さいセルと詳細のプレビューを1枚で兼ねる想定のサイズ。
THUMBNAIL_WIDTH = 640


def _thumbnail_key(video_r2_key: str) -> str:
    """元動画の R2 キー（videos/{uuid}.mp4）からサムネイルの R2 キーを導出する"""
    return f"thumbnails/{Path(video_r2_key).stem}.jpg"


def _generate_thumbnail(
    local_path: Path, video_r2_key: str, duration: float | None
) -> str | None:
    """ローカル動画から静止画を1枚切り出して R2 に置き、その R2 キーを返す。

    サムネイルは表示上の付加情報でしかないので、失敗しても例外にせず None を
    返してアップロード処理を続行する（thumbnail_path は None のままになり、
    フロントはプレースホルダを表示する）。
    先頭フレームは暗転やカメラ設置中であることが多いため、少し進んだ位置から取る。
    """
    seek = min(3.0, duration * 0.1) if duration else 1.0
    thumb_path = local_path.parent / f"{local_path.stem}_thumb.jpg"
    r2_key = _thumbnail_key(video_r2_key)
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                # -ss を -i の前に置いて入力シークさせる（先頭から全デコードしない）
                "-ss",
                str(seek),
                "-i",
                str(local_path),
                "-frames:v",
                "1",
                "-vf",
                f"scale={THUMBNAIL_WIDTH}:-2",
                "-q:v",
                "4",
                str(thumb_path),
            ],
            check=True,
            capture_output=True,
            timeout=60,
        )
        storage_service.upload_file(str(thumb_path), r2_key, content_type="image/jpeg")
        return r2_key
    except Exception as e:
        logger.warning("サムネイル生成に失敗しました r2_key=%s: %s", video_r2_key, e)
        return None
    finally:
        thumb_path.unlink(missing_ok=True)


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
    fail_callback_url = f"{BACKEND_INTERNAL_URL}/internal/jobs/{job_id}/fail"

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
        video_download_url = storage_service.generate_presigned_url(
            r2_key, expires_in=7200
        )
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
                            "fail_callback_url": fail_callback_url,
                        }
                    },
                )
                response.raise_for_status()
            # RunPod 側のジョブIDを控える。これが無いと GPU の生死確認（/status）も
            # 停止（/cancel）もできず、課金を止める手段が無くなる
            runpod_job_id = response.json().get("id")
            if runpod_job_id:
                with SessionLocal() as db:
                    job_repo.set_runpod_job_id(db, job_uuid, runpod_job_id)
            logger.info(
                "RunPod ジョブ送信成功 job_id=%s runpod_id=%s",
                job_id,
                runpod_job_id,
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
    source_duration: float | None = None,
    thumbnail_path: str | None = None,
) -> Video:
    """動画をDBに登録し、MLサービスを呼び出す共通処理"""
    video = video_repo.create(
        db=db,
        user_id=user_id,
        title=title,
        storage_path=r2_key,
        source_duration=source_duration,
        thumbnail_path=thumbnail_path,
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

    # ローカルに一時保存 → R2にアップロード → 元動画長とサムネイルを取得 → ローカル削除
    with local_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    storage_service.upload_file(str(local_path), r2_key)
    source_duration = _extract_duration(local_path)
    thumbnail_path = _generate_thumbnail(local_path, r2_key, source_duration)
    local_path.unlink(missing_ok=True)

    return _register_video_and_start_ml(
        db, user_id, title, r2_key, background_tasks, source_duration, thumbnail_path
    )


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
    meta_path.write_text(
        json.dumps(
            {
                "title": title,
                "filename": filename,
                "total_chunks": total_chunks,
            }
        )
    )
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
    """r2_keyを決定し映像の結合・アップロード・MLサービスはバックグラウンドで実行。"""
    # クォーター確認（完了時点での動画本数をチェックする）
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

    file_id = uuid.uuid4()
    r2_key = f"videos/{file_id}.mp4"

    # video作成
    video = video_repo.create(
        db=db,
        user_id=user_id,
        title=title,
        storage_path=r2_key,
    )
    video_repo.update_status(db, video.id, VideoStatus.queued)
    job = job_repo.create(db=db, video_id=video.id)
    background_tasks.add_task(
        process_chunk_upload, upload_id, r2_key, str(video.id), str(job.id)
    )

    return video


def process_chunk_upload(
    upload_id: str, r2_key: str, video_id: str, job_id: str
) -> None:
    """結合した動画をR2にアップロードし、MLサービスを呼び出す。BackgroundTasksで実行される。"""
    try:
        merged_path = LOCAL_TMP_DIR / f"{upload_id}_merged.mp4"
        upload_dir = LOCAL_TMP_DIR / upload_id

        meta = json.loads((upload_dir / "meta.json").read_text())
        total_chunks = meta["total_chunks"]

        with merged_path.open("wb") as out_f:
            for i in range(total_chunks):
                chunk_path = upload_dir / str(i)
                with chunk_path.open("rb") as chunk_f:
                    shutil.copyfileobj(chunk_f, out_f)

        storage_service.upload_file(str(merged_path), r2_key)
        duration = _extract_duration(merged_path)
        thumbnail_path = _generate_thumbnail(merged_path, r2_key, duration)
        with SessionLocal() as db:
            if duration is not None:
                video_repo.update_source_duration(db, uuid.UUID(video_id), duration)
            if thumbnail_path is not None:
                video_repo.update_thumbnail_path(
                    db, uuid.UUID(video_id), thumbnail_path
                )
        merged_path.unlink(missing_ok=True)
        shutil.rmtree(upload_dir)

        call_ml_service(r2_key, str(job_id), str(video_id))
        logger.info(
            "チャンクアップロード処理完了 video_id=%s job_id=%s", video_id, job_id
        )
    except Exception as e:
        logger.exception("チャンクアップロード処理失敗 video_id=%s: %s", video_id, e)
        with SessionLocal() as db:
            job_repo.update_status(
                db=db,
                job_id=uuid.UUID(job_id),
                status=JobStatus.failed,
                error_message=f"チャンクアップロード処理失敗: {e}",
            )
            video_repo.update_status(db, uuid.UUID(video_id), VideoStatus.failed)


def delete_video(db: Session, video_id: uuid.UUID) -> bool:
    video = video_repo.get_by_id(db, video_id)
    if video is None:
        return False

    # R2 → DB の順で削除する。DB の行は R2 キーを指す唯一のポインタであり、
    # 先に DB を消すと R2 削除失敗時にオブジェクトが誰からも参照されないまま
    # 残り続ける（オーファン化）。R2 削除に失敗した場合は例外を送出して行を残し、
    # 後から再削除できるようにする（R2 の削除は対象が無くても成功するため再実行は安全）。
    storage_service.delete_file(video.storage_path)
    if video.output_path:
        storage_service.delete_file(video.output_path)
    if video.thumbnail_path:
        storage_service.delete_file(video.thumbnail_path)

    jobs = job_repo.get_by_video_id(db, video_id)
    for job in jobs:
        notification_log_repo.delete_by_job_id(db, job.id)
    clip_repo.delete_by_video_id(db, video_id)
    job_repo.delete_by_video_id(db, video_id)
    video_repo.delete(db, video_id)

    return True


def replace_clips(
    db: Session,
    video: Video,
    clips_input: list[ClipInput],
) -> list[Clip]:
    """切り抜きを一括置換する（新規・編集・削除・並べ替えをまとめて反映）。

    区間の整合性（非負・start<end）は ClipInput 側で検証済み。元動画長が
    分かる場合は end_time の上限も検証する。新規作成される clip には動画の
    最新ジョブの id を流用する（clip.job_id は NOT NULL のため）。
    """
    if video.source_duration is not None:
        for c in clips_input:
            if c.end_time > video.source_duration:
                raise HTTPException(
                    status_code=422,
                    detail=f"区間が元動画の長さ（{video.source_duration}秒）を超えています",
                )

    job = job_repo.get_latest_by_video_id(db, video.id)
    if job is None:
        raise HTTPException(
            status_code=409, detail="解析ジョブが存在しないため編集できません"
        )

    clips_data = [
        {"start_time": c.start_time, "end_time": c.end_time} for c in clips_input
    ]
    return clip_repo.replace_for_video(db, video.id, job.id, clips_data)


def rebuild_output(db: Session, video_id: uuid.UUID, clips: list[dict]) -> str:
    """現在の clip 区間から連結動画を生成して R2 にアップロードし、
    output_path / duration / status(completed) を更新する。R2 キーを返す。

    書き出しの度に同一キー（outputs/{video_id}/play_scenes.mp4）へ上書きする。
    FFmpeg などの失敗は例外として送出し、ハンドリングは呼び出し側に委ねる。
    clips は {"start_time": float, "end_time": float} のリスト。
    """
    video = video_repo.get_by_id(db, video_id)
    if video is None:
        raise ValueError(f"動画が見つかりません video_id={video_id}")

    output_r2_key = ""
    if clips:
        with tempfile.TemporaryDirectory() as tmpdir:
            # R2 から元動画をローカル一時ファイルにダウンロード
            local_input = os.path.join(tmpdir, "input.mp4")
            presigned_url = storage_service.generate_presigned_url(video.storage_path)
            with httpx.Client(timeout=600.0) as client:
                with client.stream("GET", presigned_url) as response:
                    response.raise_for_status()
                    with open(local_input, "wb") as f:
                        for chunk in response.iter_bytes(chunk_size=65536):
                            f.write(chunk)

            # FFmpeg でシーンをカット・結合
            local_output = os.path.join(tmpdir, "play_scenes.mp4")
            clip_video(local_input, clips, local_output)

            # 処理済み動画を R2 にアップロード
            output_r2_key = f"outputs/{video_id}/play_scenes.mp4"
            storage_service.upload_file(local_output, output_r2_key)

    video_repo.update_output_path(db, video_id, output_r2_key)

    # 出力動画の再生時間を取得・保存
    if output_r2_key:
        try:
            output_url = storage_service.generate_presigned_url(
                output_r2_key, expires_in=7200
            )
            output_duration = _extract_duration(output_url)
            if output_duration is not None:
                video_repo.update_duration(db, video_id, output_duration)
        except Exception as e:
            logger.warning(
                "出力動画の再生時間取得に失敗しました video_id=%s: %s", video_id, e
            )

    video_repo.update_status(db, video_id, VideoStatus.completed)
    return output_r2_key


def process_export(video_id: uuid.UUID) -> None:
    """書き出しの背景タスクエントリポイント。

    ルーターは即座に 202 を返し、本関数が自前 DB セッションを開いて重い
    FFmpeg 処理を担当する。失敗時は ready に戻して再書き出しできるようにする
    （編集済みの clip は保持される）。
    """
    with SessionLocal() as db:
        try:
            clips = [
                {"start_time": c.start_time, "end_time": c.end_time}
                for c in clip_repo.get_by_video_id(db, video_id)
            ]
            rebuild_output(db, video_id, clips)
            logger.info("書き出し完了 video_id=%s clips=%s件", video_id, len(clips))
        except Exception as e:
            logger.exception("書き出し失敗 video_id=%s: %s", video_id, e)
            video_repo.update_status(db, video_id, VideoStatus.ready)


def recover_interrupted_exports() -> None:
    """再起動で中断された書き出しを ready に戻す（起動時リカバリ）。

    書き出しは BackgroundTasks（プロセス内）で実行されるため、processing の
    最中にプロセスが落ちると status が processing のまま取り残され、
    再書き出しもできなくなる。起動直後に実行中 job を持たない processing
    動画が残っていれば中断された書き出しとみなして ready に戻す
    （ML 解析中の動画は実行中 job を伴うので対象外。そちらのタイムアウトは
    job_reaper が処理する）。
    """
    with SessionLocal() as db:
        for video in video_repo.get_processing_without_running_job(db):
            logger.warning(
                "中断された書き出しを検知 video_id=%s → ready に戻します", video.id
            )
            video_repo.update_status(db, video.id, VideoStatus.ready)


def recover_interrupted_to_failed() -> None:
    """再起動で中断され、status=queued のまま取り残されたジョブを failed に戻す。

    状況として、アップロード完了後のプロセスが落ちた場合に、status=queued のまま取り残されることがある。
    これを failed に戻すことで、失敗として表示され、再実行できるようにする。

    起動直後に呼ばれる限り、現存しているBackgroundTasksはないため、誤検出を避けている。
    """
    with SessionLocal() as db:
        for job in job_repo.get_queued_started_null_jobs(db):
            logger.warning(
                "中断されたジョブを検知 job_id=%s → failed に戻します", job.id
            )
            job_repo.update_status(
                db=db,
                job_id=job.id,
                status=JobStatus.failed,
                error_message="プロセス再起動で中断されました",
            )
            video_repo.update_status(db, job.video_id, VideoStatus.failed)


def export_video(
    db: Session,
    video: Video,
    background_tasks: BackgroundTasks,
) -> Video:
    """ユーザー操作による動画書き出し。現在の clip 区間から出力動画を生成する。

    重い FFmpeg 処理は背景タスクに委譲し、video を processing にして即座に返す。
    """
    clips = clip_repo.get_by_video_id(db, video.id)
    if not clips:
        raise HTTPException(status_code=400, detail="書き出す切り抜きがありません")
    if video.status == VideoStatus.processing:
        raise HTTPException(status_code=409, detail="処理中のため書き出せません")

    video = video_repo.update_status(db, video.id, VideoStatus.processing)
    background_tasks.add_task(process_export, video.id)
    return video
