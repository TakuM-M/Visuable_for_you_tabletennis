import json
import os
import shutil
import subprocess
import tempfile
import threading
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
from app.repositories.job import job_repository as job_repo
from app.repositories.video import video_repository as video_repo
from app.repositories.clip import clip_repository as clip_repo
from app.repositories.notification_log import notification_log_repository as notification_log_repo
from app.schemas.clip import ClipInput
from app.services import storage_service
from app.services.video_clip_service import clip_video

logger = get_logger(__name__)


class QuotaExceededError(Exception):
    """ユーザーごとの動画本数上限に到達した際に raise する"""


class UploadRejectedError(Exception):
    """サイズ・長さの上限を超えたアップロードを拒否する際に raise する"""


def _ensure_under_quota(db: Session, user_id: uuid.UUID) -> None:
    """user_video_quota 1ユーザーあたりの動画本数上限を超えていれば QuotaExceededError を raise する"""
    if video_repo.count_by_user_id(db, user_id) >= settings.user_video_quota:
        raise QuotaExceededError(
            f"動画本数上限 {settings.user_video_quota} 本に到達しました"
        )


def _ensure_under_size_limit(total_bytes: int) -> None:
    """申告サイズが max_upload_bytes を超えていれば UploadRejectedError を raise する"""
    if total_bytes < 1:
        raise UploadRejectedError("ファイルサイズが不正です")
    if total_bytes > settings.max_upload_bytes:
        limit_gb = settings.max_upload_bytes / 1024**3
        raise UploadRejectedError(
            f"ファイルサイズが上限 {limit_gb:.1f}GB を超えています"
        )


def _ensure_under_duration_limit(duration: float | None) -> None:
    """再生時間が max_video_duration_seconds を超えていれば UploadRejectedError を raise する。

    duration が None（ffprobe 失敗）のときは判定できないので通す。ここで弾く目的は
    「GPU 実行時間と書き出し時間が現実的でない動画を落とす」ことであって、
    長さを読めなかっただけの動画まで拒否するのは行き過ぎなため。
    """
    if duration is None:
        return
    if duration > settings.max_video_duration_seconds:
        limit_min = settings.max_video_duration_seconds / 60
        raise UploadRejectedError(
            f"動画の長さが上限 {limit_min:.0f}分 を超えています"
            f"（{duration / 60:.1f}分）"
        )


def _copy_with_limit(src, dst, limit: int) -> None:
    """src を dst に書き写す。limit バイトを超えたら UploadRejectedError を raise する。

    受信しきってからサイズを見るのでは「ディスクを埋められてから気付く」ことに
    なるため、書きながら数えて超過した時点で止める。
    """
    written = 0
    while True:
        buf = src.read(_COPY_BUFFER_BYTES)
        if not buf:
            return
        written += len(buf)
        if written > limit:
            raise UploadRejectedError("アップロードサイズが上限を超えています")
        dst.write(buf)


def _chunks_total_size(upload_dir: Path) -> int:
    """保存済みチャンクの合計バイト数（meta.json は数えない）"""
    return sum(p.stat().st_size for p in upload_dir.iterdir() if p.name.isdigit())


def _extract_duration(source: Path | str) -> float | None:
    """ffprobe で動画の再生時間（秒）を取得する。

    source はローカルファイルのパスでも R2 の署名付き URL でもよい（ffprobe は
    どちらも読める）。アップロード前の長さ判定ではローカルの一時ファイルを、
    書き出し済み動画では R2 の URL を渡している。
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
                str(source),
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
# 既定値はコンテナの WORKDIR 配下。CI のようにコンテナ外で動かす場合は
# /app を作れないため、環境変数で書き込み可能な場所に差し替える
LOCAL_TMP_DIR = Path(os.getenv("LOCAL_TMP_DIR", "/app/uploads/tmp"))
LOCAL_TMP_DIR.mkdir(parents=True, exist_ok=True)

# ファイルコピー時のバッファサイズ。GB 級の動画を扱うので既定（64KB）より大きくとる
_COPY_BUFFER_BYTES = 4 * 1024 * 1024

# 書き出しの同時実行を絞るセマフォ。長時間動画ほど 1 件の FFmpeg が CPU を
# 長時間占有するため、並走を許すと API 応答まで巻き込んで遅くなる。
# BackgroundTasks はワーカープロセス内のスレッドで動くのでプロセス内ロックで足りる
_export_semaphore = threading.BoundedSemaphore(settings.export_max_concurrency)

ML_SERVICE_URL = os.getenv("ML_SERVICE_URL", "http://ml-mock:8001")
BACKEND_INTERNAL_URL = os.getenv("BACKEND_INTERNAL_URL", "http://backend:8000")
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "")
RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID", "")
INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY", "")
USE_RUNPOD = os.getenv("USE_RUNPOD", "false").lower() == "true"


def _runpod_execution_timeout_ms(source_duration: float | None) -> int:
    """動画長から RunPod の実行時間上限（ミリ秒）を見積もる。

    推論時間は動画長にほぼ比例するので base + ratio × 動画長 とし、上限で頭を打つ。
    長さが分からないときは上限値を使う。短く見積もって推論の途中で TIMED_OUT に
    されるより、長めに取って job_reaper / reconcile_runpod_jobs に拾わせるほうが
    原因が分かりやすい。
    """
    if source_duration is None:
        return settings.runpod_execution_timeout_max_seconds * 1000
    estimated = (
        settings.runpod_execution_timeout_base_seconds
        + settings.runpod_execution_timeout_ratio * source_duration
    )
    return int(min(estimated, settings.runpod_execution_timeout_max_seconds) * 1000)


def call_ml_service(r2_key: str, job_id: str, video_id: str) -> None:
    """MLサービスに処理を依頼する（RunPod or ローカル Mock）。

    BackgroundTasks / APScheduler の両方から呼ばれる前提で、内部で独自に DB セッションを開く。
    開始時に status=processing / started_at=now をセットし、ディスパッチ失敗時は
    job_service.handle_ml_failure に委譲して自動リトライ・通知を一元化する。
    """
    job_uuid = uuid.UUID(job_id)
    callback_url = f"{BACKEND_INTERNAL_URL}/internal/jobs/{job_id}/complete"
    fail_callback_url = f"{BACKEND_INTERNAL_URL}/internal/jobs/{job_id}/fail"

    # 処理開始マーク（タイムアウト判定の起点になる）。
    # 元動画長はここで拾って RunPod の実行時間上限の見積もりに使う
    with SessionLocal() as db:
        job_repo.update_status(
            db=db,
            job_id=job_uuid,
            status=JobStatus.processing,
            started_at=datetime.now(timezone.utc),
        )
        video = video_repo.update_status(
            db, uuid.UUID(video_id), VideoStatus.processing
        )
        source_duration = video.source_duration if video is not None else None

    try:
        # URL の期限は「キュー待ち + ダウンロード」を賄える長さが要る。
        # ワーカー枯渇でキューに数時間積まれてから実行されると、期限切れの URL を
        # 渡したことになり GPU が起動した直後に 403 で失敗する
        video_download_url = storage_service.generate_presigned_url(
            r2_key, expires_in=settings.ml_presigned_url_expires_seconds
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
                        },
                        # 実行時間上限を動画長から明示する。エンドポイント既定値の
                        # ままだと長い動画が推論の途中で TIMED_OUT になり、
                        # backend には「理由不明で異常終了」としか見えない
                        "policy": {
                            "executionTimeout": _runpod_execution_timeout_ms(
                                source_duration
                            )
                        },
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

    # ローカルに一時保存 → 上限チェック → R2にアップロード → サムネイル生成 → ローカル削除。
    # 長さの判定を R2 アップロードより前に置くのは、上限超過の動画を
    # 保存してから消す往復（帯域と課金）を避けるため
    try:
        with local_path.open("wb") as f:
            _copy_with_limit(file.file, f, settings.max_upload_bytes)
        source_duration = _extract_duration(local_path)
        _ensure_under_duration_limit(source_duration)
        storage_service.upload_file(str(local_path), r2_key)
        thumbnail_path = _generate_thumbnail(local_path, r2_key, source_duration)
    finally:
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
    total_bytes: int,
) -> str:
    """チャンクアップロードを初期化し、upload_idを返す。

    サイズ上限はここで先に判定する。チャンクを受け取り始めてから気付くのでは、
    上限超過が確定している動画のために回線とディスクを何 GB も使うことになる。
    """
    _ensure_under_quota(db, user_id)
    if total_chunks < 1:
        raise UploadRejectedError("チャンク数が不正です")
    _ensure_under_size_limit(total_bytes)

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
                "total_bytes": total_bytes,
            }
        )
    )
    return upload_id


def save_chunk(upload_id: str, index: int, file: UploadFile) -> None:
    """チャンクデータを一時ディレクトリに保存する。

    init での申告サイズは自己申告に過ぎないので、ここで実際の受信量を検査する。
    1 チャンク単体と累積の両方を見るのは、片方だけでは「巨大な 1 チャンク」か
    「小さいチャンクの大量送信」のどちらかでディスクを埋められるため。
    """
    upload_dir = LOCAL_TMP_DIR / upload_id
    if not upload_dir.exists():
        raise FileNotFoundError(f"Upload {upload_id} not found")

    total_chunks = json.loads((upload_dir / "meta.json").read_text())["total_chunks"]
    if not 0 <= index < total_chunks:
        raise UploadRejectedError(
            f"チャンク番号が範囲外です: index={index} total_chunks={total_chunks}"
        )

    chunk_path = upload_dir / str(index)
    # 同じ index はリトライで再送されうる。上書きされる分は累積から差し引く
    previous = chunk_path.stat().st_size if chunk_path.exists() else 0
    remaining = settings.max_upload_bytes - (_chunks_total_size(upload_dir) - previous)
    limit = min(settings.max_chunk_bytes, remaining)

    try:
        with chunk_path.open("wb") as f:
            _copy_with_limit(file.file, f, limit)
    except UploadRejectedError:
        # 書きかけを残すと次のチャンクの累積計算を狂わせる
        chunk_path.unlink(missing_ok=True)
        raise


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
    merged_path = LOCAL_TMP_DIR / f"{upload_id}_merged.mp4"
    upload_dir = LOCAL_TMP_DIR / upload_id
    try:
        meta = json.loads((upload_dir / "meta.json").read_text())
        total_chunks = meta["total_chunks"]

        with merged_path.open("wb") as out_f:
            for i in range(total_chunks):
                chunk_path = upload_dir / str(i)
                with chunk_path.open("rb") as chunk_f:
                    shutil.copyfileobj(chunk_f, out_f, _COPY_BUFFER_BYTES)
                # 書き写したチャンクはその場で消す。全部残したまま結合すると
                # ピーク時に動画サイズの 2 倍のディスクを占有する
                chunk_path.unlink(missing_ok=True)

        # R2 に上げる前に長さを見る。上限超過の動画を保存してから消すのでは
        # 帯域と保管の往復が無駄になる
        duration = _extract_duration(merged_path)
        _ensure_under_duration_limit(duration)

        storage_service.upload_file(str(merged_path), r2_key)
        # duration は上の上限チェックで取得済みのものを使い回す（ffprobe を二度
        # 走らせない）。中間ファイルの削除は finally に集約している
        thumbnail_path = _generate_thumbnail(merged_path, r2_key, duration)
        with SessionLocal() as db:
            if duration is not None:
                video_repo.update_source_duration(db, uuid.UUID(video_id), duration)
            if thumbnail_path is not None:
                video_repo.update_thumbnail_path(
                    db, uuid.UUID(video_id), thumbnail_path
                )

        call_ml_service(r2_key, str(job_id), str(video_id))
        logger.info(
            "チャンクアップロード処理完了 video_id=%s job_id=%s duration=%s",
            video_id,
            job_id,
            duration,
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
    finally:
        # 失敗時にも必ず消す。GB 級の中間ファイルを tmp_cleaner の保持期間
        # （既定 24 時間）まで残すと、失敗が続いたときにディスクが先に尽きる
        merged_path.unlink(missing_ok=True)
        shutil.rmtree(upload_dir, ignore_errors=True)


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
        logger.info(
            "書き出し開始 video_id=%s clips=%s件 総区間=%.1f秒",
            video_id,
            len(clips),
            sum(c["end_time"] - c["start_time"] for c in clips),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            # R2 から元動画をローカル一時ファイルにダウンロード。
            # timeout は 1 回のソケット操作の上限であってダウンロード全体の
            # 上限ではない。全体に上限を掛けると GB 級の動画が落ちるので、
            # 「無応答が続いたら諦める」形にする
            local_input = os.path.join(tmpdir, "input.mp4")
            presigned_url = storage_service.generate_presigned_url(video.storage_path)
            with httpx.Client(
                timeout=httpx.Timeout(
                    settings.source_download_read_timeout_seconds, connect=30.0
                )
            ) as client:
                with client.stream("GET", presigned_url) as response:
                    response.raise_for_status()
                    with open(local_input, "wb") as f:
                        for chunk in response.iter_bytes(
                            chunk_size=_COPY_BUFFER_BYTES
                        ):
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

    長い動画ほど 1 件の書き出しが CPU を長時間占有するため、セマフォで同時実行を
    絞る。順番待ちが長引いたときも ready に戻して、背景タスクのスレッドを
    無期限に抱え込まない（詰まると API 応答そのものが返らなくなる）。
    """
    if not _export_semaphore.acquire(timeout=settings.export_queue_timeout_seconds):
        logger.warning("書き出しの順番待ちがタイムアウト video_id=%s", video_id)
        with SessionLocal() as db:
            video_repo.update_status(db, video_id, VideoStatus.ready)
        return

    try:
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
    finally:
        _export_semaphore.release()


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

    video_repo.update_status(db, video.id, VideoStatus.processing)
    background_tasks.add_task(process_export, video.id)
    return video
