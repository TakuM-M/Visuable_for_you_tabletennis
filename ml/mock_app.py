"""
模擬MLサービス

本物のMLパイプラインが完成するまでの代替サービス。
5秒後に偽のクリップデータをバックエンドに返す。
FFmpegでプレーシーンを連結した動画を生成する。
"""
import asyncio
import subprocess
import uuid
from pathlib import Path

import httpx
from fastapi import BackgroundTasks, FastAPI
from pydantic import BaseModel

app = FastAPI(title="Mock ML Service")

OUTPUT_DIR = Path("/app/uploads/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class ProcessRequest(BaseModel):
    job_id: str
    video_path: str
    callback_url: str


def get_video_duration(video_path: str) -> float:
    """ffprobeで動画の長さ（秒）を取得する"""
    result = subprocess.run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ], capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


async def run_mock_processing(job_id: str, video_path: str, callback_url: str) -> None:
    """模擬処理：5秒待ってからFFmpegで動画を連結してコールバックで送信"""
    await asyncio.sleep(5)

    # 動画の長さに合わせてプレーシーンを計算（前半1/3と後半1/3をプレーシーンとする）
    try:
        duration = get_video_duration(video_path)
    except Exception as e:
        print(f"動画長さ取得失敗: {e}")
        duration = 30.0

    t1 = round(duration * 0.0, 1)
    t2 = round(duration * 0.33, 1)
    t3 = round(duration * 0.66, 1)
    t4 = round(duration * 1.0, 1)

    clips = [
        {"start_time": t1, "end_time": t2},
        {"start_time": t3, "end_time": t4},
    ]

    output_path = str(OUTPUT_DIR / f"{uuid.uuid4()}.mp4")

    try:
        # FFmpegで各シーンを切り出して連結する
        # 1. 各シーンを再エンコードして一時ファイルを作成（キーフレーム問題を回避）
        temp_files = []
        for i, clip in enumerate(clips):
            temp_path = str(OUTPUT_DIR / f"temp_{job_id}_{i}.mp4")
            subprocess.run([
                "ffmpeg", "-y",
                "-i", video_path,
                "-ss", str(clip["start_time"]),
                "-to", str(clip["end_time"]),
                "-c:v", "libx264",
                "-c:a", "aac",
                temp_path,
            ], check=True, capture_output=True)
            temp_files.append(temp_path)

        # 2. 連結リストファイルを作成
        list_path = str(OUTPUT_DIR / f"list_{job_id}.txt")
        with open(list_path, "w") as f:
            for temp_path in temp_files:
                f.write(f"file '{temp_path}'\n")

        # 3. 連結して1つの動画に（-movflags +faststart でブラウザ再生を最適化）
        subprocess.run([
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", list_path,
            "-c", "copy",
            "-movflags", "+faststart",
            output_path,
        ], check=True, capture_output=True)

        # 4. 一時ファイルを削除
        for temp_path in temp_files:
            Path(temp_path).unlink(missing_ok=True)
        Path(list_path).unlink(missing_ok=True)

        print(f"動画連結完了: {output_path}")

    except subprocess.CalledProcessError as e:
        print(f"FFmpeg失敗 job_id={job_id}: {e.stderr.decode()}")
        output_path = ""

    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                callback_url,
                json={"job_id": job_id, "clips": clips, "output_path": output_path},
                timeout=10.0,
            )
    except Exception as e:
        print(f"コールバック送信失敗 job_id={job_id}: {e}")


@app.post("/process")
async def process_video(
    request: ProcessRequest, background_tasks: BackgroundTasks
) -> dict:
    """動画処理開始（バックグラウンドで模擬処理を実行）"""
    print(f"処理開始 job_id={request.job_id} video_path={request.video_path}")
    background_tasks.add_task(
        run_mock_processing, request.job_id, request.video_path, request.callback_url
    )
    return {"message": "処理開始"}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
