"""
模擬MLサービス

本物のMLパイプラインが完成するまでの代替サービス。
5秒後に偽のクリップタイムスタンプをバックエンドに返す。
動画のカット・結合はバックエンドが行う。
"""
import asyncio
import os
import subprocess

import httpx
from fastapi import BackgroundTasks, FastAPI
from pydantic import BaseModel

app = FastAPI(title="Mock ML Service")


class ProcessRequest(BaseModel):
    job_id: str
    video_path: str
    callback_url: str


def get_video_duration(video_path: str) -> float:
    """ffprobeで動画の長さ（秒）を取得する（ローカルパス・HTTP URL両対応）"""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    return float(result.stdout.strip())


async def run_mock_processing(job_id: str, video_path: str, callback_url: str) -> None:
    """模擬処理：5秒待ってからクリップタイムスタンプをコールバックで送信"""
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

    try:
        headers = {}
        api_key = os.getenv("INTERNAL_API_KEY")
        if api_key:
            headers["X-Internal-Api-Key"] = api_key

        async with httpx.AsyncClient() as client:
            response = await client.post(
                callback_url,
                json={"job_id": job_id, "clips": clips},
                headers=headers,
                timeout=30.0,
            )
            response.raise_for_status()
        print(f"コールバック送信完了 job_id={job_id} status={response.status_code} clips={len(clips)}件")
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
