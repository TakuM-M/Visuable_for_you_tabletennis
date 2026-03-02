"""
模擬MLサービス

本物のMLパイプラインが完成するまでの代替サービス。
5秒後に偽のクリップデータをバックエンドに返す。
"""
import asyncio

import httpx
from fastapi import BackgroundTasks, FastAPI
from pydantic import BaseModel

app = FastAPI(title="Mock ML Service")


class ProcessRequest(BaseModel):
    job_id: str
    video_path: str
    callback_url: str


async def run_mock_processing(job_id: str, callback_url: str) -> None:
    """模擬処理：5秒待ってからクリップデータをコールバックで送信"""
    await asyncio.sleep(5)

    # 偽のクリップデータ（実際のMLが完成したら本物の検出結果に差し替える）
    clips = [
        {"start_time": 10.0, "end_time": 25.0},
        {"start_time": 45.0, "end_time": 70.0},
        {"start_time": 100.0, "end_time": 130.0},
    ]

    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                callback_url,
                json={"job_id": job_id, "clips": clips},
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
        run_mock_processing, request.job_id, request.callback_url
    )
    return {"message": "処理開始"}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
