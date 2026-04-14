"""
ML推論サービス

mock_app.py と同じインターフェースで、本物の InferencePipeline を呼び出す。

POST /process  → バックグラウンドで推論 → callback_url に POST
GET  /health
"""
import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from fastapi import BackgroundTasks, FastAPI
from pydantic import BaseModel

from src.pipelines.inference_pipeline import InferencePipeline
from src.pipelines.config import (
    InferencePipelineConfig,
    PlayerPoseExporterConfig,
    PlaySceneDetectionConfig,
)

app = FastAPI(title="ML Service")

# backend と共有する uploads ボリュームのマウント先
OUTPUT_DIR = Path("/app/uploads/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# モデルパス（環境変数で上書き可）
TABLE_MODEL_PATH = os.getenv(
    "TABLE_MODEL_PATH", "/workspace/models/table_detection/best.pt"
)
POSE_MODEL_PATH = os.getenv(
    "POSE_MODEL_PATH", "/workspace/models/pretrained/yolo11l-pose.pt"
)
PLAY_CLASSIFIER_MODEL_PATH = os.getenv(
    "PLAY_CLASSIFIER_MODEL_PATH", "/workspace/models/play_classifier/lstm_model.pth"
)
PLAY_CLASSIFIER_CONFIG_PATH = os.getenv(
    "PLAY_CLASSIFIER_CONFIG_PATH", "/workspace/models/play_classifier/lstm_config.json"
)
DEVICE = os.getenv("ML_DEVICE", "cpu")

_pipeline: InferencePipeline | None = None
# 同時処理は 1 件のみ（モデルはスレッドセーフでないため）
_executor = ThreadPoolExecutor(max_workers=1)


def _build_pipeline() -> InferencePipeline:
    """InferencePipeline を構築してモデルをロードする"""
    print(f"モデルロード開始 (device={DEVICE})")
    config = InferencePipelineConfig(
        pose_export=PlayerPoseExporterConfig.create_default(
            table_model_path=TABLE_MODEL_PATH,
            pose_model_path=POSE_MODEL_PATH,
            device=DEVICE,
        ),
        scene_detection=PlaySceneDetectionConfig(
            model_path=PLAY_CLASSIFIER_MODEL_PATH,
            config_path=PLAY_CLASSIFIER_CONFIG_PATH,
            device=DEVICE,
        ),
    )
    pipeline = InferencePipeline(config)
    print("モデルロード完了")
    return pipeline


def get_pipeline() -> InferencePipeline:
    """初回呼び出し時にモデルをロードして返す（以降はキャッシュ）"""
    global _pipeline
    if _pipeline is None:
        _pipeline = _build_pipeline()
    return _pipeline


class ProcessRequest(BaseModel):
    job_id: str
    video_path: str
    callback_url: str


def _run_inference(video_path: str, output_dir: str) -> dict:
    """同期推論処理（ThreadPoolExecutor で呼び出される）"""
    pipeline = get_pipeline()
    return pipeline.process_video(
        input_video=video_path,
        output_dir=output_dir,
        base_name="video",
    )


async def _run_processing(job_id: str, video_path: str, callback_url: str) -> None:
    """推論をバックグラウンドで実行してコールバックを送信"""
    output_dir = str(OUTPUT_DIR / job_id)
    clips = []

    try:
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            _executor, _run_inference, video_path, output_dir
        )

        # フレーム番号 → 秒数に変換
        fps = get_pipeline().pose_exporter.config.video_processing.target_fps
        scenes = results["scene_detection"]["scenes"]
        clips = [
            {"start_time": round(s / fps, 2), "end_time": round(e / fps, 2)}
            for s, e in scenes
        ]
        print(f"推論完了 job_id={job_id}, scenes={len(scenes)}")

    except Exception as e:
        import traceback
        print(f"推論失敗 job_id={job_id}: {e}")
        traceback.print_exc()

    # 成功・失敗にかかわらずコールバックを送信
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
    """動画処理開始（バックグラウンドで推論を実行）"""
    print(f"処理開始 job_id={request.job_id} video_path={request.video_path}")
    background_tasks.add_task(
        _run_processing, request.job_id, request.video_path, request.callback_url
    )
    return {"message": "処理開始"}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
