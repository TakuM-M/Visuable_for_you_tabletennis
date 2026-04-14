"""
RunPod Serverless Handler for Table Tennis Play Scene Detection
"""
import os
import tempfile

import httpx
import runpod

from src.pipelines.inference_pipeline import InferencePipeline
from src.pipelines.config import (
    InferencePipelineConfig,
    PlayerPoseExporterConfig,
    PlaySceneDetectionConfig,
)

TABLE_MODEL_PATH = os.getenv("TABLE_MODEL_PATH", "/workspace/models/table_detection/best.pt")
POSE_MODEL_PATH = os.getenv("POSE_MODEL_PATH", "/workspace/models/pretrained/yolo11l-pose.pt")
PLAY_CLASSIFIER_MODEL_PATH = os.getenv("PLAY_CLASSIFIER_MODEL_PATH", "/workspace/models/play_classifier/lstm_model.pth")
PLAY_CLASSIFIER_CONFIG_PATH = os.getenv("PLAY_CLASSIFIER_CONFIG_PATH", "/workspace/models/play_classifier/lstm_config.json")
DEVICE = os.getenv("ML_DEVICE", "cuda")


print(f"モデルロード開始 (device={DEVICE})")
_pipeline = InferencePipeline(
    InferencePipelineConfig(
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
)
print("モデルロード完了")


def handler(job: dict) -> dict:
    """
    RunPod Serverless ハンドラー

    Input:
        job["input"]["video_download_url"]: backendから動画を取得するURL
        job["input"]["job_id"]:             バックエンドのジョブID
        job["input"]["callback_url"]:       処理完了時にPOSTするURL

    Output:
        {"clips": [{"start_time": float, "end_time": float}, ...]}
    """
    inp = job["input"]
    video_url: str = inp["video_download_url"]
    job_id: str = inp["job_id"]
    callback_url: str = inp["callback_url"]

    clips: list[dict] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = f"{tmpdir}/input.mp4"
        
        print(f"動画ダウンロード開始 job_id={job_id}")
        with httpx.Client(verify=False, timeout=300.0) as client:
            with client.stream("GET", video_url) as response:
                response.raise_for_status()
                with open(video_path, "wb") as f:
                    for chunk in response.iter_bytes(chunk_size=65536):
                        f.write(chunk)
        print(f"動画ダウンロード完了 job_id={job_id}")

        # 推論実行
        results = _pipeline.process_video(
            input_video=video_path,
            output_dir=tmpdir,
            base_name="video",
        )

    fps = _pipeline.pose_exporter.config.video_processing.target_fps
    scenes = results["scene_detection"]["scenes"]
    clips = [
        {"start_time": round(s / fps, 2), "end_time": round(e / fps, 2)}
        for s, e in scenes
    ]
    print(f"推論完了 job_id={job_id}, scenes={len(scenes)}")

    try:
        httpx.post(
            callback_url,
            json={"job_id": job_id, "clips": clips},
            timeout=10.0,
            verify=False,
        )
        print(f"コールバック送信完了 job_id={job_id}")
    except Exception as e:
        print(f"コールバック送信失敗 job_id={job_id}: {e}")

    return {"clips": clips}


runpod.serverless.start({"handler": handler})
