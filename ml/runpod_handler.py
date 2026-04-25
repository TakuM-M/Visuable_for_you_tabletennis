"""
RunPod Serverless Handler for Table Tennis Play Scene Detection
"""
import json
import os
import tempfile

import httpx
import runpod

from src.pipelines.inference_pipeline import InferencePipeline
from src.pipelines.config import (
    InferencePipelineConfig,
    PlayerPoseExporterConfig,
    TableDetectionConfig,
    PoseTrackingConfig,
    PlayerClassificationConfig,
    TrackingExportConfig,
    VideoProcessingConfig,
    PlaySceneDetectionConfig,
)

# 設定ファイルの読み込み
CONFIG_PATH = os.getenv("PIPELINE_CONFIG_PATH", "/workspace/configs/runpod_config.json")

with open(CONFIG_PATH, "r") as f:
    config_dict = json.load(f)
print(f"設定ファイル読み込み完了: {CONFIG_PATH}")

# 環境変数によるモデルパスの上書き（Network Volume対応）
models = config_dict["models"]
TABLE_MODEL_PATH = os.getenv("TABLE_MODEL_PATH", models["table_detection"])
POSE_MODEL_PATH = os.getenv("POSE_MODEL_PATH", models["pose_estimation"])
PLAY_CLASSIFIER_MODEL_PATH = os.getenv("PLAY_CLASSIFIER_MODEL_PATH", models["play_classifier"])
PLAY_CLASSIFIER_CONFIG_PATH = os.getenv("PLAY_CLASSIFIER_CONFIG_PATH", models.get("play_classifier_config"))
DEVICE = os.getenv("ML_DEVICE", config_dict.get("device", "cuda"))


def _build_pipeline_config() -> InferencePipelineConfig:
    """JSON設定からInferencePipelineConfigを構築"""
    td = config_dict.get("table_detection", {})
    pt = config_dict.get("pose_tracking", {})
    pc = config_dict.get("player_classification", {})
    te = config_dict.get("tracking_export", {})
    vp = config_dict.get("video_processing", {})
    sd = config_dict.get("scene_detection", {})
    pl = config_dict.get("pipeline", {})

    pose_export_config = PlayerPoseExporterConfig(
        table_detection=TableDetectionConfig(
            model_path=TABLE_MODEL_PATH,
            device=DEVICE,
            cache_valid_frames=td.get("cache_valid_frames", 500),
            min_confidence=td.get("min_confidence", 0.6),
            max_detection_attempts=td.get("max_detection_attempts", 200),
        ),
        pose_tracking=PoseTrackingConfig(
            model_path=POSE_MODEL_PATH,
            device=DEVICE,
            conf_threshold=pt.get("conf_threshold", 0.5),
            iou_threshold=pt.get("iou_threshold", 0.7),
            table_distance_threshold=pt.get("table_distance_threshold", 0.2),
            min_keypoint_confidence=pt.get("min_keypoint_confidence", 0.3),
            imgsz=pt.get("imgsz", 640),
            half=pt.get("half", False),
        ),
        player_classification=PlayerClassificationConfig(
            near_table_threshold=pc.get("near_table_threshold", 0.1),
            min_tracking_frames=pc.get("min_tracking_frames", 10),
            max_players=pc.get("max_players", 2),
            max_inactive_frames=pc.get("max_inactive_frames", 30),
            min_player_score=pc.get("min_player_score", 0.3),
            recent_frames_window=pc.get("recent_frames_window", 146),
            max_consecutive_other_count=pc.get("max_consecutive_other_count", 30),
            movement_noise_threshold=pc.get("movement_noise_threshold", 5.0),
        ),
        tracking_export=TrackingExportConfig(
            min_consecutive_frames=te.get("min_consecutive_frames", 30),
            max_frame_gap=te.get("max_frame_gap", 5),
        ),
        video_processing=VideoProcessingConfig(
            target_fps=vp.get("target_fps", 30.0),
            show_progress=vp.get("show_progress", True),
            output_codec=vp.get("output_codec", "mp4v"),
        ),
        save_output=pl.get("save_output", False),
    )

    scene_detection_config = PlaySceneDetectionConfig(
        model_path=PLAY_CLASSIFIER_MODEL_PATH,
        config_path=PLAY_CLASSIFIER_CONFIG_PATH,
        device=DEVICE,
        threshold=sd.get("threshold", 0.5),
        min_scene_duration=sd.get("min_scene_duration", 10),
        batch_size=sd.get("batch_size", 64),
    )

    return InferencePipelineConfig(
        pose_export=pose_export_config,
        scene_detection=scene_detection_config,
        show_progress=pl.get("show_progress", True),
        save_output=pl.get("save_output", False),
    )

print(f"モデルロード開始 (device={DEVICE})")
_pipeline = InferencePipeline(_build_pipeline_config())
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
        with httpx.Client(timeout=300.0) as client:
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
        )
        print(f"コールバック送信完了 job_id={job_id}")
    except Exception as e:
        print(f"コールバック送信失敗 job_id={job_id}: {e}")

    return {"clips": clips}


runpod.serverless.start({"handler": handler})
