"""
RunPod Serverless Handler for Table Tennis Play Scene Detection
"""

import json
import os
import tempfile
import time
import traceback

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


CONFIG_PATH = os.getenv("PIPELINE_CONFIG_PATH", "/workspace/configs/runpod_config.json")
with open(CONFIG_PATH, "r") as f:
    config_dict = json.load(f)
print(f"設定ファイル読み込み完了: {CONFIG_PATH}")

models = config_dict["models"]
TABLE_MODEL_PATH = os.getenv("TABLE_MODEL_PATH", models["table_detection"])
POSE_MODEL_PATH = os.getenv("POSE_MODEL_PATH", models["pose_estimation"])
PLAY_CLASSIFIER_MODEL_PATH = os.getenv(
    "PLAY_CLASSIFIER_MODEL_PATH", models["play_classifier"]
)
PLAY_CLASSIFIER_CONFIG_PATH = os.getenv(
    "PLAY_CLASSIFIER_CONFIG_PATH", models.get("play_classifier_config")
)
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
            # 学習データと同じ 15fps 間引き。設定ファイル側で明示的に上書きしない
            # 限りこの値を使う（30fps にすると学習時と条件がずれ、GPU 時間も倍になる）
            target_fps=vp.get("target_fps", 15.0),
            show_progress=vp.get("show_progress", True),
            output_codec=vp.get("output_codec", "mp4v"),
        ),
        save_intermediate_files=pl.get("save_intermediate_files", False),
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
        save_intermediate_files=pl.get("save_intermediate_files", False),
    )


print(f"モデルロード開始 (device={DEVICE})")
_pipeline = InferencePipeline(_build_pipeline_config())
print("モデルロード完了")


CALLBACK_MAX_ATTEMPTS = 3
CALLBACK_TIMEOUT_SECONDS = 30.0


def _post_callback(url: str, payload: dict, job_id: str, label: str) -> None:
    """backendへコールバックを送る。送信自体に失敗しても例外は投げない。

    数回リトライするのは、ここで一度で諦めると backend からは
    「RunPod は COMPLETED なのにコールバック未達」にしか見えず、推論を
    まるごとやり直すことになるため。長い動画ほどそのやり直しが高くつく。
    リトライしても駄目な場合はGPU側からできることが無く、backend側の
    タイムアウト監視（job_reaper / reconcile_runpod_jobs）に委ねるしかない。
    """
    if not url:
        print(f"{label}コールバックURLが未指定のため送信をスキップ job_id={job_id}")
        return

    headers = {}
    api_key = os.getenv("INTERNAL_API_KEY")
    if api_key:
        headers["X-Internal-Api-Key"] = api_key

    for attempt in range(1, CALLBACK_MAX_ATTEMPTS + 1):
        try:
            response = httpx.post(
                url, json=payload, headers=headers, timeout=CALLBACK_TIMEOUT_SECONDS
            )
            response.raise_for_status()
            print(
                f"{label}コールバック送信完了 job_id={job_id} status={response.status_code}"
            )
            return
        except Exception as e:
            print(
                f"{label}コールバック送信失敗 ({attempt}/{CALLBACK_MAX_ATTEMPTS}) "
                f"job_id={job_id}: {e}"
            )
            if attempt < CALLBACK_MAX_ATTEMPTS:
                time.sleep(2**attempt)


def _run_inference(video_url: str, job_id: str) -> list[dict]:
    """動画をダウンロードして推論し、クリップ区間（秒）のリストを返す"""
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = f"{tmpdir}/input.mp4"

        print(f"動画ダウンロード開始 job_id={job_id}")
        started = time.monotonic()
        # timeout は 1 回のソケット操作にかかる上限であってダウンロード全体の
        # 上限ではないので、GB 級でもこの値のままでよい
        with httpx.Client(timeout=300.0) as client:
            with client.stream("GET", video_url) as response:
                response.raise_for_status()
                with open(video_path, "wb") as f:
                    for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                        f.write(chunk)
        size_mb = os.path.getsize(video_path) / 1024**2
        print(
            f"動画ダウンロード完了 job_id={job_id} size={size_mb:.1f}MB "
            f"所要={time.monotonic() - started:.1f}秒"
        )

        # 動画長・処理時間は「長い動画で実行時間上限に収まるか」を後から
        # 検証するための材料になるので必ず残す
        inference_started = time.monotonic()
        results = _pipeline.process_video(
            input_video=video_path,
            output_dir=tmpdir,
            base_name="video",
        )
        inference_seconds = time.monotonic() - inference_started

    # フレーム番号は元動画の実フレーム番号のため、元動画のFPSで割る
    video_fps = results["pose_export"]["video_fps"]
    scenes = results["scene_detection"]["scenes"]
    total_frames = results["pose_export"]["processed_frames"]
    print(
        f"推論時間 job_id={job_id} 処理フレーム={total_frames} "
        f"所要={inference_seconds:.1f}秒"
    )
    return [
        {"start_time": round(s / video_fps, 2), "end_time": round(e / video_fps, 2)}
        for s, e in scenes
    ]


def handler(job: dict) -> dict:
    """
    RunPod Serverless ハンドラー

    Input:
        job["input"]["video_download_url"]: backendから動画を取得するURL
        job["input"]["job_id"]:             バックエンドのジョブID
        job["input"]["callback_url"]:       処理完了時にPOSTするURL
        job["input"]["fail_callback_url"]:  処理失敗時にPOSTするURL（任意）

    Output:
        成功時: {"clips": [{"start_time": float, "end_time": float}, ...]}
        失敗時: {"error": str, "clips": []}
    """
    inp = job["input"]
    video_url: str = inp["video_download_url"]
    job_id: str = inp["job_id"]
    callback_url: str = inp["callback_url"]
    fail_callback_url: str = inp.get("fail_callback_url", "")

    try:
        clips = _run_inference(video_url, job_id)
    except Exception as e:
        # 例外を正常 return する。再送出するリトライの主導権は backend に一本化する。
        print(f"推論失敗 job_id={job_id}: {e}")
        traceback.print_exc()
        error = f"{type(e).__name__}: {e}"
        _post_callback(
            fail_callback_url, {"job_id": job_id, "error": error}, job_id, "失敗"
        )
        return {"error": error, "clips": []}

    print(f"推論完了 job_id={job_id}, scenes={len(clips)}")
    _post_callback(callback_url, {"job_id": job_id, "clips": clips}, job_id, "完了")
    return {"clips": clips}


runpod.serverless.start({"handler": handler})
