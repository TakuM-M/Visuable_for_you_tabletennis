from pathlib import Path
from typing import Dict, Any, Optional

from src.pipelines.config import InferencePipelineConfig
from src.pipelines.player_pose_exporter import PlayerPoseExporter
from src.pipelines.play_scene_detector import PlaySceneDetector
from src.core.exceptions import PipelineError
from src.visualization.result_visualizer import save_prediction_graph


class InferencePipeline:
    """End-to-End推論パイプライン"""

    def __init__(self, config: InferencePipelineConfig):
        """
        初期化

        Args:
            config: 推論パイプライン全体の設定
        """
        self.config = config
        self.show_progress = config.show_progress
        self.pose_exporter = PlayerPoseExporter(config.pose_export)
        self.scene_detector = PlaySceneDetector(config.scene_detection)

    def process_video(
        self,
        input_video: str,
        output_dir: str,
        base_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        動画を処理してプレーシーンを検出・抽出

        Args:
            input_video: 入力動画パス
            output_dir: 出力ディレクトリ
            base_name: 出力ファイルのベース名（Noneの場合は入力動画名を使用）

        Returns:
            処理結果の統計情報

        Raises:
            PipelineError: パイプライン処理中にエラーが発生した場合
        """
        input_path = Path(input_video)
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        if base_name is None:
            base_name = input_path.stem

        save_intermediate_files = self.config.save_intermediate_files
        pose_video_path = output_dir_path / f"{base_name}_poses.mp4"
        pose_csv_path = output_dir_path / f"{base_name}_poses.csv"

        try:
            print(f"\n{'=' * 70}")
            print("Task1: 骨格データ抽出")
            print(f"{'=' * 70}\n")

            pose_results = self.pose_exporter.process_video(
                input_video=str(input_video),
                output_video=str(pose_video_path) if save_intermediate_files else None,
                csv_output=str(pose_csv_path) if save_intermediate_files else None,
            )

            # Task2: プレーシーン検出
            print(f"\n{'=' * 70}")
            print("Task2: プレーシーン検出")
            print(f"{'=' * 70}\n")

            result_df, scenes = self.scene_detector.detect_from_exporter(
                exporter=self.pose_exporter.tracking_exporter,
                show_progress=self.show_progress,
            )

            # 予測結果を保存（元動画のFPSで時間変換）
            video_fps = pose_results.get(
                "video_fps", self.pose_exporter.config.video_processing.target_fps
            )
            if save_intermediate_files:
                self.scene_detector.save_results(
                    result_df=result_df,
                    scenes=scenes,
                    output_dir=str(output_dir_path),
                    base_name=base_name,
                    fps=video_fps,
                )
                save_prediction_graph(
                    result_df=result_df,
                    scenes=scenes,
                    output_dir=output_dir_path,
                    base_name=base_name,
                    threshold=self.scene_detector.threshold,
                    fps=video_fps,
                )

            # 統計情報をまとめる
            results = {
                "input_video": str(input_video),
                "output_dir": str(output_dir_path),
                "pose_export": {
                    "pose_video": str(pose_video_path),
                    "pose_csv": str(pose_csv_path),
                    "processed_frames": pose_results["processed_frames"],
                    "player_ids": pose_results["player_ids"],
                    "video_fps": pose_results.get("video_fps"),
                },
                "scene_detection": {
                    "total_scenes": len(scenes),
                    "scenes": scenes,
                    "threshold": self.scene_detector.threshold,
                    "min_scene_duration": self.scene_detector.min_scene_duration,
                },
                "output_files": {
                    "pose_video": str(pose_video_path)
                    if save_intermediate_files
                    else None,
                    "pose_csv": str(pose_csv_path) if save_intermediate_files else None,
                },
            }

            print(f"\n{'=' * 70}")
            print("End-to-End推論パイプライン完了")
            print(f"{'=' * 70}")
            print(f"\n主要な出力:")
            if save_intermediate_files:
                print(f"  骨格データ動画: {pose_video_path}")
                print(f"  骨格データCSV: {pose_csv_path}")
            print(f"\n処理結果:")
            print(f"  検出シーン数: {len(scenes)}")
            print(f"  プレイヤー数: {len(pose_results['player_ids'])}")
            print(f"{'=' * 70}\n")

            return results

        except Exception as e:
            raise PipelineError(
                f"推論パイプライン処理中にエラーが発生しました: {str(e)}"
            ) from e
