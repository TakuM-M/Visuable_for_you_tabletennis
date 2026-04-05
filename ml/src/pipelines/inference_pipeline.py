"""
End-to-End推論パイプライン

動画から骨格データ抽出→プレーシーン検出→ハイライト動画作成までを統合
"""
from pathlib import Path
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import pandas as pd

from src.pipelines.player_pose_exporter import PlayerPoseExporter
from src.pipelines.play_scene_detector import PlaySceneDetector
from src.pipelines.config import (
    InferencePipelineConfig,
    PlayerPoseExporterConfig,
    PlaySceneDetectionConfig,
)
from src.pipelines.exceptions import PipelineError


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

    @classmethod
    def create_default(
        cls,
        table_model_path: str,
        pose_model_path: str,
        play_classifier_model_path: str,
        device: str = 'cuda',
        detection_threshold: float = 0.5,
        min_scene_duration: int = 10
    ) -> 'InferencePipeline':
        """
        デフォルト設定でInferencePipelineを作成

        Args:
            table_model_path: 卓球台検出モデルのパス
            pose_model_path: 姿勢推定モデルのパス
            play_classifier_model_path: プレー検知モデルのパス
            device: 使用デバイス
            detection_threshold: プレー中判定の閾値
            min_scene_duration: 最小シーン長（フレーム数）

        Returns:
            デフォルト設定のInferencePipeline
        """
        config = InferencePipelineConfig.create_default(
            table_model_path=table_model_path,
            pose_model_path=pose_model_path,
            play_classifier_model_path=play_classifier_model_path,
            device=device,
            detection_threshold=detection_threshold,
            min_scene_duration=min_scene_duration
        )

        return cls(config=config)

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

        # configから設定を取得
        save_output = self.config.save_output

        pose_video_path = output_dir_path / f"{base_name}_poses.mp4"
        pose_csv_path = output_dir_path / f"{base_name}_poses.csv"

        try:
            print(f"\n{'='*70}")
            print("Task1: 骨格データ抽出")
            print(f"{'='*70}\n")

            pose_results = self.pose_exporter.process_video(
                input_video=str(input_video),
                output_video=str(pose_video_path) if save_output else None,
                csv_output=str(pose_csv_path) if save_output else None,
                show_progress=self.show_progress
            )

            # Task2: プレーシーン検出
            print(f"\n{'='*70}")
            print("Task2: プレーシーン検出")
            print(f"{'='*70}\n")

            result_df, scenes = self.scene_detector.detect_from_exporter(
                exporter=self.pose_exporter.tracking_exporter,
                show_progress=self.show_progress
            )

            # 予測結果を保存
            if save_output:
                self.scene_detector.save_results(
                    result_df=result_df,
                    scenes=scenes,
                    output_dir=str(output_dir_path),
                    base_name=base_name,
                    fps=self.pose_exporter.config.video_processing.target_fps
                )

            # グラフ作成
            if save_output:
                self._save_prediction_graph(
                    result_df=result_df,
                    scenes=scenes,
                    output_dir=output_dir_path,
                    base_name=base_name,
                    threshold=self.scene_detector.threshold,
                    fps=self.pose_exporter.config.video_processing.target_fps
                )

            # 統計情報をまとめる
            results = {
                'input_video': str(input_video),
                'output_dir': str(output_dir_path),
                'pose_export': {
                    'pose_video': str(pose_video_path),
                    'pose_csv': str(pose_csv_path),
                    'processed_frames': pose_results['processed_frames'],
                    'player_ids': pose_results['player_ids']
                },
                'scene_detection': {
                    'total_scenes': len(scenes),
                    'scenes': scenes,
                    'threshold': self.scene_detector.threshold,
                    'min_scene_duration': self.scene_detector.min_scene_duration
                },
                'output_files': {
                    'pose_video': str(pose_video_path) if save_output else None,
                    'pose_csv': str(pose_csv_path) if save_output else None,
                }
            }

            print(f"\n{'='*70}")
            print("End-to-End推論パイプライン完了")
            print(f"{'='*70}")
            print(f"\n主要な出力:")
            if save_output:
                print(f"  骨格データ動画: {pose_video_path}")
                print(f"  骨格データCSV: {pose_csv_path}")
            print(f"\n処理結果:")
            print(f"  検出シーン数: {len(scenes)}")
            print(f"  プレイヤー数: {len(pose_results['player_ids'])}")
            print(f"{'='*70}\n")

            return results

        except Exception as e:
            raise PipelineError(f"推論パイプライン処理中にエラーが発生しました: {str(e)}") from e

    def _save_prediction_graph(
        self,
        result_df: pd.DataFrame,
        scenes: list,
        output_dir: Path,
        base_name: str,
        threshold: float,
        fps: float
    ):
        """
        予測結果のグラフを保存

        Args:
            result_df: 予測結果のDataFrame
            scenes: 検出されたシーン
            output_dir: 出力ディレクトリ
            base_name: ベース名
            threshold: 判定閾値
            fps: FPS
        """
        print(f"\n予測グラフを作成中...")

        plt.figure(figsize=(16, 6))

        # 予測確率をプロット
        plt.subplot(2, 1, 1)
        plt.plot(result_df['frame'], result_df['probability'], linewidth=1, alpha=0.7, color='blue')
        plt.axhline(y=threshold, color='red', linestyle='--', label=f'閾値 ({threshold})')
        plt.fill_between(
            result_df['frame'],
            0,
            result_df['probability'],
            where=(result_df['probability'] >= threshold),
            alpha=0.3,
            color='green',
            label='プレー中'
        )
        plt.xlabel('フレーム番号')
        plt.ylabel('プレー中確率')
        plt.title('プレーシーン予測結果 - 確率')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(-0.05, 1.05)

        # 予測ラベルをプロット
        plt.subplot(2, 1, 2)
        plt.plot(result_df['frame'], result_df['prediction'], linewidth=1.5, color='green')
        plt.fill_between(
            result_df['frame'],
            0,
            result_df['prediction'],
            alpha=0.3,
            color='green'
        )

        # 検出されたシーンを赤線でマーク
        for start, end in scenes:
            plt.axvline(x=start, color='red', linestyle=':', alpha=0.5, linewidth=1)
            plt.axvline(x=end, color='red', linestyle=':', alpha=0.5, linewidth=1)

        plt.xlabel('フレーム番号')
        plt.ylabel('予測ラベル (0: 非プレー, 1: プレー)')
        plt.title(f'プレーシーン予測結果 - 分類 (検出シーン数: {len(scenes)})')
        plt.grid(True, alpha=0.3)
        plt.ylim(-0.1, 1.1)
        plt.yticks([0, 1], ['非プレー', 'プレー'])

        plt.tight_layout()

        # グラフを保存
        output_graph_path = output_dir / f"{base_name}_prediction_graph.png"
        plt.savefig(output_graph_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"予測グラフを保存しました: {output_graph_path}")
