"""
動画切り出し・連結パイプライン

検出されたプレーシーンのみを切り抜いて連結した動画を作成する
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from tqdm import tqdm

from src.pipelines.exceptions import VideoInputError, VideoProcessingError, ExportError


class VideoComposer:
    """プレーシーン動画作成パイプライン"""

    DEFAULT_CODEC = 'mp4v'
    DEFAULT_OUTPUT_FPS = 30.0

    def __init__(
        self,
        output_codec: str = DEFAULT_CODEC,
        output_fps: Optional[float] = None,
        add_scene_info: bool = True,
        show_progress: bool = True
    ):
        """
        初期化

        Args:
            output_codec: 出力動画のコーデック
            output_fps: 出力動画のFPS（Noneの場合は元動画のFPSを使用）
            add_scene_info: シーン情報をオーバーレイするか
            show_progress: プログレスバーを表示するか
        """
        self.output_codec = output_codec
        self.output_fps = output_fps
        self.add_scene_info = add_scene_info
        self.show_progress = show_progress

        print(f"VideoComposer初期化完了:")
        print(f"  コーデック: {self.output_codec}")
        print(f"  出力FPS: {self.output_fps or '元動画と同じ'}")
        print(f"  シーン情報表示: {self.add_scene_info}")

    def compose_play_scenes(
        self,
        input_video_path: str,
        scenes: List[Tuple[int, int]],
        output_path: str,
        frame_interval: int = 1
    ) -> Dict[str, Any]:
        """
        プレー中シーンのみを切り抜いた動画を作成

        Args:
            input_video_path: 元の動画ファイルパス
            scenes: [(start_frame, end_frame), ...] シーンのリスト
            output_path: 出力動画ファイルパス
            frame_interval: 元動画での処理時のフレーム間隔

        Returns:
            作成結果の統計情報

        Raises:
            VideoInputError: 入力動画が開けない場合
            VideoProcessingError: 動画処理中にエラーが発生した場合
            ExportError: 出力動画の保存に失敗した場合
        """
        if not scenes:
            print("警告: 切り抜くシーンがありません")
            return {
                'total_scenes': 0,
                'total_frames': 0,
                'duration_sec': 0.0,
                'output_path': None
            }

        print(f"\nプレー中シーンを切り抜いた動画を作成中...")
        print(f"  入力動画: {input_video_path}")
        print(f"  出力動画: {output_path}")
        print(f"  シーン数: {len(scenes)}")

        # 元の動画を開く
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise VideoInputError(input_video_path, "動画ファイルを開けませんでした")

        try:
            # 動画情報を取得
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

            # 出力FPSを決定
            output_fps = self.output_fps if self.output_fps is not None else video_fps

            print(f"\n動画情報:")
            print(f"  解像度: {width}x{height}")
            print(f"  元動画FPS: {video_fps:.2f}")
            print(f"  出力FPS: {output_fps:.2f}")
            print(f"  フレーム間隔: {frame_interval}")

            # 出力ディレクトリを作成
            output_path_obj = Path(output_path)
            output_path_obj.parent.mkdir(parents=True, exist_ok=True)

            # VideoWriterの初期化
            fourcc = cv2.VideoWriter_fourcc(*self.output_codec)
            out = cv2.VideoWriter(str(output_path), fourcc, output_fps, (width, height))

            if not out.isOpened():
                raise VideoProcessingError(f"VideoWriterの初期化に失敗しました: {output_path}")

            # 各シーンを処理
            total_written_frames = 0
            stats = self._process_scenes(
                cap=cap,
                out=out,
                scenes=scenes,
                frame_interval=frame_interval,
                output_fps=output_fps,
                width=width,
                height=height
            )

            return stats

        finally:
            cap.release()
            if 'out' in locals():
                out.release()

    def _process_scenes(
        self,
        cap: cv2.VideoCapture,
        out: cv2.VideoWriter,
        scenes: List[Tuple[int, int]],
        frame_interval: int,
        output_fps: float,
        width: int,
        height: int
    ) -> Dict[str, Any]:
        """
        各シーンを処理して動画に書き込む

        Args:
            cap: VideoCapture
            out: VideoWriter
            scenes: シーンのリスト
            frame_interval: フレーム間隔
            output_fps: 出力FPS
            width: フレーム幅
            height: フレーム高さ

        Returns:
            処理統計
        """
        total_written_frames = 0

        iterator = tqdm(scenes, desc="シーン処理中") if self.show_progress else scenes

        for scene_idx, (start_frame, end_frame) in enumerate(iterator):
            # start_frameとend_frameは処理済みフレーム番号なので、
            # 元動画のフレーム番号に変換（整数に変換）
            original_start = int(start_frame * frame_interval)
            original_end = int(end_frame * frame_interval)

            # シーンの開始位置にシーク
            cap.set(cv2.CAP_PROP_POS_FRAMES, original_start)

            # このシーンのフレームを読み込んで書き込む
            for frame_idx in range(original_start, original_end + 1, frame_interval):
                ret, frame = cap.read()
                if not ret:
                    break

                # シーン情報をオーバーレイ
                if self.add_scene_info:
                    frame = self._add_scene_overlay(
                        frame=frame,
                        scene_idx=scene_idx,
                        total_scenes=len(scenes),
                        current_time=total_written_frames / output_fps,
                        height=height
                    )

                out.write(frame)
                total_written_frames += 1

        # 統計計算
        output_duration = total_written_frames / output_fps

        stats = {
            'total_scenes': len(scenes),
            'total_frames': total_written_frames,
            'duration_sec': output_duration,
            'output_fps': output_fps,
            'output_path': str(out)
        }

        print(f"\n✓ 動画作成完了:")
        print(f"  出力フレーム数: {total_written_frames}")
        print(f"  出力時間: {output_duration:.1f}秒 ({output_duration/60:.2f}分)")
        print(f"  出力FPS: {output_fps}")

        return stats

    def _add_scene_overlay(
        self,
        frame: np.ndarray,
        scene_idx: int,
        total_scenes: int,
        current_time: float,
        height: int
    ) -> np.ndarray:
        """
        フレームにシーン情報をオーバーレイ

        Args:
            frame: 元フレーム
            scene_idx: シーンインデックス
            total_scenes: 総シーン数
            current_time: 現在の再生時間（秒）
            height: フレーム高さ

        Returns:
            オーバーレイ済みフレーム
        """
        # シーン番号と再生時間をオーバーレイ
        text = f"Scene {scene_idx + 1}/{total_scenes} | Time: {current_time:.1f}s"

        cv2.putText(
            frame,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        return frame

    def create_highlights_video(
        self,
        input_video_path: str,
        scenes: List[Tuple[int, int]],
        output_path: str,
        frame_interval: int = 1,
        max_scenes: Optional[int] = None,
        min_scene_duration: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        ハイライト動画を作成（シーンのフィルタリング機能付き）

        Args:
            input_video_path: 元の動画ファイルパス
            scenes: [(start_frame, end_frame), ...] シーンのリスト
            output_path: 出力動画ファイルパス
            frame_interval: 元動画での処理時のフレーム間隔
            max_scenes: 最大シーン数（長いシーンを優先）
            min_scene_duration: 最小シーン長（フレーム数）

        Returns:
            作成結果の統計情報
        """
        # シーンのフィルタリング
        filtered_scenes = self._filter_scenes(
            scenes=scenes,
            max_scenes=max_scenes,
            min_scene_duration=min_scene_duration
        )

        if not filtered_scenes:
            print("警告: フィルタリング後のシーンがありません")
            return {
                'total_scenes': 0,
                'filtered_scenes': 0,
                'total_frames': 0,
                'duration_sec': 0.0,
                'output_path': None
            }

        print(f"\nハイライト動画作成:")
        print(f"  元のシーン数: {len(scenes)}")
        print(f"  フィルタ後: {len(filtered_scenes)}")

        # 通常の動画作成処理を実行
        stats = self.compose_play_scenes(
            input_video_path=input_video_path,
            scenes=filtered_scenes,
            output_path=output_path,
            frame_interval=frame_interval
        )

        stats['filtered_scenes'] = len(filtered_scenes)
        stats['original_scenes'] = len(scenes)

        return stats

    def _filter_scenes(
        self,
        scenes: List[Tuple[int, int]],
        max_scenes: Optional[int] = None,
        min_scene_duration: Optional[int] = None
    ) -> List[Tuple[int, int]]:
        """
        シーンをフィルタリング

        Args:
            scenes: 元のシーンリスト
            max_scenes: 最大シーン数
            min_scene_duration: 最小シーン長

        Returns:
            フィルタリング済みシーンリスト
        """
        filtered = scenes.copy()

        # 最小シーン長でフィルタ
        if min_scene_duration is not None:
            filtered = [
                (start, end) for start, end in filtered
                if (end - start + 1) >= min_scene_duration
            ]

        # 長いシーンを優先してmax_scenesまで絞る
        if max_scenes is not None and len(filtered) > max_scenes:
            # シーンの長さでソート（降順）
            filtered_with_duration = [
                (start, end, end - start + 1) for start, end in filtered
            ]
            filtered_with_duration.sort(key=lambda x: x[2], reverse=True)

            # 上位max_scenesを取得してフレーム順に並べ直す
            filtered = [
                (start, end) for start, end, _ in filtered_with_duration[:max_scenes]
            ]
            filtered.sort(key=lambda x: x[0])

        return filtered
