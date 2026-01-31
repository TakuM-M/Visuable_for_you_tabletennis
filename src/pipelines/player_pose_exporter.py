"""
プレイヤー姿勢検出・エクスポートパイプライン
"""
import cv2
from pathlib import Path
from typing import Optional, Dict, Any, Set, Tuple
from tqdm import tqdm

from src.core.data_classes import TableInfo
from src.detection.table_detector import TableDetector
from src.detection.yolopose_tracker import YOLOPose_Tracker
from src.detection.player_classifier import PlayerClassifier
from src.detection.tracking_exporter import TrackingExporter
from src.visualization.player_classifier_visualizer import PlayerClassifierVisualizer

from src.pipelines.config import PipelineConfig
from src.pipelines.exceptions import (
    TableDetectionError,
    VideoInputError,
    VideoProcessingError,
    ExportError
)


class PlayerPoseExporter:
    """プレイヤーの姿勢を検出してCSVにエクスポートするパイプライン"""

    # 定数定義
    DEFAULT_FALLBACK_TOTAL_FRAMES = 10000
    VIDEO_CODEC_FOURCC = 'mp4v'

    def __init__(self, config: PipelineConfig):
        """
        Args:
            config: パイプライン設定
        """
        self.config = config
        
        # コンポーネントの初期化
        self.table_detector = TableDetector(
            yolo_model_path=config.table_detection.model_path,
            cache_valid_frames=config.table_detection.cache_valid_frames
        )
        self.pose_tracker = YOLOPose_Tracker(
            model_path=config.pose_tracking.model_path,
            device=config.pose_tracking.device
        )
        self.player_classifier = PlayerClassifier(
            max_players=config.player_classification.max_players,
            min_player_score=config.player_classification.min_player_score
        )
        self.tracking_exporter = TrackingExporter(
            min_consecutive_frames=config.tracking_export.min_consecutive_frames,
            max_frame_gap=config.tracking_export.max_frame_gap,
            min_confidence=config.pose_tracking.min_keypoint_confidence
        )
        self.visualizer = PlayerClassifierVisualizer(
            self.table_detector,
            self.pose_tracker,
            self.player_classifier
        )

    @classmethod
    def create_default(
        cls,
        table_model_path: str,
        pose_model_path: str,
        device: str = 'cuda'
    ) -> 'PlayerPoseExporter':
        """
        デフォルト設定でPlayerPoseExporterを作成

        Args:
            table_model_path: 卓球台検出モデルのパス
            pose_model_path: 姿勢推定モデルのパス
            device: 使用デバイス

        Returns:
            デフォルト設定のPlayerPoseExporter
        """
        config = PipelineConfig.create_default(
            table_model_path=table_model_path,
            pose_model_path=pose_model_path,
            device=device
        )
        return cls(config)

    def process_video(
        self,
        input_video: str,
        output_video: str,
        csv_output: str,
        target_fps: Optional[float] = None,
        show_progress: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        動画からプレイヤーの姿勢を検出してCSVに出力

        Args:
            input_video: 入力動画パス
            output_video: 出力動画パス
            csv_output: CSV出力パス
            target_fps: 処理FPS（Noneの場合は設定値を使用）
            show_progress: プログレスバーを表示するか（Noneの場合は設定値を使用）

        Returns:
            処理結果の統計情報

        Raises:
            VideoInputError: 動画ファイルが開けない場合
            TableDetectionError: 卓球台を検出できない場合
            ExportError: CSV出力に失敗した場合
        """
        # オプション引数のデフォルト値を設定から取得
        if target_fps is None:
            target_fps = self.config.video_processing.target_fps
        if show_progress is None:
            show_progress = self.config.video_processing.show_progress

        # 動画の初期化
        cap, video_writer, video_info = self._initialize_video_processing(
            input_video, output_video, target_fps
        )

        try:
            # 卓球台検出
            table_info = self._detect_table_with_validation(cap)

            # フレーム位置をリセット
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # フレーム処理
            results = self._process_frames(
                cap=cap,
                video_writer=video_writer,
                table_info=table_info,
                video_info=video_info,
                show_progress=show_progress
            )

            # CSV出力
            self._export_results(csv_output, results['player_ids'])

            print(f"\n出力ビデオ: {output_video} ({target_fps:.1f}fps)")

            return results

        finally:
            self._cleanup_resources(cap, video_writer)

    def _initialize_video_processing(
        self,
        input_video: str,
        output_video: str,
        target_fps: float
    ) -> Tuple[cv2.VideoCapture, cv2.VideoWriter, Dict[str, Any]]:
        """
        動画処理の初期化

        Args:
            input_video: 入力動画パス
            output_video: 出力動画パス
            target_fps: 目標FPS

        Returns:
            (VideoCapture, VideoWriter, video_info)のタプル

        Raises:
            VideoInputError: 動画ファイルが開けない場合
        """
        print(f"\n動画ファイルを開いています: {input_video}...")
        cap = cv2.VideoCapture(input_video)

        if not cap.isOpened():
            raise VideoInputError(input_video, "ファイルが存在しないか、形式が不正です")

        # 動画情報を取得
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_step = max(1, round(video_fps / target_fps))

        video_info = {
            'width': width,
            'height': height,
            'video_fps': video_fps,
            'total_frames': total_frames,
            'frame_step': frame_step,
            'target_fps': target_fps
        }

        # 動画情報を表示
        self._print_video_info(video_info)

        # VideoWriterの初期化
        fourcc = cv2.VideoWriter_fourcc(*self.VIDEO_CODEC_FOURCC)
        video_writer = cv2.VideoWriter(output_video, fourcc, target_fps, (width, height))

        print(f"CSV出力パス: {output_video}\n")

        return cap, video_writer, video_info

    def _print_video_info(self, video_info: Dict[str, Any]):
        """動画情報を表示"""
        print(f"\n入力情報:")
        print(f"  解像度: {video_info['width']}x{video_info['height']}")
        print(f"  動画FPS: {video_info['video_fps']:.2f}")
        print(f"  総フレーム数: {video_info['total_frames']}")
        print(f"  処理FPS: {video_info['target_fps']:.2f}")
        print(f"  出力動画FPS: {video_info['target_fps']:.2f} (処理したフレームのみ出力)\n")

        print(f"プレイヤー分類設定:")
        print(f"  最大プレイヤー数: {self.config.player_classification.max_players}")
        print(f"  最小スコア閾値: {self.config.player_classification.min_player_score:.2f}\n")

    def _detect_table_with_validation(
        self,
        cap: cv2.VideoCapture
    ) -> TableInfo:
        """
        卓球台検出と検証

        Args:
            cap: ビデオキャプチャオブジェクト

        Returns:
            検出された卓球台情報

        Raises:
            TableDetectionError: 卓球台を検出できない場合
        """
        table_info = self._detect_table(
            cap,
            max_attempts=self.config.table_detection.max_detection_attempts,
            min_confidence=self.config.table_detection.min_confidence
        )

        if table_info is None:
            raise TableDetectionError(
                f"卓球台を{self.config.table_detection.max_detection_attempts}回の試行で検出できませんでした",
                attempts=self.config.table_detection.max_detection_attempts,
                min_confidence=self.config.table_detection.min_confidence
            )

        return table_info

    def _detect_table(
        self,
        cap: cv2.VideoCapture,
        max_attempts: int,
        min_confidence: float,
    ) -> Optional[TableInfo]:
        """
        動画の異なる位置からサンプリングして検出を試行

        Args:
            cap: ビデオキャプチャオブジェクト
            max_attempts: 最大試行回数
            min_confidence: 検出に必要な最小信頼度閾値

        Returns:
            TableInfo: 検出された卓球台情報、失敗時はNone
        """
        print(f"\n卓球台を検出中（最大{max_attempts}回試行、信頼度閾値: {min_confidence:.2f}）...")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            print("警告: 動画の総フレーム数を取得できませんでした")
            total_frames = self.DEFAULT_FALLBACK_TOTAL_FRAMES

        # ex1 sample_step=1: 1フレームごとに処理
        # ex2 sample_step=total_frames//max_attempts=10 : 動画全体を均等に10分割して処理
        sample_step = max(1, total_frames // max_attempts)
        print(f"  総フレーム数: {total_frames}")
        print(f"  サンプリング間隔: {sample_step}フレーム\n")

        for attempt in range(max_attempts):
            frame_pos = min(attempt * sample_step, total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)

            ret, frame = cap.read()
            if not ret:
                print(f"警告: フレーム読み込み失敗（試行 {attempt + 1}/{max_attempts}, フレーム位置: {frame_pos}）")
                continue

            actual_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            table_info = self.table_detector.detect_table_frame(
                frame=frame,
                frame_idx=actual_frame_idx,
                force_detect=True
            )

            if table_info is not None and table_info.confidence >= min_confidence:
                self._print_table_detection_success(
                    table_info, attempt + 1, max_attempts,
                    actual_frame_idx, total_frames
                )
                return table_info
            elif table_info is not None:
                print(f"  信頼度不足: {table_info.confidence:.3f} < {min_confidence:.2f} "
                      f"(試行 {attempt + 1}/{max_attempts}, フレーム: {actual_frame_idx})")

        print(f"✗ {max_attempts}回の試行で十分な信頼度の卓球台を検出できませんでした")
        return None

    def _print_table_detection_success(
        self,
        table_info: TableInfo,
        attempt: int,
        max_attempts: int,
        frame_idx: int,
        total_frames: int
    ):
        """卓球台検出成功時の情報を表示"""
        print(f"✓ 卓球台を検出しました（試行 {attempt}/{max_attempts}）")
        print(f"  フレーム位置: {frame_idx}/{total_frames} ({frame_idx/total_frames*100:.1f}%)")
        print(f"  信頼度: {table_info.confidence:.3f}")
        print(f"  座標: {table_info.bbox}")

    def _process_frames(
        self,
        cap: cv2.VideoCapture,
        video_writer: cv2.VideoWriter,
        table_info: Optional[TableInfo],
        video_info: Dict[str, Any],
        show_progress: bool
    ) -> Dict[str, Any]:
        """
        全フレームを処理

        Args:
            cap: ビデオキャプチャ
            video_writer: ビデオライター
            table_info: 卓球台情報
            video_info: 動画情報
            show_progress: プログレスバー表示

        Returns:
            処理結果の統計情報
        """
        frame_count = 0
        processed_count = 0
        player_ids: Set[int] = set()

        self._print_processing_info(video_info)

        pbar = tqdm(
            total=video_info['total_frames'],
            desc="Processing",
            disable=not show_progress
        )

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                pbar.update(1)

                # フレームスキップ判定
                if frame_count % video_info['frame_step'] != 0:
                    continue

                processed_count += 1

                # フレーム処理
                player_ids = self._process_single_frame(
                    frame=frame,
                    frame_count=frame_count,
                    table_info=table_info,
                    video_fps=video_info['video_fps'],
                    player_ids=player_ids
                )

                # 結果描画と保存
                display_frame = self._create_display_frame(
                    frame=frame,
                    table_info=table_info,
                    frame_count=frame_count,
                    processed_count=processed_count,
                    total_frames=video_info['total_frames'],
                    height=video_info['height'],
                    player_ids=player_ids
                )
                video_writer.write(display_frame)

        finally:
            pbar.close()

        # 結果を表示
        self._print_results(frame_count, processed_count, player_ids)

        return {
            'total_frames': frame_count,
            'processed_frames': processed_count,
            'player_ids': sorted(player_ids),
            'candidates_count': len(self.player_classifier.candidates)
        }

    def _print_processing_info(self, video_info: Dict[str, Any]):
        """処理情報を表示"""
        print("処理開始...\n")
        print(f"  フレームステップ: {video_info['frame_step']} "
              f"({video_info['frame_step']}フレームごとに1回処理)")
        print(f"  予測処理フレーム数: 約{video_info['total_frames'] // video_info['frame_step']}フレーム")
        print(f"  処理率: {100.0 / video_info['frame_step']:.1f}%\n")

    def _process_single_frame(
        self,
        frame,
        frame_count: int,
        table_info: Optional[TableInfo],
        video_fps: float,
        player_ids: Set[int]
    ) -> Set[int]:
        """
        単一フレームを処理

        Args:
            frame: フレーム画像
            frame_count: フレーム番号
            table_info: 卓球台情報
            video_fps: 動画のFPS
            player_ids: 現在のプレイヤーIDセット

        Returns:
            更新されたプレイヤーIDセット
        """
        # 姿勢検出とトラッキング
        if table_info:
            persons = self.pose_tracker.track_frame_with_table_filter(frame, table_info)
        else:
            persons = self.pose_tracker.track_frame(frame)

        # プレイヤー分類器を更新
        if table_info and persons:
            self.player_classifier.update(persons, table_info, frame_count)

        # プレイヤーを分類
        if table_info:
            selected_ids, removed_ids = self.player_classifier.classify_players()
            player_ids = set(selected_ids)

            if removed_ids:
                self.pose_tracker.remove_validated_track_ids(removed_ids)

        # トラッキングデータをエクスポート用に保存
        if player_ids:
            player_persons = [p for p in persons if p.track_id in player_ids]
            if player_persons:
                timestamp = frame_count / video_fps
                self.tracking_exporter.add_frame(frame_count, timestamp, player_persons)

        return player_ids

    def _create_display_frame(
        self,
        frame,
        table_info: Optional[TableInfo],
        frame_count: int,
        processed_count: int,
        total_frames: int,
        height: int,
        player_ids: Set[int]
    ):
        """
        表示用フレームを作成

        Args:
            frame: 元のフレーム
            table_info: 卓球台情報
            frame_count: フレーム番号
            processed_count: 処理済みフレーム数
            total_frames: 総フレーム数
            height: フレームの高さ
            player_ids: プレイヤーIDセット

        Returns:
            描画済みフレーム
        """
        # 姿勢検出結果を取得（再度検出せず、最後の結果を使用）
        if table_info:
            persons = self.pose_tracker.track_frame_with_table_filter(frame, table_info)
        else:
            persons = self.pose_tracker.track_frame(frame)

        # 結果を描画
        display_frame = self.visualizer.draw_results(
            frame, table_info, persons, player_ids
        )
        display_frame = self.visualizer.draw_candidate_info(
            display_frame, player_ids
        )

        # フレーム情報を表示
        cv2.putText(
            display_frame,
            f"Frame: {frame_count}/{total_frames} (Processed: {processed_count})",
            (10, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display_frame,
            f"Detected: {len(persons)} persons, Players: {len(player_ids)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        return display_frame

    def _print_results(self, frame_count: int, processed_count: int, player_ids: Set[int]):
        """処理結果を表示"""
        print(f"\n✓ 処理完了:")
        print(f"  処理フレーム数: {frame_count}")
        print(f"  実際に処理したフレーム数: {processed_count}")
        print(f"  検出されたプレイヤーID: {sorted(player_ids)}")
        print(f"  候補者数: {len(self.player_classifier.candidates)}")

        # 候補者詳細情報
        if self.player_classifier.candidates:
            self._print_candidate_details(player_ids)

    def _print_candidate_details(self, player_ids: Set[int]):
        """候補者詳細情報を表示"""
        print(f"\n=== 候補者詳細 ===")
        candidates = []
        for track_id, candidate in self.player_classifier.candidates.items():
            if candidate.total_frames >= self.player_classifier.min_tracking_frames:
                score = self.player_classifier._calculate_player_score(candidate)
                candidates.append((track_id, candidate, score))

        candidates.sort(key=lambda x: x[2], reverse=True)

        for track_id, candidate, score in candidates:
            is_player = track_id in player_ids
            print(f"\nID {track_id} {'[PLAYER]' if is_player else ''}:")
            print(f"  スコア: {score:.3f}")
            print(f"  フレーム数: {candidate.total_frames}")
            print(f"  総運動量: {candidate.total_movement:.1f}")
            print(f"  卓球台付近比率: {candidate.near_table_ratio:.1%}")

    def _export_results(self, csv_output: str, player_ids: Set[int]):
        """
        結果をCSVにエクスポート

        Args:
            csv_output: CSV出力パス
            player_ids: プレイヤーIDセット

        Raises:
            ExportError: エクスポートに失敗した場合
        """
        try:
            # 連続性フィルタリングを適用
            self.tracking_exporter.filter_by_consecutive_frames()

            # プレイヤーの役割情報を作成
            player_roles = {track_id: "player" for track_id in player_ids}
            self.tracking_exporter.export_csv(csv_output, player_roles)

            print(f"\nプレイヤー骨格データをCSVに保存しました: {csv_output}")

        except Exception as e:
            raise ExportError(csv_output, str(e))

    def _cleanup_resources(
        self,
        cap: cv2.VideoCapture,
        video_writer: cv2.VideoWriter
    ):
        """リソースのクリーンアップ"""
        cap.release()
        video_writer.release()
