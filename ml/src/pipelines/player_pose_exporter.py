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

from src.pipelines.config import PlayerPoseExporterConfig
from src.pipelines.exceptions import (
    TableDetectionError,
    VideoInputError,
    VideoProcessingError,
    ExportError
)


class PlayerPoseExporter:
    """プレイヤーの姿勢を検出してCSVにエクスポートするパイプライン"""
    
    DEFAULT_FALLBACK_TOTAL_FRAMES = 10000
    VIDEO_CODEC_FOURCC = 'mp4v'

    def __init__(self, config: PlayerPoseExporterConfig):
        """
        Args:
            config: パイプライン設定
        """
        self.config = config
        
        self.table_detector = TableDetector(
            yolo_model_path=config.table_detection.model_path,
            cache_valid_frames=config.table_detection.cache_valid_frames,
            device=config.table_detection.device,
        )
        self.pose_tracker = YOLOPose_Tracker(
            model_path=config.pose_tracking.model_path,
            conf_threshold=config.pose_tracking.conf_threshold,
            iou_threshold=config.pose_tracking.iou_threshold,
            table_distance_threshold=config.pose_tracking.table_distance_threshold,
            device=config.pose_tracking.device,
            imgsz=config.pose_tracking.imgsz,
            half=config.pose_tracking.half,
        )
        self.player_classifier = PlayerClassifier(
            near_table_threshold=config.player_classification.near_table_threshold,
            min_tracking_frames=config.player_classification.min_tracking_frames,
            max_players=config.player_classification.max_players,
            max_inactive_frames=config.player_classification.max_inactive_frames,
            min_player_score=config.player_classification.min_player_score,
            recent_frames_window=config.player_classification.recent_frames_window,
            max_consecutive_other_count=config.player_classification.max_consecutive_other_count,
            movement_noise_threshold=config.player_classification.movement_noise_threshold,
        )
        self.tracking_exporter = TrackingExporter(
            min_consecutive_frames=config.tracking_export.min_consecutive_frames,
            max_frame_gap=config.tracking_export.max_frame_gap,
            min_confidence=config.pose_tracking.min_keypoint_confidence
        )

        self.visualizer = None
        if config.save_output:
            self.visualizer = PlayerClassifierVisualizer(
                self.table_detector,
                self.pose_tracker,
                self.player_classifier
            )

    def process_video(
        self,
        input_video: str,
        output_video: str,
        csv_output: str,
    ) -> Dict[str, Any]:
        """動画を処理してプレイヤーの姿勢を検出し、結果をCSVにエクスポートする"""
        
        target_fps = self.config.video_processing.target_fps
        show_progress = self.config.video_processing.show_progress

        cap, video_writer, video_info = self._initialize_video_processing(
            input_video,
            output_video,
            csv_output,
            target_fps
        )

        try:
            table_info = self._detect_table_with_validation(cap)

            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            results = self._process_frames(
                cap=cap,
                video_writer=video_writer,
                table_info=table_info,
                video_info=video_info,
                show_progress=show_progress
            )
            
            # 連続性フィルタリングを適用および正規化
            self.tracking_exporter.filter_by_consecutive_frames()
            self.tracking_exporter.normalize_poses()

            # CSV出力
            if self.visualizer is not None and video_writer is not None:
                self._export_results(csv_output, results['player_ids'])

            print(f"\n出力ビデオ: {output_video} ({target_fps:.1f}fps)")

            # frame_intervalとvideo_fpsを結果に追加
            results['frame_interval'] = video_info['frame_step']
            results['video_fps'] = video_info['video_fps']

            return results

        finally:
            self._cleanup_resources(cap, video_writer)

    def _initialize_video_processing(
        self,
        input_video: str,
        output_video: str,
        csv_output: str,
        target_fps: float
    ) -> Tuple[cv2.VideoCapture, cv2.VideoWriter, Dict[str, Any]]:
        """初期化処理: 動画のオープン、情報取得、VideoWriterの初期化"""
        
        print(f"\n動画ファイルを開いています: {input_video}...")
        cap = cv2.VideoCapture(input_video)

        if not cap.isOpened():
            raise VideoInputError(input_video, "ファイルが存在しないか、形式が不正です")

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

        self._print_video_info(video_info)
        
        video_writer = None
        if self.visualizer is not None:
            output_dir = Path(output_video).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*self.VIDEO_CODEC_FOURCC)
            video_writer = cv2.VideoWriter(output_video, fourcc, target_fps, (width, height))
            if not video_writer.isOpened():
                raise VideoProcessingError(f"VideoWriterの初期化に失敗しました: {output_video}")
            print(f"出力動画パス: {output_video}")
            print(f"CSV出力パス: {csv_output}\n")

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
        """卓球台検出と検証"""
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
                if not cap.grab():
                    break

                frame_count += 1
                pbar.update(1)

                if frame_count % video_info['frame_step'] != 0:
                    continue

                ret, frame = cap.retrieve()
                if not ret:
                    continue

                processed_count += 1

                player_ids = self._process_single_frame(
                    frame=frame,
                    frame_count=frame_count,
                    table_info=table_info,
                    video_fps=video_info['video_fps'],
                    player_ids=player_ids
                )

                if video_writer is not None and self.visualizer is not None:
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
        persons = self.pose_tracker.track_frame_with_table_filter(frame, table_info)

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
            # プレイヤーの役割情報を作成
            player_roles = {track_id: "player" for track_id in player_ids}
            self.tracking_exporter.export_csv(csv_output, player_roles)

            print(f"\nプレイヤー骨格データをCSVに保存しました: {csv_output}")

        except Exception as e:
            raise ExportError(csv_output, str(e))

    def _cleanup_resources(
        self,
        cap: cv2.VideoCapture,
        video_writer: Optional[cv2.VideoWriter]
    ):
        """リソースのクリーンアップ"""
        cap.release()
        if video_writer is not None:
            video_writer.release()
