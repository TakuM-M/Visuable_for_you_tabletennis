"""
プレイヤー姿勢検出・エクスポートパイプライン
"""
import cv2
from pathlib import Path
from typing import Optional, Dict, Any, Set
from tqdm import tqdm

from src.detection.table_detector import TableDetector
from src.detection.yolopose_tracker import YOLOPose_Tracker
from src.detection.player_classifier import PlayerClassifier
from src.detection.tracking_exporter import TrackingExporter
from src.visualization.player_classifier_visualizer import PlayerClassifierVisualizer


class PlayerPoseExporter:
    """プレイヤーの姿勢を検出してCSVにエクスポートするパイプライン"""

    def __init__(
        self,
        table_model_path: str,
        pose_model_path: str,
        device: str = 'cuda',
        table_cache_valid_frames: int = 1000,
        max_players: int = 4,
        min_player_score: float = 0.3,
        min_consecutive_frames: int = 30,
        max_frame_gap: int = 5,
        min_keypoint_confidence: float = 0.3
    ):
        """
        Args:
            table_model_path: 卓球台検出モデルのパス
            pose_model_path: 姿勢推定モデルのパス
            device: 使用デバイス ('cuda' or 'cpu')
            table_cache_valid_frames: 卓球台検出のキャッシュ有効フレーム数
            max_players: 最大プレイヤー数
            min_player_score: プレイヤー判定の最小スコア閾値
            min_consecutive_frames: トラッキング連続性フィルタの最小フレーム数
            max_frame_gap: トラッキング連続性フィルタの最大フレーム間隔
            min_keypoint_confidence: 骨格正規化時のキーポイント最小信頼度
        """
        self.table_model_path = table_model_path
        self.pose_model_path = pose_model_path
        self.device = device
        self.max_players = max_players
        self.min_player_score = min_player_score
        self.table_cache_valid_frames = table_cache_valid_frames

        # TrackingExporter用の設定
        self.min_consecutive_frames = min_consecutive_frames
        self.max_frame_gap = max_frame_gap
        self.min_keypoint_confidence = min_keypoint_confidence

        # コンポーネントの初期化
        self.table_detector = TableDetector(
            yolo_model_path=table_model_path,
            cache_valid_frames=table_cache_valid_frames
        )
        self.pose_tracker = YOLOPose_Tracker(
            model_path=pose_model_path,
            device=device
        )
        self.player_classifier = PlayerClassifier(
            max_players=max_players,
            min_player_score=min_player_score
        )
        self.tracking_exporter = TrackingExporter(
            min_consecutive_frames=min_consecutive_frames,
            max_frame_gap=max_frame_gap,
            min_confidence=min_keypoint_confidence
        )
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
        target_fps: float = 30.0,
        max_detection_attempts: int = 100,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        動画からプレイヤーの姿勢を検出してCSVに出力

        Args:
            input_video: 入力動画パス
            output_video: 出力動画パス
            csv_output: CSV出力パス
            target_fps: 処理FPS
            max_detection_attempts: 卓球台検出の最大試行回数
            show_progress: プログレスバーを表示するか

        Returns:
            処理結果の統計情報
        """
        print(f"\n動画ファイルを開いています: {input_video}...")
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            raise RuntimeError("エラー: 動画ファイルを開けませんでした")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"\n入力情報:")
        print(f"  解像度: {width}x{height}")
        print(f"  動画FPS: {video_fps:.2f}")
        print(f"  総フレーム数: {total_frames}")
        print(f"  処理FPS: {target_fps:.2f}")
        print(f"  出力動画FPS: {target_fps:.2f} (処理したフレームのみ出力)\n")

        # ex1: video_fps=60, target_fps=30 -> frame_step=2 (2フレームに1回処理)
        # ex2: video_fps=30, target_fps=30 -> frame_step=1 (1フレームに1回処理)
        frame_step = max(1, round(video_fps / target_fps))

        print(f"プレイヤー分類設定:")
        print(f"  最大プレイヤー数: {self.max_players}")
        print(f"  最小スコア閾値: {self.min_player_score:.2f}\n")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video, fourcc, target_fps, (width, height))

        print(f"CSV出力: {csv_output}\n")

        try:
            table_info = self._detect_table(cap, max_detection_attempts, min_confidence=0.6)

            if table_info is None:
                raise RuntimeError("エラー: 卓球台を検出できませんでした")

            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            results = self._process_frames(
                cap=cap,
                video_writer=video_writer,
                table_info=table_info,
                width=width,
                height=height,
                video_fps=video_fps,
                total_frames=total_frames,
                frame_step=frame_step,
                show_progress=show_progress
            )

            self._export_csv(
                csv_output=csv_output,
                player_ids=results['player_ids']
            )

            print(f"\n出力ビデオ: {output_video} ({target_fps:.1f}fps)")

            return results

        finally:
            cap.release()
            video_writer.release()

    def _detect_table(
        self,
        cap: cv2.VideoCapture,
        max_attempts: int,
        min_confidence: float = 0.6,
    ):
        """
        動画の異なる位置からサンプリングして検出を試行
        sanple_step=総フレーム数/最大試行回数

        Args:
            cap: ビデオキャプチャオブジェクト
            max_attempts: 最大試行回数
            min_confidence: 検出に必要な最小信頼度閾値（デフォルト: 0.7）

        Returns:
            TableInfo: 検出された卓球台情報、失敗時はNone
        """
        print(f"\n卓球台を検出中（最大{max_attempts}回試行、信頼度閾値: {min_confidence:.2f}）...")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            print("警告: 動画の総フレーム数を取得できませんでした")
            total_frames = 10000  # フォールバック値

        # 動画全体から均等にサンプリング位置を計算
        # ex: max_attempts=100, total_frames=3000 → 30フレームごとにサンプリング
        sample_step = max(1, total_frames // max_attempts)
        print(f"  総フレーム数: {total_frames}")
        print(f"  サンプリング間隔: {sample_step}フレーム\n")

        for attempt in range(max_attempts):
            # サンプリング位置を計算（動画全体から均等に分散）
            frame_pos = attempt * sample_step
            # 動画の範囲を超えないようにクリップ
            frame_pos = min(frame_pos, total_frames - 1)

            # 指定位置にシーク
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)

            ret, frame = cap.read()
            if not ret:
                print(f"警告: フレーム読み込み失敗（試行 {attempt + 1}/{max_attempts}, フレーム位置: {frame_pos}）")
                continue

            # 実際のフレーム位置を取得（シークが正確でない場合がある）
            actual_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            # 卓球台を検出（force_detect=Trueでキャッシュを無視）
            table_info = self.table_detector.detect_table_frame(
                frame=frame,
                frame_idx=actual_frame_idx,
                force_detect=True
            )

            if table_info is not None:
                if table_info.confidence >= min_confidence:
                    print(f"✓ 卓球台を検出しました（試行 {attempt + 1}/{max_attempts}）")
                    print(f"  フレーム位置: {actual_frame_idx}/{total_frames} ({actual_frame_idx/total_frames*100:.1f}%)")
                    print(f"  信頼度: {table_info.confidence:.3f}")
                    print(f"  座標: {table_info.bbox}")
                    return table_info
                else:
                    print(f"  信頼度不足: {table_info.confidence:.3f} < {min_confidence:.2f} (試行 {attempt + 1}/{max_attempts}, フレーム: {actual_frame_idx})")

        print(f"✗ {max_attempts}回の試行で十分な信頼度の卓球台を検出できませんでした")
        return None

    def _process_frames(
        self,
        cap: cv2.VideoCapture,
        video_writer: cv2.VideoWriter,
        table_info,
        width: int,
        height: int,
        video_fps: float,
        total_frames: int,
        frame_step: int,
        show_progress: bool
    ) -> Dict[str, Any]:
        """全フレームを処理"""
        frame_count = 0
        processed_count = 0
        player_ids: Set[int] = set()

        print("処理開始...\n")
        print(f"  フレームステップ: {frame_step} ({frame_step}フレームごとに1回処理)")
        print(f"  予測処理フレーム数: 約{total_frames // frame_step}フレーム")
        print(f"  処理率: {100.0 / frame_step:.1f}%\n")

        # プログレスバー
        pbar = tqdm(total=total_frames, desc="Processing", disable=not show_progress)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                pbar.update(1)

                if frame_count % frame_step != 0:
                    continue
                
                processed_count += 1
                
                # 卓球台の位置情報からフィルタリングした人物のみを返す
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

                if player_ids:
                    player_persons = [p for p in persons if p.track_id in player_ids]
                    if player_persons:
                        timestamp = frame_count / video_fps
                        self.tracking_exporter.add_frame(frame_count, timestamp, player_persons)

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

                # ビデオに保存
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

    def _print_results(self, frame_count: int, processed_count: int, player_ids: Set[int]):
        """処理結果を表示"""
        print(f"\n✓ 処理完了:")
        print(f"  処理フレーム数: {frame_count}")
        print(f"  実際に処理したフレーム数: {processed_count}")
        print(f"  検出されたプレイヤーID: {sorted(player_ids)}")
        print(f"  候補者数: {len(self.player_classifier.candidates)}")

        # 候補者詳細情報
        if self.player_classifier.candidates:
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

    def _export_csv(
        self,
        csv_output: str,
        player_ids: Set[int]
    ):
        """CSV出力"""
        # 連続性フィルタリングを適用（初期化時の設定を使用）
        self.tracking_exporter.filter_by_consecutive_frames()

        # プレイヤーの役割情報を作成
        player_roles = {track_id: "player" for track_id in player_ids}
        self.tracking_exporter.export_csv(csv_output, player_roles)
        print(f"\nプレイヤー骨格データをCSVに保存しました: {csv_output}")
