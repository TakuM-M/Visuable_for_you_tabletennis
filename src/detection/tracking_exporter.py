"""
トラッキング結果出力モジュール
フレームごとのキーポイント保存する
"""
import csv
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

from src.core.data_classes import PersonTrack, KEYPOINT_NAMES


@dataclass
class FrameData:
    """フレームデータ"""
    frame_num: int
    timestamp: float
    persons: List[PersonTrack]


class TrackingExporter:
    """トラッキング結果の出力クラス"""

    def __init__(
        self,
        min_consecutive_frames: int = 30,
        max_frame_gap: int = 5,
        min_confidence: float = 0.3
    ):
        """
        出力器の初期化

        Args:
            min_consecutive_frames: 連続性フィルタで保持する最小フレーム数（デフォルト: 30）
            max_frame_gap: 連続とみなす最大フレーム間隔（デフォルト: 5）
            min_confidence: 正規化時のキーポイント最小信頼度（デフォルト: 0.3）
        """
        self.frame_data_list: List[FrameData] = []
        self.is_normalized: bool = False  # 正規化済みフラグ

        # 連続性フィルタリング設定
        self.min_consecutive_frames = min_consecutive_frames
        self.max_frame_gap = max_frame_gap

        # 正規化設定
        self.min_confidence = min_confidence

        # キーポイントのインデックス（COCO形式）
        self.LEFT_HIP_IDX = KEYPOINT_NAMES.index("left_hip")
        self.RIGHT_HIP_IDX = KEYPOINT_NAMES.index("right_hip")

    def add_frame(
        self,
        frame_num: int,
        timestamp: float,
        persons: List[PersonTrack]
    ):
        """
        フレームデータを追加

        Args:
            frame_num: フレーム番号
            timestamp: タイムスタンプ（秒）
            persons: トラッキング結果
        """
        self.frame_data_list.append(FrameData(
            frame_num=frame_num,
            timestamp=timestamp,
            persons=persons
        ))

    def filter_by_consecutive_frames(
        self,
        min_consecutive_frames: Optional[int] = None,
        max_frame_gap: Optional[int] = None
    ) -> None:
        """
        連続して出現している区間のみを保持し、断片的な出現を除外する

        各トラッキングIDについて:
        1. フレーム番号のリストを取得
        2. 連続区間を検出（max_frame_gap以下の欠けは許容）
        3. min_consecutive_frames未満の区間を削除

        Args:
            min_consecutive_frames: 保持する最小連続フレーム数（Noneの場合は初期化時の値を使用）
            max_frame_gap: 連続とみなす最大フレーム間隔（Noneの場合は初期化時の値を使用）
        """
        # Noneの場合はインスタンス変数の値を使用
        if min_consecutive_frames is None:
            min_consecutive_frames = self.min_consecutive_frames
        if max_frame_gap is None:
            max_frame_gap = self.max_frame_gap
        if not self.frame_data_list:
            return

        # トラッキングIDごとにフレーム番号をグループ化
        track_id_frames: Dict[int, List[int]] = {}
        for frame_data in self.frame_data_list:
            for person in frame_data.persons:
                track_id = person.track_id
                if track_id not in track_id_frames:
                    track_id_frames[track_id] = []
                track_id_frames[track_id].append(frame_data.frame_num)

        # 各トラッキングIDについて連続区間を検出
        valid_frame_sets: Dict[int, set] = {}
        for track_id, frame_nums in track_id_frames.items():
            frame_nums_sorted = sorted(set(frame_nums))

            # 連続区間を検出
            consecutive_segments = []
            current_segment = [frame_nums_sorted[0]]
            for i in range(1, len(frame_nums_sorted)):
                frame_gap = frame_nums_sorted[i] - frame_nums_sorted[i-1]

                if frame_gap <= max_frame_gap:
                    # ほぼ連続とみなす
                    current_segment.append(frame_nums_sorted[i])
                else:
                    # 区間が途切れた
                    consecutive_segments.append(current_segment)
                    current_segment = [frame_nums_sorted[i]]

            # 最後の区間を追加
            consecutive_segments.append(current_segment)

            # 最小連続フレーム数を満たす区間のみを保持
            valid_frames = set()
            for segment in consecutive_segments:
                if len(segment) >= min_consecutive_frames:
                    valid_frames.update(segment)

            if valid_frames:
                valid_frame_sets[track_id] = valid_frames

        # フィルタリング: 有効な区間のみを保持
        filtered_frame_data_list = []
        removed_count = 0

        for frame_data in self.frame_data_list:
            filtered_persons = []

            for person in frame_data.persons:
                track_id = person.track_id
                frame_num = frame_data.frame_num

                # このトラッキングIDのこのフレームが有効な区間に含まれるかチェック
                if track_id in valid_frame_sets and frame_num in valid_frame_sets[track_id]:
                    filtered_persons.append(person)
                else:
                    removed_count += 1

            # このフレームに有効な人物がいる場合のみ保持
            if filtered_persons:
                filtered_frame_data_list.append(FrameData(
                    frame_num=frame_data.frame_num,
                    timestamp=frame_data.timestamp,
                    persons=filtered_persons
                ))

        self.frame_data_list = filtered_frame_data_list

        print(f"\n連続性フィルタリング完了:")
        print(f"  最小連続フレーム数: {min_consecutive_frames}")
        print(f"  最大フレーム間隔: {max_frame_gap}")
        print(f"  削除されたデータ数: {removed_count}")
        print(f"  保持されたトラッキングID: {sorted(valid_frame_sets.keys())}")

    def normalize_poses(
        self,
        min_confidence: Optional[float] = None
    ) -> Dict[str, any]:
        """
        保持している全フレームの骨格データを正規化する（訓練データと同じ方法）

        正規化方法:
        1. 腰の中心（left_hip と right_hip の中点）を原点とする
        2. 腰幅（left_hip と right_hip の距離）でスケールを正規化
        3. 各キーポイントを相対座標に変換

        Args:
            min_confidence: キーポイントの最小信頼度（Noneの場合は初期化時の値を使用）

        Returns:
            正規化の統計情報（成功数、失敗数、スケールファクターの統計など）
        """
        # Noneの場合はインスタンス変数の値を使用
        if min_confidence is None:
            min_confidence = self.min_confidence
        if self.is_normalized:
            print("警告: 既に正規化済みです。再度正規化はスキップされます。")
            return {}

        if not self.frame_data_list:
            print("警告: 正規化するデータがありません。")
            return {}

        total_persons = 0
        valid_count = 0
        invalid_count = 0
        scale_factors = []

        print("\n骨格データを正規化中...")

        # 全フレーム・全人物を正規化
        for frame_data in self.frame_data_list:
            for person in frame_data.persons:
                total_persons += 1

                # 腰の中心と腰幅を計算
                hip_center, hip_width, is_valid = self._compute_hip_center_and_width(
                    person.keypoints, min_confidence
                )

                if not is_valid:
                    invalid_count += 1
                    # 正規化できない場合は元の座標をそのまま保持
                    continue

                valid_count += 1
                scale_factors.append(hip_width)

                # 各キーポイントを正規化（インプレース更新）
                for i in range(17):
                    kp_x, kp_y, conf = person.keypoints[i]

                    if conf >= min_confidence:
                        # 腰中心を原点とした相対座標
                        relative_x = kp_x - hip_center[0]
                        relative_y = kp_y - hip_center[1]

                        # 腰幅でスケール正規化
                        person.keypoints[i, 0] = relative_x / hip_width
                        person.keypoints[i, 1] = relative_y / hip_width
                    else:
                        # 信頼度が低い場合は0に設定
                        person.keypoints[i, 0] = 0.0
                        person.keypoints[i, 1] = 0.0

        self.is_normalized = True

        # 統計情報
        stats = {
            'total_persons': total_persons,
            'valid_count': valid_count,
            'invalid_count': invalid_count,
            'scale_factors': scale_factors
        }

        print(f"正規化完了:")
        print(f"  総データ数: {total_persons}")
        print(f"  正規化成功: {valid_count}")
        print(f"  正規化失敗: {invalid_count}")

        if scale_factors:
            print(f"  腰幅統計:")
            print(f"    平均: {np.mean(scale_factors):.2f}px")
            print(f"    標準偏差: {np.std(scale_factors):.2f}px")
            print(f"    範囲: {np.min(scale_factors):.2f} - {np.max(scale_factors):.2f}px")

        return stats

    def _compute_hip_center_and_width(
        self,
        keypoints: np.ndarray,
        min_confidence: float
    ) -> tuple:
        """
        腰の中心座標と腰幅を計算（訓練データと同じ方法）

        Args:
            keypoints: キーポイント配列 (17, 3) [x, y, confidence]
            min_confidence: 最小信頼度

        Returns:
            ((center_x, center_y), hip_width, is_valid)
        """
        left_hip = keypoints[self.LEFT_HIP_IDX]
        right_hip = keypoints[self.RIGHT_HIP_IDX]

        # 両方の腰が検出されているかチェック
        if (left_hip[2] >= min_confidence and
            right_hip[2] >= min_confidence):
            # 腰の中心を計算
            center_x = (left_hip[0] + right_hip[0]) / 2.0
            center_y = (left_hip[1] + right_hip[1]) / 2.0

            # 腰幅を計算（スケール係数）
            dx = right_hip[0] - left_hip[0]
            dy = right_hip[1] - left_hip[1]
            hip_width = np.sqrt(dx * dx + dy * dy)

            # 腰幅が極端に小さい場合は無効
            if hip_width > 0:
                return ((center_x, center_y), hip_width, True)

        # 両方検出されていない、または腰幅が0の場合は無効
        return ((0.0, 0.0), 1.0, False)

    def get_pose_data_for_dataset(
        self,
        track_id: Optional[int] = None
    ) -> tuple:
        """
        InMemoryPoseSequenceDatasetへの入力用データを取得

        Args:
            track_id: 特定のトラッキングIDのみ抽出（Noneの場合は全データ）

        Returns:
            (pose_data, frames) のタプル
            - pose_data: 正規化済み骨格データ (num_frames, 34) [17キーポイント × 2座標]
            - frames: フレーム番号配列 (num_frames,)
        """
        if not self.is_normalized:
            raise ValueError(
                "データが正規化されていません。先に normalize_poses() を実行してください。"
            )

        if not self.frame_data_list:
            return np.array([]).reshape(0, 34), np.array([])

        pose_list = []
        frame_list = []

        for frame_data in self.frame_data_list:
            for person in frame_data.persons:
                # track_idが指定されている場合はフィルタリング
                if track_id is not None and person.track_id != track_id:
                    continue

                # キーポイントのx, y座標のみを抽出 (17, 2) -> (34,)
                keypoints_xy = person.keypoints[:, :2].flatten()
                pose_list.append(keypoints_xy)
                frame_list.append(frame_data.frame_num)

        if not pose_list:
            return np.array([]).reshape(0, 34), np.array([])

        pose_data = np.array(pose_list, dtype=np.float32)
        frames = np.array(frame_list, dtype=np.int64)

        return pose_data, frames

    def get_all_track_ids(self) -> List[int]:
        """
        保持している全てのトラッキングIDを取得

        Returns:
            トラッキングIDのリスト（ソート済み）
        """
        track_ids = set()
        for frame_data in self.frame_data_list:
            for person in frame_data.persons:
                track_ids.add(person.track_id)
        return sorted(track_ids)

    def export_csv(
        self,
        output_path: str,
        player_roles: Optional[Dict[int, str]] = None
    ):
        """
        全人物のトラッキング結果をCSV出力

        Args:
            output_path: 出力CSVファイルパス
            player_roles: トラッキングIDごとの役割 (optional)
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # CSVヘッダーを作成
        header = ["track_id", "frame", "timestamp", "role", "confidence", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]

        # キーポイントのカラムを追加
        for kp_name in KEYPOINT_NAMES:
            header.extend([
                f"{kp_name}_x",
                f"{kp_name}_y",
                f"{kp_name}_conf"
            ])

        # CSV書き込み
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)

            for frame_data in self.frame_data_list:
                for person in frame_data.persons:
                    # 役割を取得
                    role = "unknown"
                    if player_roles and person.track_id in player_roles:
                        role = player_roles[person.track_id]

                    # 基本情報
                    row = [
                        person.track_id,
                        frame_data.frame_num,
                        f"{frame_data.timestamp:.3f}",
                        role,
                        f"{person.confidence:.3f}",
                        person.bbox[0],
                        person.bbox[1],
                        person.bbox[2],
                        person.bbox[3]
                    ]

                    # キーポイント座標を追加
                    for kp in person.keypoints:
                        row.extend([
                            f"{kp[0]:.2f}",
                            f"{kp[1]:.2f}",
                            f"{kp[2]:.3f}"
                        ])

                    writer.writerow(row)

        print(f"全トラッキングデータをCSVに保存しました: {output_file}")
        print(f"  総フレーム数: {len(self.frame_data_list)}")
        print(f"  総データ行数: {sum(len(fd.persons) for fd in self.frame_data_list)}")

    def export_player_csv(
        self,
        output_path: str,
        player_roles: Dict[int, str],
        roles_to_export: Optional[List[str]] = None
    ):
        """
        特定の役割の人物のみをCSV出力

        Args:
            output_path: 出力CSVファイルパス
            player_roles: トラッキングIDごとの役割
            roles_to_export: 出力する役割のリスト（None の場合は全て）
        """
        if roles_to_export is None:
            roles_to_export = ["near_player", "far_player"]

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # 出力対象のトラッキングIDを抽出
        target_track_ids = set()
        for track_id, role in player_roles.items():
            if role in roles_to_export:
                target_track_ids.add(track_id)

        # CSVヘッダーを作成
        header = ["track_id", "frame", "timestamp", "role", "confidence", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]

        # キーポイントのカラムを追加
        for kp_name in KEYPOINT_NAMES:
            header.extend([
                f"{kp_name}_x",
                f"{kp_name}_y",
                f"{kp_name}_conf"
            ])

        # CSV書き込み
        exported_count = 0
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)

            for frame_data in self.frame_data_list:
                for person in frame_data.persons:
                    # 出力対象のトラッキングIDかチェック
                    if person.track_id not in target_track_ids:
                        continue

                    # 役割を取得
                    role = player_roles[person.track_id]

                    # 基本情報
                    row = [
                        person.track_id,
                        frame_data.frame_num,
                        f"{frame_data.timestamp:.3f}",
                        role,
                        f"{person.confidence:.3f}",
                        person.bbox[0],
                        person.bbox[1],
                        person.bbox[2],
                        person.bbox[3]
                    ]

                    # キーポイント座標を追加
                    for kp in person.keypoints:
                        row.extend([
                            f"{kp[0]:.2f}",
                            f"{kp[1]:.2f}",
                            f"{kp[2]:.3f}"
                        ])

                    writer.writerow(row)
                    exported_count += 1

        print(f"選手トラッキングデータをCSVに保存しました: {output_file}")
        print(f"  出力対象役割: {roles_to_export}")
        print(f"  出力トラッキングID: {sorted(target_track_ids)}")
        print(f"  総データ行数: {exported_count}")

    def get_statistics(self) -> Dict:
        """
        エクスポートデータの統計情報を取得

        Returns:
            統計情報の辞書
        """
        if not self.frame_data_list:
            return {}

        total_frames = len(self.frame_data_list)
        total_detections = sum(len(fd.persons) for fd in self.frame_data_list)

        # トラッキングIDごとの出現回数
        track_id_counts = {}
        for frame_data in self.frame_data_list:
            for person in frame_data.persons:
                track_id = person.track_id
                track_id_counts[track_id] = track_id_counts.get(track_id, 0) + 1

        return {
            "total_frames": total_frames,
            "total_detections": total_detections,
            "avg_persons_per_frame": total_detections / total_frames if total_frames > 0 else 0,
            "unique_track_ids": len(track_id_counts),
            "track_id_counts": track_id_counts
        }

    def export_summary(self, output_path: str, player_roles: Optional[Dict[int, str]] = None):
        """
        トラッキング結果のサマリーをテキストファイルに出力

        Args:
            output_path: 出力ファイルパス
            player_roles: トラッキングIDごとの役割 (optional)
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        stats = self.get_statistics()

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=== トラッキング結果サマリー ===\n\n")
            f.write(f"総フレーム数: {stats.get('total_frames', 0)}\n")
            f.write(f"総検出数: {stats.get('total_detections', 0)}\n")
            f.write(f"平均人数/フレーム: {stats.get('avg_persons_per_frame', 0):.2f}\n")
            f.write(f"ユニークトラッキングID数: {stats.get('unique_track_ids', 0)}\n\n")

            f.write("=== トラッキングID別出現回数 ===\n")
            track_id_counts = stats.get('track_id_counts', {})

            for track_id, count in sorted(track_id_counts.items()):
                role = "unknown"
                if player_roles and track_id in player_roles:
                    role = player_roles[track_id]

                appearance_ratio = count / stats['total_frames'] * 100 if stats['total_frames'] > 0 else 0
                f.write(f"  ID {track_id:3d}: {count:5d} フレーム ({appearance_ratio:5.1f}%) - {role}\n")

            if player_roles:
                f.write("\n=== 役割別トラッキングID ===\n")
                near_ids = [tid for tid, role in player_roles.items() if role == "near_player"]
                far_ids = [tid for tid, role in player_roles.items() if role == "far_player"]
                other_ids = [tid for tid, role in player_roles.items() if role == "other"]

                f.write(f"  手前選手: {near_ids}\n")
                f.write(f"  相手選手: {far_ids}\n")
                f.write(f"  その他: {other_ids}\n")

        print(f"サマリーを保存しました: {output_file}")