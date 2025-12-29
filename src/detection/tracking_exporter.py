"""
トラッキング結果CSV出力モジュール
フレームごとのキーポイント座標をCSV形式で保存する
"""
import csv
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

from .yolo_tracker import PersonTrack, KEYPOINT_NAMES


@dataclass
class FrameData:
    """フレームデータ"""
    frame_num: int
    timestamp: float
    persons: List[PersonTrack]


class TrackingExporter:
    """トラッキング結果のCSV出力クラス"""

    def __init__(self):
        """CSV出力器の初期化"""
        self.frame_data_list: List[FrameData] = []

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

    def export_separate_player_csvs(
        self,
        output_dir: str,
        player_roles: Dict[int, str],
        base_filename: str = "player"
    ):
        """
        手前選手と相手選手を別々のCSVファイルに出力

        Args:
            output_dir: 出力ディレクトリ
            player_roles: トラッキングIDごとの役割
            base_filename: ベースファイル名
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 手前選手のCSV
        near_output = output_path / f"{base_filename}_near.csv"
        self.export_player_csv(str(near_output), player_roles, roles_to_export=["near_player"])

        # 相手選手のCSV
        far_output = output_path / f"{base_filename}_far.csv"
        self.export_player_csv(str(far_output), player_roles, roles_to_export=["far_player"])

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
