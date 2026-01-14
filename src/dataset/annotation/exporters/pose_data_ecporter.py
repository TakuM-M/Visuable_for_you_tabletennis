import cv2
import numpy as np
import sys
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import csv

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.detection.yolo_tracker import YOLOPoseTracker, PersonTrack, KEYPOINT_NAMES
from src.dataset.annotation.processors.pose_normalizer import PoseNormalizer

class PoseDataExporter:
    """姿勢データをCSVに出力するクラス"""

    def __init__(self, use_normalization: bool = True):
        """
        初期化

        Args:
            use_normalization: 正規化された座標を使用するか（デフォルト: True）
        """
        self.pose_data = []
        self.use_normalization = use_normalization
        self.normalizer = PoseNormalizer() if use_normalization else None

    def add_frame_data(self, frame_num: int, timestamp: float, person: PersonTrack):
        """
        フレームの姿勢データを追加

        Args:
            frame_num: フレーム番号
            timestamp: タイムスタンプ（秒）
            person: トラッキングされた人物
        """
        if self.use_normalization and self.normalizer:
            # 正規化されたデータを使用
            normalized = self.normalizer.normalize(person, frame_num, timestamp)
            frame_data = normalized.to_dict()
        else:
            # 元の絶対座標を使用（従来の方式）
            frame_data = {
                'frame': frame_num,
                'timestamp': timestamp,
                'track_id': person.track_id,
                'bbox_x1': person.bbox[0],
                'bbox_y1': person.bbox[1],
                'bbox_x2': person.bbox[2],
                'bbox_y2': person.bbox[3],
                'confidence': person.confidence
            }

            # キーポイントデータを追加
            for i, keypoint_name in enumerate(KEYPOINT_NAMES):
                kp = person.keypoints[i]
                frame_data[f'{keypoint_name}_x'] = kp[0]
                frame_data[f'{keypoint_name}_y'] = kp[1]
                frame_data[f'{keypoint_name}_conf'] = kp[2]

        self.pose_data.append(frame_data)

    def export_csv(self, output_path: str):
        """
        姿勢データをCSVファイルに出力

        Args:
            output_path: 出力ファイルパス
        """
        if not self.pose_data:
            print("警告: 出力するデータがありません")
            return

        # CSVのヘッダーを作成
        if self.use_normalization:
            # 正規化データのフィールド名
            fieldnames = PoseNormalizer.get_csv_fieldnames()
        else:
            # 従来のフィールド名
            fieldnames = ['frame', 'timestamp', 'track_id', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'confidence']
            for keypoint_name in KEYPOINT_NAMES:
                fieldnames.extend([
                    f'{keypoint_name}_x',
                    f'{keypoint_name}_y',
                    f'{keypoint_name}_conf'
                ])

        # CSVに書き込み
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.pose_data)

        data_type = "正規化された姿勢データ" if self.use_normalization else "姿勢データ"
        print(f"{data_type}をCSVに保存しました: {output_path}")
        print(f"  総フレーム数: {len(self.pose_data)}")

        if self.use_normalization:
            valid_count = sum(1 for d in self.pose_data if d.get('is_valid', False))
            print(f"  正規化成功: {valid_count}/{len(self.pose_data)} フレーム ({valid_count/len(self.pose_data)*100:.1f}%)")

    def get_statistics(self):
        """統計情報を取得"""
        if not self.pose_data:
            return {}

        total_frames = len(self.pose_data)
        track_ids = set(data['track_id'] for data in self.pose_data)

        # 各キーポイントの検出率を計算
        keypoint_detection_rates = {}
        for keypoint_name in KEYPOINT_NAMES:
            detected_count = sum(
                1 for data in self.pose_data
                if data[f'{keypoint_name}_conf'] > 0.5
            )
            keypoint_detection_rates[keypoint_name] = detected_count / total_frames

        return {
            'total_frames': total_frames,
            'track_ids': list(track_ids),
            'keypoint_detection_rates': keypoint_detection_rates
        }