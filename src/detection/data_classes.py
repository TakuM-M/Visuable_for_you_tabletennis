"""
detection パッケージのデータクラス定義

このモジュールには以下のデータクラスが含まれます:
- CameraAngle: カメラアングルの種類を表すEnum
- TableInfo: 卓球台の検出情報
- PlayerCandidate: プレイヤー候補の情報
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np


class CameraAngle(Enum):
    """カメラアングルの種類"""
    ENDLINE = "endline"          # エンドライン側（選手の後ろ）
    SIDELINE = "sideline"        # サイドライン側（横）
    DIAGONAL_TOP = "diagonal_top"  # 斜め上（プロ試合）
    UNKNOWN = "unknown"


@dataclass
class TableInfo:
    """卓球台情報 - バウンディングボックスと基本情報のみ"""
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    frame_idx: int
    camera_angle: CameraAngle = CameraAngle.UNKNOWN  # 将来の拡張用に保持

    @property
    def center(self) -> Tuple[float, float]:
        """バウンディングボックスの中心座標"""
        return (
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2
        )

    @property
    def width(self) -> float:
        """バウンディングボックスの幅"""
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        """バウンディングボックスの高さ"""
        return self.bbox[3] - self.bbox[1]

    @property
    def aspect_ratio(self) -> float:
        """アスペクト比（幅/高さ）"""
        return self.width / self.height if self.height > 0 else 0

    def estimate_camera_angle(self, frame_shape: Tuple[int, int, int]) -> CameraAngle:
        """
        バウンディングボックスの特徴から画角を推定（将来の拡張用）

        判定基準:
        - アスペクト比: 横長 → サイドライン、縦長 → エンドライン
        - 画面占有率と位置: 斜め上の判定

        Args:
            frame_shape: (height, width, channels)

        Returns:
            推定されたカメラアングル
        """
        frame_height, frame_width = frame_shape[:2]

        # 画面占有率
        area_ratio = (self.width * self.height) / (frame_width * frame_height)

        # 画面内の縦方向位置（0.0=上端、1.0=下端）
        vertical_position = self.center[1] / frame_height

        # アスペクト比による簡易判定
        if self.aspect_ratio > 2.5:
            return CameraAngle.SIDELINE
        elif self.aspect_ratio < 1.2:
            return CameraAngle.ENDLINE
        elif vertical_position < 0.4 and area_ratio < 0.15:
            return CameraAngle.DIAGONAL_TOP
        elif 1.2 <= self.aspect_ratio <= 2.5:
            return CameraAngle.ENDLINE

        return CameraAngle.UNKNOWN


@dataclass
class PlayerCandidate:
    """
    プレイヤー候補の情報

    PlayerDetectorが内部的に使用するデータクラス
    """
    track_id: int
    first_seen_frame: int
    last_seen_frame: int
    positions: List[str]  # ["near", "near", "far", ...]
    keypoints_history: List[np.ndarray]  # 骨格データ履歴
    total_movement: float
    near_table_count: int
    total_frames: int

    @property
    def tracking_duration_frames(self) -> int:
        """tracking継続フレーム数"""
        return self.last_seen_frame - self.first_seen_frame

    def tracking_duration(self, fps: float) -> float:
        """
        tracking継続時間（秒）

        Args:
            fps: 処理時のフレームレート

        Returns:
            継続時間（秒）
        """
        return self.tracking_duration_frames / fps

    @property
    def near_table_ratio(self) -> float:
        """卓球台付近にいた時間の比率"""
        return self.near_table_count / self.total_frames if self.total_frames > 0 else 0.0
