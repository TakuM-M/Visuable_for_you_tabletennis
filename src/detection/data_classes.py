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
    """卓球台情報 - バウンディングボックスのみで奥行き推定"""
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    frame_idx: int
    camera_angle: CameraAngle = CameraAngle.UNKNOWN

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
        バウンディングボックスの特徴から画角を推定

        判定基準:
        - アスペクト比: 横長 → サイドライン、縦長 → エンドライン
        - 画面占有率: 大きい → 近い（エンドライン or サイドライン）
        - 位置: 画面上部 → 斜め上

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

        # アスペクト比による判定
        # 実際の卓球台: 長さ274cm × 幅152.5cm ≒ 1.8

        if self.aspect_ratio > 2.5:
            # 非常に横長 → サイドライン側から
            return CameraAngle.SIDELINE

        elif self.aspect_ratio < 1.2:
            # 縦長 → エンドライン側から
            return CameraAngle.ENDLINE

        elif vertical_position < 0.4 and area_ratio < 0.15:
            # 画面上部 + 小さい → 斜め上から
            return CameraAngle.DIAGONAL_TOP

        elif 1.2 <= self.aspect_ratio <= 2.5:
            # 中間 → エンドライン側（やや横）
            return CameraAngle.ENDLINE

        return CameraAngle.UNKNOWN

    def get_near_far_boundary(self) -> float:
        """
        画角に応じた前後判定の境界y座標

        Returns:
            境界y座標（これより下が手前、上が奥）
        """
        if self.camera_angle == CameraAngle.SIDELINE:
            # サイドライン: 前後の概念が弱い → 卓球台中央を境界
            return self.center[1]

        elif self.camera_angle == CameraAngle.ENDLINE:
            # エンドライン: 奥行きがある
            # バウンディングボックスの上から1/3あたりが境界
            # （遠近法で奥側が上部に圧縮される）
            return self.bbox[1] + self.height * 0.35

        elif self.camera_angle == CameraAngle.DIAGONAL_TOP:
            # 斜め上: 奥行き感が強い
            # バウンディングボックスの上から1/4あたりが境界
            return self.bbox[1] + self.height * 0.25

        else:
            # 不明な場合はデフォルト
            return self.center[1]

    def get_near_area(self, margin_ratio: float = 0.4) -> dict:
        """
        手前側エリア（画角適応）

        Args:
            margin_ratio: 卓球台サイズに対する余白比率

        Returns:
            エリアの座標 {'x1': float, 'y1': float, 'x2': float, 'y2': float}
        """
        boundary_y = self.get_near_far_boundary()

        if self.camera_angle == CameraAngle.SIDELINE:
            # サイドライン: 左右に分かれる
            margin = self.width * margin_ratio
            return {
                'x1': self.bbox[0] - margin,
                'y1': self.bbox[1] - margin,
                'x2': self.center[0],  # 中央まで
                'y2': self.bbox[3] + margin
            }

        else:
            # エンドライン / 斜め上: 下側が手前
            margin_x = self.width * margin_ratio
            margin_y = self.height * margin_ratio

            return {
                'x1': self.bbox[0] - margin_x,
                'y1': boundary_y,
                'x2': self.bbox[2] + margin_x,
                'y2': self.bbox[3] + margin_y * 1.5  # 手前側は広めに
            }

    def get_far_area(self, margin_ratio: float = 0.4) -> dict:
        """
        奥側エリア（画角適応）

        Args:
            margin_ratio: 卓球台サイズに対する余白比率

        Returns:
            エリアの座標 {'x1': float, 'y1': float, 'x2': float, 'y2': float}
        """
        boundary_y = self.get_near_far_boundary()

        if self.camera_angle == CameraAngle.SIDELINE:
            # サイドライン: 右側が奥
            margin = self.width * margin_ratio
            return {
                'x1': self.center[0],  # 中央から
                'y1': self.bbox[1] - margin,
                'x2': self.bbox[2] + margin,
                'y2': self.bbox[3] + margin
            }

        else:
            # エンドライン / 斜め上: 上側が奥
            margin_x = self.width * margin_ratio
            margin_y = self.height * margin_ratio

            return {
                'x1': self.bbox[0] - margin_x,
                'y1': self.bbox[1] - margin_y * 0.8,  # 奥側は控えめに
                'x2': self.bbox[2] + margin_x,
                'y2': boundary_y
            }


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
