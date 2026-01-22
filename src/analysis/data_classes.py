"""
analysis パッケージのデータクラス定義

このモジュールには以下のデータクラスが含まれます:
- PlayerTableRelation: プレイヤーと卓球台の位置関係
"""

from dataclasses import dataclass
from typing import Optional

from ..detection.data_classes import CameraAngle


@dataclass
class PlayerTableRelation:
    """プレイヤーと卓球台の位置関係"""
    track_id: int
    position: str  # "near", "far", "left", "right"（画角により異なる）
    side: Optional[str]  # "left" or "right"（エンドライン時は左右、サイドライン時はNone）
    distance_normalized: float  # 0.0-1.0以上（卓球台からの正規化距離）
    is_in_play_area: bool  # プレイエリア内にいるか
    camera_angle: CameraAngle  # カメラアングル
