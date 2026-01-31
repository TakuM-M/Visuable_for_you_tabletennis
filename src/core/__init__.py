"""
コアモジュール

検出・トラッキング・データ処理で共通利用するデータクラスと定数
"""
from .data_classes import (
    CameraAngle,
    TableInfo,
    PlayerCandidate,
    PersonTrack,
    KEYPOINT_NAMES
)

__all__ = [
    'CameraAngle',
    'TableInfo',
    'PlayerCandidate',
    'PersonTrack',
    'KEYPOINT_NAMES'
]
