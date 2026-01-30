"""
detection パッケージ

YOLOを用いた物体検出・追跡機能を提供
"""

# proto_type_02 用のデータクラス
from ..core.data_classes import CameraAngle, TableInfo, PlayerCandidate, PersonTrack

__all__ = [
    'CameraAngle',
    'TableInfo',
    'PlayerCandidate',
    'PersonTrack',
]
