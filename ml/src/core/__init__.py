"""
コアモジュール

検出・トラッキング・データ処理で共通利用するデータクラス・定数・例外
"""
from .data_classes import (
    CameraAngle,
    TableInfo,
    PlayerCandidate,
    PersonTrack,
    KEYPOINT_NAMES
)
from .exceptions import (
    PipelineError,
    TableDetectionError,
    VideoProcessingError,
    VideoInputError,
    ExportError,
    DataInputError,
    AugmentationError,
    TrainingError,
)

__all__ = [
    'CameraAngle',
    'TableInfo',
    'PlayerCandidate',
    'PersonTrack',
    'KEYPOINT_NAMES',
    'PipelineError',
    'TableDetectionError',
    'VideoProcessingError',
    'VideoInputError',
    'ExportError',
    'DataInputError',
    'AugmentationError',
    'TrainingError',
]
