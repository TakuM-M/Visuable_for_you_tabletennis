"""
パイプライン設定用のデータクラス
"""
from dataclasses import dataclass, field
from typing import Optional, Tuple
# =====================================================
# プレイヤー姿勢エクスポーター設定
# =====================================================
@dataclass
class TableDetectionConfig:
    """卓球台検出の設定"""
    model_path: str
    cache_valid_frames: int = 1000
    min_confidence: float = 0.6
    max_detection_attempts: int = 100
    device: str = 'cuda'

    def __post_init__(self):
        """バリデーション"""
        if self.cache_valid_frames < 0:
            raise ValueError("cache_valid_frames must be non-negative")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0.0 and 1.0")
        if self.max_detection_attempts < 1:
            raise ValueError("max_detection_attempts must be at least 1")
        if self.device not in ['cuda', 'cpu', 'mps']:
            raise ValueError(f"Unsupported device: {self.device}")


@dataclass
class PoseTrackingConfig:
    """姿勢推定・トラッキングの設定"""
    model_path: str
    conf_threshold: float = 0.5
    iou_threshold: float = 0.7
    table_distance_threshold: float = 0.2
    min_keypoint_confidence: float = 0.3
    device: str = 'cuda'
    imgsz: int = 640
    half: bool = False

    def __post_init__(self):
        """バリデーション"""
        if not 0.0 <= self.conf_threshold <= 1.0:
            raise ValueError("conf_threshold must be between 0.0 and 1.0")
        if not 0.0 <= self.iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be between 0.0 and 1.0")
        if self.table_distance_threshold < 0.0:
            raise ValueError("table_distance_threshold must be non-negative")
        if not 0.0 <= self.min_keypoint_confidence <= 1.0:
            raise ValueError("min_keypoint_confidence must be between 0.0 and 1.0")
        if self.device not in ['cuda', 'cpu', 'mps']:
            raise ValueError(f"Unsupported device: {self.device}")
        if self.imgsz < 32:
            raise ValueError("imgsz must be at least 32")


@dataclass
class PlayerClassificationConfig:
    """プレイヤー分類の設定"""
    near_table_threshold: float = 0.1
    min_tracking_frames: int = 10
    max_players: int = 2
    max_inactive_frames: int = 30
    min_player_score: float = 0.3
    recent_frames_window: int = 146
    max_consecutive_other_count: int = 30
    movement_noise_threshold: float = 5.0

    def __post_init__(self):
        """バリデーション"""
        if self.near_table_threshold < 0.0:
            raise ValueError("near_table_threshold must be non-negative")
        if self.min_tracking_frames < 1:
            raise ValueError("min_tracking_frames must be at least 1")
        if self.max_players < 1:
            raise ValueError("max_players must be at least 1")
        if self.max_inactive_frames < 1:
            raise ValueError("max_inactive_frames must be at least 1")
        if not 0.0 <= self.min_player_score <= 1.0:
            raise ValueError("min_player_score must be between 0.0 and 1.0")
        if self.recent_frames_window < 1:
            raise ValueError("recent_frames_window must be at least 1")
        if self.max_consecutive_other_count < 1:
            raise ValueError("max_consecutive_other_count must be at least 1")
        if self.movement_noise_threshold < 0.0:
            raise ValueError("movement_noise_threshold must be non-negative")


@dataclass
class TrackingExportConfig:
    """トラッキングエクスポートの設定"""
    min_consecutive_frames: int = 30
    max_frame_gap: int = 5

    def __post_init__(self):
        """バリデーション"""
        if self.min_consecutive_frames < 1:
            raise ValueError("min_consecutive_frames must be at least 1")
        if self.max_frame_gap < 0:
            raise ValueError("max_frame_gap must be non-negative")


@dataclass
class VideoProcessingConfig:
    """動画処理の設定"""
    target_fps: float = 30.0
    show_progress: bool = True
    output_codec: str = 'mp4v'

    def __post_init__(self):
        """バリデーション"""
        if self.target_fps <= 0:
            raise ValueError("target_fps must be positive")


@dataclass
class PlayerPoseExporterConfig:
    """プレイヤー姿勢エクスポーターの設定"""
    table_detection: TableDetectionConfig
    pose_tracking: PoseTrackingConfig
    player_classification: PlayerClassificationConfig
    tracking_export: TrackingExportConfig
    video_processing: VideoProcessingConfig = field(default_factory=VideoProcessingConfig)
    save_intermediate_files: bool = True

# =====================================================
# プレーシーン検出
# =====================================================
@dataclass
class PlaySceneDetectionConfig:
    """プレーシーン検出の設定"""
    model_path: str
    config_path: Optional[str] = None
    device: str = 'cuda'
    threshold: float = 0.5
    min_scene_duration: int = 10
    batch_size: int = 64
    smoothing_window: int = 5  # メディアンフィルタのウィンドウサイズ（0で無効化）

    def __post_init__(self):
        """バリデーション"""
        if self.device not in ['cuda', 'cpu', 'mps']:
            raise ValueError(f"Unsupported device: {self.device}")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")
        if self.min_scene_duration < 1:
            raise ValueError("min_scene_duration must be at least 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.smoothing_window < 0:
            raise ValueError("smoothing_window must be non-negative")

# =====================================================
# 動画作成設定
# =====================================================

@dataclass
class VideoCompositionConfig:
    """動画作成の設定"""
    output_codec: str = 'mp4v'
    output_fps: Optional[float] = None
    add_scene_info: bool = True
    show_progress: bool = True
    max_scenes: Optional[int] = None
    min_scene_duration_for_highlights: Optional[int] = None
    scene_buffer_before_sec: float = 0.0
    scene_buffer_after_sec: float = 0.0

    def __post_init__(self):
        """バリデーション"""
        if self.output_fps is not None and self.output_fps <= 0:
            raise ValueError("output_fps must be positive")
        if self.max_scenes is not None and self.max_scenes < 1:
            raise ValueError("max_scenes must be at least 1")
        if self.min_scene_duration_for_highlights is not None and self.min_scene_duration_for_highlights < 1:
            raise ValueError("min_scene_duration_for_highlights must be at least 1")
        if self.scene_buffer_before_sec < 0.0:
            raise ValueError("scene_buffer_before_sec must be non-negative")
        if self.scene_buffer_after_sec < 0.0:
            raise ValueError("scene_buffer_after_sec must be non-negative")
        
# =====================================================
# 推論パイプライン全体の設定
# =====================================================
@dataclass
class InferencePipelineConfig:
    """推論パイプライン全体の設定"""
    pose_export: PlayerPoseExporterConfig
    scene_detection: PlaySceneDetectionConfig
    show_progress: bool = True
    save_intermediate_files: bool = True