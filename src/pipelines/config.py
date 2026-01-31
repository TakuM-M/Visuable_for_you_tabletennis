"""
パイプライン設定用のデータクラス
"""
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class TableDetectionConfig:
    """卓球台検出の設定"""
    model_path: str
    cache_valid_frames: int = 1000
    min_confidence: float = 0.6
    max_detection_attempts: int = 100

    def __post_init__(self):
        """バリデーション"""
        if self.cache_valid_frames < 0:
            raise ValueError("cache_valid_frames must be non-negative")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0.0 and 1.0")
        if self.max_detection_attempts < 1:
            raise ValueError("max_detection_attempts must be at least 1")


@dataclass
class PoseTrackingConfig:
    """姿勢推定・トラッキングの設定"""
    model_path: str
    device: str = 'cuda'
    min_keypoint_confidence: float = 0.3

    def __post_init__(self):
        """バリデーション"""
        if self.device not in ['cuda', 'cpu', 'mps']:
            raise ValueError(f"Unsupported device: {self.device}")
        if not 0.0 <= self.min_keypoint_confidence <= 1.0:
            raise ValueError("min_keypoint_confidence must be between 0.0 and 1.0")


@dataclass
class PlayerClassificationConfig:
    """プレイヤー分類の設定"""
    max_players: int = 4
    min_player_score: float = 0.3

    def __post_init__(self):
        """バリデーション"""
        if self.max_players < 1:
            raise ValueError("max_players must be at least 1")
        if not 0.0 <= self.min_player_score <= 1.0:
            raise ValueError("min_player_score must be between 0.0 and 1.0")


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
class PipelineConfig:
    """パイプライン全体の設定"""
    table_detection: TableDetectionConfig
    pose_tracking: PoseTrackingConfig
    player_classification: PlayerClassificationConfig
    tracking_export: TrackingExportConfig
    video_processing: VideoProcessingConfig = field(default_factory=VideoProcessingConfig)

    @classmethod
    def create_default(
        cls,
        table_model_path: str,
        pose_model_path: str,
        device: str = 'cuda'
    ) -> 'PipelineConfig':
        """
        デフォルト設定でPipelineConfigを作成

        Args:
            table_model_path: 卓球台検出モデルのパス
            pose_model_path: 姿勢推定モデルのパス
            device: 使用デバイス ('cuda', 'cpu', 'mps')

        Returns:
            デフォルト設定のPipelineConfig
        """
        return cls(
            table_detection=TableDetectionConfig(model_path=table_model_path),
            pose_tracking=PoseTrackingConfig(model_path=pose_model_path, device=device),
            player_classification=PlayerClassificationConfig(),
            tracking_export=TrackingExportConfig(),
            video_processing=VideoProcessingConfig()
        )


@dataclass
class AugmentationConfig:
    """データ拡張の設定"""

    # 左右反転
    horizontal_flip: bool = False
    horizontal_flip_prob: float = 0.5

    # ガウシアンノイズ
    add_noise: bool = False
    noise_std: float = 0.02

    # 回転
    rotation: bool = False
    rotation_range: float = 15.0

    # スケーリング
    scaling: bool = False
    scale_range: Tuple[float, float] = (0.9, 1.1)

    # 関節ドロップアウト
    keypoint_dropout: bool = False
    dropout_prob: float = 0.1

    # 時間的ジッター（系列データ用）
    temporal_jitter: bool = False
    jitter_std: float = 0.5

    # 時間スケーリング（系列データ用）
    temporal_scaling: bool = False
    temporal_scale_range: Tuple[float, float] = (0.8, 1.2)

    # ランダムシード
    random_seed: Optional[int] = None

    def __post_init__(self):
        """バリデーション"""
        if not 0.0 <= self.horizontal_flip_prob <= 1.0:
            raise ValueError("horizontal_flip_prob must be between 0.0 and 1.0")
        if self.noise_std < 0:
            raise ValueError("noise_std must be non-negative")
        if self.rotation_range < 0:
            raise ValueError("rotation_range must be non-negative")
        if len(self.scale_range) != 2 or self.scale_range[0] > self.scale_range[1]:
            raise ValueError("scale_range must be (min, max) with min <= max")
        if not 0.0 <= self.dropout_prob <= 1.0:
            raise ValueError("dropout_prob must be between 0.0 and 1.0")
        if self.jitter_std < 0:
            raise ValueError("jitter_std must be non-negative")
        if len(self.temporal_scale_range) != 2 or self.temporal_scale_range[0] > self.temporal_scale_range[1]:
            raise ValueError("temporal_scale_range must be (min, max) with min <= max")


@dataclass
class AugmentationPipelineConfig:
    """データ拡張パイプラインの設定"""

    # データ拡張設定
    augmentation: AugmentationConfig

    # 拡張実行設定
    augmentation_factor: int = 5  # 元データの何倍に増やすか
    preserve_original: bool = True  # 元データも出力に含めるか

    # シーケンス処理設定
    is_sequence: bool = False  # 時系列データとして処理するか
    sequence_length: Optional[int] = None  # シーケンスの長さ（Noneの場合は全データ）

    # 出力設定
    save_metadata: bool = True  # 拡張メタデータを保存するか
    show_progress: bool = True  # プログレスバーを表示するか

    def __post_init__(self):
        """バリデーション"""
        if self.augmentation_factor < 1:
            raise ValueError("augmentation_factor must be at least 1")
        if self.sequence_length is not None and self.sequence_length < 1:
            raise ValueError("sequence_length must be at least 1")

    @classmethod
    def create_default(
        cls,
        augmentation_factor: int = 5,
        random_seed: Optional[int] = None
    ) -> 'AugmentationPipelineConfig':
        """
        デフォルト設定でAugmentationPipelineConfigを作成

        Args:
            augmentation_factor: データを何倍に拡張するか
            random_seed: ランダムシード

        Returns:
            デフォルト設定のAugmentationPipelineConfig
        """
        augmentation = AugmentationConfig(
            horizontal_flip=True,
            horizontal_flip_prob=0.5,
            add_noise=True,
            noise_std=0.02,
            rotation=True,
            rotation_range=15.0,
            random_seed=random_seed
        )
        return cls(
            augmentation=augmentation,
            augmentation_factor=augmentation_factor
        )