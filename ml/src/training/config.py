from dataclasses import dataclass, field
from typing import Optional

# =====================================================
# モデル訓練パイプライン設定
# =====================================================
@dataclass
class ModelConfig:
    """モデルの設定"""
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.3

    def __post_init__(self):
        """バリデーション"""
        if self.hidden_size < 1:
            raise ValueError("hidden_size must be at least 1")
        if self.num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be between 0.0 and 1.0")


@dataclass
class DatasetConfig:
    """データセットの設定（複数CSV対応）"""
    # 動画データディレクトリのリスト
    train_data_dirs: list  # 例: ['data/video1', 'data/video2']
    val_data_dirs: Optional[list] = None

    # CSVファイル名
    csv_filename: str = 'original_pose_data.csv'
    label_filename: str = 'play_labels.csv'

    # データセット設定
    sequence_length: int = 30
    stride: int = 5
    batch_size: int = 32
    num_workers: int = 4
    use_motion_features: bool = False  # 速度・加速度特徴量を追加するか（34→102次元）

    # オンラインデータ拡張（訓練時のみ）
    use_augmentation: bool = False

    def __post_init__(self):
        """バリデーション"""
        if not self.train_data_dirs or len(self.train_data_dirs) == 0:
            raise ValueError("train_data_dirs must contain at least one directory")
        if self.sequence_length < 1:
            raise ValueError("sequence_length must be at least 1")
        if self.stride < 1:
            raise ValueError("stride must be at least 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")


@dataclass
class OptimizerConfig:
    """最適化器の設定"""
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    scheduler_min_lr: float = 1e-6

    def __post_init__(self):
        """バリデーション"""
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.scheduler_patience < 1:
            raise ValueError("scheduler_patience must be at least 1")
        if not 0.0 < self.scheduler_factor < 1.0:
            raise ValueError("scheduler_factor must be between 0.0 and 1.0")
        if self.scheduler_min_lr <= 0:
            raise ValueError("scheduler_min_lr must be positive")


@dataclass
class TrainingConfig:
    """学習の設定"""
    epochs: int = 50
    save_every: int = 10
    device: str = 'cuda'
    use_tensorboard: bool = True
    early_stopping_patience: Optional[int] = None

    def __post_init__(self):
        """バリデーション"""
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1")
        if self.save_every < 1:
            raise ValueError("save_every must be at least 1")
        if self.device not in ['cuda', 'cpu', 'mps']:
            raise ValueError(f"Unsupported device: {self.device}")
        if self.early_stopping_patience is not None and self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be at least 1")


@dataclass
class TrainingPipelineConfig:
    """学習パイプライン全体の設定"""
    model: ModelConfig
    dataset: DatasetConfig
    optimizer: OptimizerConfig
    training: TrainingConfig
    output_dir: str = 'output/training'

    @classmethod
    def create_default(
        cls,
        train_data_dirs: list,
        val_data_dirs: Optional[list] = None,
        output_dir: str = 'output/training',
        device: str = 'cuda'
    ) -> 'TrainingPipelineConfig':
        """
        デフォルト設定でTrainingPipelineConfigを作成

        Args:
            train_data_dirs: 訓練用動画データディレクトリのリスト
            val_data_dirs: 検証用動画データディレクトリのリスト
            output_dir: 出力ディレクトリ
            device: 使用デバイス

        Returns:
            デフォルト設定のTrainingPipelineConfig
        """
        return cls(
            model=ModelConfig(),
            dataset=DatasetConfig(
                train_data_dirs=train_data_dirs,
                val_data_dirs=val_data_dirs
            ),
            optimizer=OptimizerConfig(),
            training=TrainingConfig(device=device),
            output_dir=output_dir
        )
