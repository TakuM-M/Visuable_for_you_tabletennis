"""
CSVベースの骨格シーケンスデータセット

正規化されたCSVファイルから骨格シーケンスデータを読み込む
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List

from src.datasets.base_dataset import BasePoseSequenceDataset


class CSVPoseSequenceDataset(BasePoseSequenceDataset):
    """
    CSVベースの骨格シーケンスデータセット

    正規化された骨格データCSVファイルから固定長シーケンスを作成
    """

    def __init__(
        self,
        csv_path: str,
        label_path: Optional[str] = None,
        sequence_length: int = 30,
        stride: int = 1,
        keypoint_features: Optional[List[str]] = None,
        use_motion_features: bool = False,
        augmentor=None
    ):
        """
        CSV骨格シーケンスデータセットを初期化

        Args:
            csv_path: 正規化された骨格データCSVのパス
            label_path: ラベルファイルのパス（CSV形式: frame,label）
                       Noneの場合、全フレームを非プレイ(0)としてラベル付け
            sequence_length: シーケンス長（フレーム数）
            stride: シーケンス抽出時のストライド
            keypoint_features: 使用するキーポイント名のリスト（Noneの場合は全て使用）
            use_motion_features: 速度・加速度特徴量を追加するか（34→102次元）
            augmentor: オンラインデータ拡張（訓練時のみ使用）

        Note:
            CSVデータは既に正規化されていることを想定
        """
        self.csv_path = Path(csv_path)
        self.label_path = Path(label_path) if label_path else None

        # 基底クラスを初期化
        super().__init__(
            sequence_length=sequence_length,
            stride=stride,
            keypoint_features=keypoint_features,
            use_motion_features=use_motion_features,
            augmentor=augmentor
        )

        # データを読み込み
        self._load_data()

        # 速度・加速度特徴量を追加
        if self.use_motion_features:
            self._compute_motion_features()

        # シーケンスインデックスを作成
        self.sequence_indices = self._create_sequence_indices()

        print(f"CSVPoseSequenceDataset initialized:")
        print(f"  CSV path: {self.csv_path}")
        print(f"  Total frames: {len(self.features)}")
        print(f"  Total sequences: {len(self.sequence_indices)}")
        print(f"  Sequence length: {self.sequence_length}")
        print(f"  Feature dimensions: {self.features.shape[1]}")
        print(f"  Play frames: {np.sum(self.labels == 1)} / {len(self.labels)}")

    def _load_data(self):
        """CSVファイルからデータを読み込む"""
        # データを読み込み
        data_df = pd.read_csv(self.csv_path)

        # ラベルを読み込み
        if self.label_path and self.label_path.exists():
            labels_df = pd.read_csv(self.label_path)
            # フレーム番号でマージ
            data_df = data_df.merge(
                labels_df,
                on='frame',
                how='left'
            )
            data_df['label'] = data_df['label'].fillna(0).astype(int)
        else:
            # ラベルがない場合は全て0（非プレイ）としてマーク
            data_df['label'] = 0

        # 特徴量カラムを作成
        feature_columns = []
        for kp_name in self.keypoint_names:
            feature_columns.extend([
                f'{kp_name}_norm_x',
                f'{kp_name}_norm_y'
            ])

        # 特徴量を抽出
        self.features = data_df[feature_columns].values.astype(np.float32)
        self.labels = data_df['label'].values.astype(np.int64)
        self.frames = data_df['frame'].values.astype(np.int64)