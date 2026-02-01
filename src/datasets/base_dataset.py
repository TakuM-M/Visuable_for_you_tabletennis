"""
基底データセットクラスと共通機能

全てのデータセットクラスが継承する基底クラスを定義
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Optional
from abc import ABC, abstractmethod
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.core import KEYPOINT_NAMES


class BasePoseSequenceDataset(Dataset, ABC):
    """
    骨格シーケンスデータセットの基底クラス

    共通の機能:
    - シーケンス切り出し
    - 統計情報の取得
    """

    def __init__(
        self,
        sequence_length: int = 30,
        stride: int = 1,
        keypoint_features: Optional[List[str]] = None
    ):
        """
        骨格シーケンスデータセットの基底クラスを初期化

        Args:
            sequence_length: シーケンス長（フレーム数）
            stride: シーケンス抽出時のストライド
            keypoint_features: 使用するキーポイント名のリスト（Noneの場合は全て使用）

        Note:
            入力データは既に正規化されていることを想定
            正規化はデータセット作成前にデータエクスポーターで処理される必要がある
        """
        super().__init__()
        self.sequence_length = sequence_length
        self.stride = stride

        # 使用するキーポイントを決定
        if keypoint_features is None:
            self.keypoint_names = KEYPOINT_NAMES
        else:
            self.keypoint_names = keypoint_features

        # データストレージ（サブクラスで設定）
        self.features: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
        self.frames: Optional[np.ndarray] = None

        # シーケンスインデックス
        self.sequence_indices: List[Tuple[int, int]] = []

    @abstractmethod
    def _load_data(self):
        """
        データを読み込む（サブクラスで実装）

        self.features, self.labels, self.framesを設定する必要がある
        """
        pass

    def _create_sequence_indices(self) -> List[Tuple[int, int]]:
        """
        シーケンスの開始・終了インデックスのリストを作成

        Returns:
            [(start_idx, end_idx), ...] のリスト
        """
        if self.features is None:
            raise ValueError("featuresが設定されていません。先に_load_dataを呼び出してください。")

        indices = []
        max_start = len(self.features) - self.sequence_length

        for start_idx in range(0, max_start + 1, self.stride):
            end_idx = start_idx + self.sequence_length
            indices.append((start_idx, end_idx))

        return indices


    def __len__(self) -> int:
        """データセットのサイズ"""
        return len(self.sequence_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        データの取得

        Args:
            idx: インデックス

        Returns:
            features: 特徴量テンソル (sequence_length, num_features)
            labels: ラベルテンソル (sequence_length,)
            metadata: メタデータ（フレーム番号など）
        """
        if idx < 0 or idx >= len(self.sequence_indices):
            raise IndexError(f"インデックス {idx} が範囲外です [0, {len(self.sequence_indices)})")

        start_idx, end_idx = self.sequence_indices[idx]

        # 特徴量とラベルを切り出し
        features = self.features[start_idx:end_idx]
        labels = self.labels[start_idx:end_idx]
        frames = self.frames[start_idx:end_idx]

        # テンソルに変換
        features_tensor = torch.from_numpy(features)
        labels_tensor = torch.from_numpy(labels).float()

        # メタデータ
        metadata = {
            'start_frame': int(frames[0]),
            'end_frame': int(frames[-1]),
            'start_idx': start_idx,
            'end_idx': end_idx
        }

        return features_tensor, labels_tensor, metadata

    def get_statistics(self) -> Dict:
        """データセットの統計情報を取得"""
        if self.features is None or self.labels is None:
            raise ValueError("データがまだ読み込まれていません。")

        return {
            'total_frames': len(self.features),
            'total_sequences': len(self.sequence_indices),
            'sequence_length': self.sequence_length,
            'stride': self.stride,
            'num_features': self.features.shape[1],
            'num_keypoints': len(self.keypoint_names),
            'play_frames': int(np.sum(self.labels == 1)),
            'non_play_frames': int(np.sum(self.labels == 0)),
            'play_ratio': float(np.mean(self.labels == 1))
        }


def collate_fn(batch):
    """
    カスタムcollate関数（バッチ作成用）

    可変長シーケンスに対応する場合に使用
    """
    features, labels, metadata = zip(*batch)

    features_batch = torch.stack(features)
    labels_batch = torch.stack(labels)

    return features_batch, labels_batch, metadata
