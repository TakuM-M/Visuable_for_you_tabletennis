"""
プレーシーンデータセット

学習用のデータセットクラス
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json


class PlaySceneDataset(Dataset):
    """
    プレー中/プレー外を判定するための学習用データセット
    """

    def __init__(
        self,
        data_path: Path,
        sequence_length: int = 30,
        transform=None,
        train: bool = True
    ):
        """
        Args:
            data_path: データファイルのパス（npzまたはjson）
            sequence_length: 時系列データのシーケンス長（フレーム数）
            transform: データ変換処理
            train: 学習用データかどうか
        """
        self.data_path = Path(data_path)
        self.sequence_length = sequence_length
        self.transform = transform
        self.train = train

        self.pose_sequences: List[np.ndarray] = []
        self.labels: List[int] = []

        self._load_data()

    def _load_data(self) -> None:
        """
        データを読み込む
        """
        if self.data_path.suffix == '.npz':
            self._load_from_npz()
        elif self.data_path.suffix == '.json':
            self._load_from_json()
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")

    def _load_from_npz(self) -> None:
        """
        NPZファイルからデータを読み込む
        """
        data = np.load(self.data_path)
        self.pose_sequences = data['pose_sequences']
        self.labels = data['labels']

    def _load_from_json(self) -> None:
        """
        JSONファイルからデータを読み込む
        """
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # TODO: JSON形式からデータを読み込む処理を実装
        pass

    def __len__(self) -> int:
        """
        データセットのサイズを返す
        """
        return len(self.pose_sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        指定されたインデックスのデータを返す

        Args:
            idx: データのインデックス

        Returns:
            (pose_sequence, label) のタプル
        """
        pose_sequence = self.pose_sequences[idx]
        label = self.labels[idx]

        # numpy配列をTensorに変換
        pose_tensor = torch.FloatTensor(pose_sequence)
        label_tensor = torch.LongTensor([label])

        if self.transform:
            pose_tensor = self.transform(pose_tensor)

        return pose_tensor, label_tensor

    def get_statistics(self) -> Dict:
        """
        データセットの統計情報を取得

        Returns:
            統計情報の辞書
        """
        labels_array = np.array(self.labels)

        return {
            "total_samples": len(self.labels),
            "playing_samples": np.sum(labels_array == 1),
            "non_playing_samples": np.sum(labels_array == 0),
            "class_balance": np.bincount(labels_array) / len(labels_array),
            "sequence_length": self.sequence_length,
            "pose_shape": self.pose_sequences[0].shape if len(self.pose_sequences) > 0 else None
        }


def create_train_val_split(
    dataset: PlaySceneDataset,
    val_ratio: float = 0.2,
    random_seed: int = 42
) -> Tuple[Dataset, Dataset]:
    """
    データセットを訓練用と検証用に分割

    Args:
        dataset: 分割対象のデータセット
        val_ratio: 検証用データの割合
        random_seed: 乱数シード

    Returns:
        (train_dataset, val_dataset) のタプル
    """
    from torch.utils.data import random_split

    total_size = len(dataset)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size

    generator = torch.Generator().manual_seed(random_seed)
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator
    )

    return train_dataset, val_dataset
