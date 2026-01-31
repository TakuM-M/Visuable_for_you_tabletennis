"""
複数CSVファイル対応のデータセット

複数の動画から抽出したCSVファイルを統合して1つのデータセットとして扱う
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Tuple, Optional, Dict

from src.datasets.dataset import PoseSequenceDataset


class MultiCSVPoseDataset(Dataset):
    """
    複数のCSVファイルを統合したデータセット

    各動画のCSVファイルをPoseSequenceDatasetとして読み込み、
    内部的に結合して1つのデータセットとして扱う
    """

    def __init__(
        self,
        csv_label_pairs: List[Tuple[str, str]],
        sequence_length: int = 30,
        stride: int = 5,
        normalize_features: bool = True
    ):
        """
        初期化

        Args:
            csv_label_pairs: [(csv_path, label_path), ...] のリスト
            sequence_length: シーケンス長
            stride: ストライド
            normalize_features: 特徴量を正規化するか
        """
        self.csv_label_pairs = csv_label_pairs
        self.sequence_length = sequence_length
        self.stride = stride
        self.normalize_features = normalize_features

        # 各CSVファイルのデータセットを作成
        self.datasets: List[PoseSequenceDataset] = []
        self.dataset_lengths: List[int] = []
        self.cumulative_lengths: List[int] = [0]

        print("複数CSVデータセット初期化中...")
        for i, (csv_path, label_path) in enumerate(csv_label_pairs):
            print(f"  [{i+1}/{len(csv_label_pairs)}] {Path(csv_path).name}")

            dataset = PoseSequenceDataset(
                csv_path=csv_path,
                label_path=label_path,
                sequence_length=sequence_length,
                stride=stride,
                normalize_features=normalize_features
            )

            self.datasets.append(dataset)
            self.dataset_lengths.append(len(dataset))
            self.cumulative_lengths.append(
                self.cumulative_lengths[-1] + len(dataset)
            )

            print(f"    → {len(dataset)} シーケンス")

        self.total_length = self.cumulative_lengths[-1]

        print(f"\n統合完了: 合計 {self.total_length} シーケンス")
        print(f"  動画数: {len(self.datasets)}")
        print(f"  シーケンス長: {sequence_length}")
        print(f"  ストライド: {stride}")

    def __len__(self) -> int:
        """データセットの長さを返す"""
        return self.total_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        指定されたインデックスのデータを返す

        Args:
            idx: データのインデックス

        Returns:
            (features, labels, metadata) のタプル
        """
        if idx < 0 or idx >= self.total_length:
            raise IndexError(f"Index {idx} out of range [0, {self.total_length})")

        # どのデータセットに属するかを特定
        dataset_idx = 0
        for i in range(len(self.cumulative_lengths) - 1):
            if self.cumulative_lengths[i] <= idx < self.cumulative_lengths[i + 1]:
                dataset_idx = i
                break

        # そのデータセット内でのインデックス
        local_idx = idx - self.cumulative_lengths[dataset_idx]

        # データを取得
        features, labels, metadata = self.datasets[dataset_idx][local_idx]

        # メタデータに動画情報を追加
        metadata['dataset_idx'] = dataset_idx
        metadata['csv_path'] = str(self.csv_label_pairs[dataset_idx][0])

        return features, labels, metadata

    def get_statistics(self) -> Dict:
        """データセットの統計情報を取得"""
        stats = {
            'total_sequences': self.total_length,
            'num_videos': len(self.datasets),
            'sequence_length': self.sequence_length,
            'stride': self.stride,
            'videos': []
        }

        for i, (csv_path, label_path) in enumerate(self.csv_label_pairs):
            video_stats = {
                'csv_path': str(csv_path),
                'label_path': str(label_path),
                'num_sequences': self.dataset_lengths[i]
            }
            stats['videos'].append(video_stats)

        return stats

    @classmethod
    def from_directories(
        cls,
        data_dirs: List[str],
        csv_filename: str = 'original_pose_data.csv',
        label_filename: str = 'play_labels.csv',
        sequence_length: int = 30,
        stride: int = 5,
        normalize_features: bool = True
    ) -> 'MultiCSVPoseDataset':
        """
        ディレクトリリストから複数CSVデータセットを作成

        Args:
            data_dirs: データディレクトリのリスト
            csv_filename: CSVファイル名
            label_filename: ラベルファイル名
            sequence_length: シーケンス長
            stride: ストライド
            normalize_features: 正規化するか

        Returns:
            MultiCSVPoseDataset
        """
        csv_label_pairs = []

        for data_dir in data_dirs:
            csv_path = Path(data_dir) / csv_filename
            label_path = Path(data_dir) / label_filename

            if not csv_path.exists():
                print(f"警告: CSVファイルが見つかりません: {csv_path}")
                continue

            if not label_path.exists():
                print(f"警告: ラベルファイルが見つかりません: {label_path}")
                continue

            csv_label_pairs.append((str(csv_path), str(label_path)))

        if not csv_label_pairs:
            raise ValueError("有効なCSVファイルが見つかりませんでした")

        return cls(
            csv_label_pairs=csv_label_pairs,
            sequence_length=sequence_length,
            stride=stride,
            normalize_features=normalize_features
        )
