"""
複数CSV骨格シーケンスデータセット

複数のCSVファイルを単一のデータセットに結合
複数動画の骨格データを扱う
"""
from pathlib import Path
from typing import List, Tuple, Dict
from torch.utils.data import Dataset

from src.datasets.csv_dataset import CSVPoseSequenceDataset


class MultiCSVPoseDataset(Dataset):
    """
    複数CSV骨格シーケンスデータセット

    複数のCSVファイルを個別のPoseSequenceDatasetとして読み込み
    単一のデータセットに結合
    """

    def __init__(
        self,
        csv_label_pairs: List[Tuple[str, str]],
        sequence_length: int = 30,
        stride: int = 5
    ):
        """
        複数CSVデータセットを初期化

        Args:
            csv_label_pairs: (csv_path, label_path)タプルのリスト
            sequence_length: シーケンス長
            stride: シーケンス抽出時のストライド

        Note:
            CSVデータは既に正規化されていることを想定
        """
        self.csv_label_pairs = csv_label_pairs
        self.sequence_length = sequence_length
        self.stride = stride

        # 各CSV用の個別データセット
        self.datasets: List[CSVPoseSequenceDataset] = []
        self.dataset_lengths: List[int] = []
        self.cumulative_lengths: List[int] = [0]

        print(f"Initializing MultiCSVPoseDataset with {len(csv_label_pairs)} videos...")

        # 全データセットを読み込み
        for i, (csv_path, label_path) in enumerate(csv_label_pairs):
            print(f"  [{i+1}/{len(csv_label_pairs)}] {Path(csv_path).name}")

            dataset = CSVPoseSequenceDataset(
                csv_path=csv_path,
                label_path=label_path,
                sequence_length=sequence_length,
                stride=stride
            )

            self.datasets.append(dataset)
            self.dataset_lengths.append(len(dataset))
            self.cumulative_lengths.append(
                self.cumulative_lengths[-1] + len(dataset)
            )

            print(f"    -> {len(dataset)} sequences")

        self.total_length = self.cumulative_lengths[-1]

        print(f"\nMultiCSVPoseDataset initialized:")
        print(f"  Total videos: {len(self.datasets)}")
        print(f"  Total sequences: {self.total_length}")
        print(f"  Sequence length: {sequence_length}")
        print(f"  Stride: {stride}")


    def __len__(self) -> int:
        """データセットの長さを返す"""
        return self.total_length

    def __getitem__(self, idx: int) -> Tuple:
        """
        指定されたインデックスのデータを取得

        Args:
            idx: データインデックス

        Returns:
            (features, labels, metadata) タプル
        """
        if idx < 0 or idx >= self.total_length:
            raise IndexError(f"インデックス {idx} が範囲外です [0, {self.total_length})")

        # このインデックスがどのデータセットに属するか検索
        dataset_idx = 0
        for i in range(len(self.cumulative_lengths) - 1):
            if self.cumulative_lengths[i] <= idx < self.cumulative_lengths[i + 1]:
                dataset_idx = i
                break

        # そのデータセット内でのローカルインデックスを取得
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
            video_stats = self.datasets[i].get_statistics()
            video_stats['csv_path'] = str(csv_path)
            video_stats['label_path'] = str(label_path)
            stats['videos'].append(video_stats)

        return stats

    @classmethod
    def from_directories(
        cls,
        data_dirs: List[str],
        csv_filename: str = 'normalized_pose_data.csv',
        label_filename: str = 'play_labels.csv',
        sequence_length: int = 30,
        stride: int = 5
    ) -> 'MultiCSVPoseDataset':
        """
        ディレクトリリストからMultiCSVPoseDatasetを作成

        Args:
            data_dirs: データディレクトリのリスト
            csv_filename: CSVファイル名
            label_filename: ラベルファイル名
            sequence_length: シーケンス長
            stride: ストライド

        Returns:
            MultiCSVPoseDatasetインスタンス
        """
        csv_label_pairs = []

        for data_dir in data_dirs:
            csv_path = Path(data_dir) / csv_filename
            label_path = Path(data_dir) / label_filename

            if not csv_path.exists():
                print(f"Warning: CSV file not found: {csv_path}")
                continue

            if not label_path.exists():
                print(f"Warning: Label file not found: {label_path}")
                continue

            csv_label_pairs.append((str(csv_path), str(label_path)))

        if not csv_label_pairs:
            raise ValueError("有効なCSVファイルが見つかりません")

        return cls(
            csv_label_pairs=csv_label_pairs,
            sequence_length=sequence_length,
            stride=stride
        )
