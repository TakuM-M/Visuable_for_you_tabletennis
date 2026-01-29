"""
プレー検知用データセット

正規化された骨格データCSVから時系列データセットを作成
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.detection.yolo_tracker import KEYPOINT_NAMES


class PoseSequenceDataset(Dataset):
    """
    時系列骨格データのデータセット

    正規化された骨格データから固定長のシーケンスを切り出してラベル付けする
    """

    def __init__(
        self,
        csv_path: str,
        label_path: Optional[str] = None,
        sequence_length: int = 30,
        stride: int = 1,
        keypoint_features: List[str] = None,
        normalize_features: bool = True
    ):
        """
        初期化

        Args:
            csv_path: 正規化された骨格データのCSVパス
            label_path: ラベルファイルのパス（CSV形式: frame,label）
                       Noneの場合は全フレームをプレー外(0)とする
            sequence_length: シーケンスの長さ（フレーム数）
            stride: シーケンス切り出しのストライド
            keypoint_features: 使用するキーポイント名のリスト（Noneの場合は全キーポイント）
            normalize_features: 特徴量を正規化するか
        """
        self.csv_path = Path(csv_path)
        self.label_path = Path(label_path) if label_path else None
        self.sequence_length = sequence_length
        self.stride = stride
        self.normalize_features = normalize_features

        # データ読み込み
        self.data_df = pd.read_csv(csv_path)

        # ラベル読み込み
        if self.label_path and self.label_path.exists():
            self.labels_df = pd.read_csv(label_path)
            # フレーム番号でマージ
            self.data_df = self.data_df.merge(
                self.labels_df,
                on='frame',
                how='left'
            )
            self.data_df['label'] = self.data_df['label'].fillna(0).astype(int)
        else:
            # ラベルがない場合は全て0（プレー外）
            self.data_df['label'] = 0

        # 使用するキーポイントを決定
        if keypoint_features is None:
            self.keypoint_names = KEYPOINT_NAMES
        else:
            self.keypoint_names = keypoint_features

        # 特徴量カラムを作成
        self.feature_columns = []
        for kp_name in self.keypoint_names:
            self.feature_columns.extend([
                f'{kp_name}_norm_x',
                f'{kp_name}_norm_y'
            ])

        # 特徴量を抽出
        self.features = self.data_df[self.feature_columns].values.astype(np.float32)
        self.labels = self.data_df['label'].values.astype(np.int64)
        self.frames = self.data_df['frame'].values

        # 特徴量の正規化（オプション）
        if self.normalize_features:
            self.feature_mean = np.mean(self.features, axis=0)
            self.feature_std = np.std(self.features, axis=0) + 1e-8
            self.features = (self.features - self.feature_mean) / self.feature_std
        else:
            self.feature_mean = None
            self.feature_std = None

        # シーケンスのインデックスを作成
        self.sequence_indices = self._create_sequence_indices()

        print(f"データセット初期化完了:")
        print(f"  CSVパス: {self.csv_path}")
        print(f"  総フレーム数: {len(self.data_df)}")
        print(f"  シーケンス数: {len(self.sequence_indices)}")
        print(f"  シーケンス長: {self.sequence_length}")
        print(f"  特徴量次元: {self.features.shape[1]}")
        print(f"  プレー中フレーム: {np.sum(self.labels == 1)} / {len(self.labels)}")

    def _create_sequence_indices(self) -> List[Tuple[int, int]]:
        """
        シーケンスの開始・終了インデックスのリストを作成

        Returns:
            [(start_idx, end_idx), ...] のリスト
        """
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
            'start_frame': frames[0],
            'end_frame': frames[-1],
            'start_idx': start_idx,
            'end_idx': end_idx
        }

        return features_tensor, labels_tensor, metadata

    def get_statistics(self) -> Dict:
        """データセットの統計情報を取得"""
        return {
            'total_frames': len(self.data_df),
            'total_sequences': len(self.sequence_indices),
            'sequence_length': self.sequence_length,
            'stride': self.stride,
            'num_features': self.features.shape[1],
            'num_keypoints': len(self.keypoint_names),
            'play_frames': int(np.sum(self.labels == 1)),
            'non_play_frames': int(np.sum(self.labels == 0)),
            'play_ratio': float(np.mean(self.labels == 1))
        }


class InMemoryPoseSequenceDataset(Dataset):
    """
    メモリ上の骨格データから直接データセットを作成

    推論時やリアルタイム処理で使用
    CSVの入出力を省略してメモリ効率と速度を向上
    """

    def __init__(
        self,
        pose_data: np.ndarray,
        frames: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        sequence_length: int = 30,
        stride: int = 1,
        keypoint_features: List[str] = None,
        normalize_features: bool = True,
        feature_mean: Optional[np.ndarray] = None,
        feature_std: Optional[np.ndarray] = None
    ):
        """
        初期化

        Args:
            pose_data: 正規化済み骨格データ (num_frames, num_features)
                      num_features = len(keypoint_names) * 2 (x, y座標)
            frames: フレーム番号配列 (num_frames,)。Noneの場合は0から連番
            labels: ラベル配列 (num_frames,)。Noneの場合は全て0
            sequence_length: シーケンスの長さ（フレーム数）
            stride: シーケンス切り出しのストライド
            keypoint_features: 使用するキーポイント名のリスト（Noneの場合は全キーポイント）
            normalize_features: 特徴量を正規化するか
            feature_mean: 正規化用の平均値（学習時と同じものを使用）。Noneの場合は自動計算
            feature_std: 正規化用の標準偏差（学習時と同じものを使用）。Noneの場合は自動計算
        """
        self.sequence_length = sequence_length
        self.stride = stride
        self.normalize_features = normalize_features

        # 使用するキーポイントを決定
        if keypoint_features is None:
            self.keypoint_names = KEYPOINT_NAMES
        else:
            self.keypoint_names = keypoint_features

        # データの検証
        expected_features = len(self.keypoint_names) * 2
        if pose_data.shape[1] != expected_features:
            raise ValueError(
                f"pose_dataの特徴量次元が不正です。"
                f"期待値: {expected_features} (キーポイント数: {len(self.keypoint_names)} × 2), "
                f"実際: {pose_data.shape[1]}"
            )

        # 特徴量を保存
        self.features = pose_data.astype(np.float32)

        # フレーム番号
        if frames is None:
            self.frames = np.arange(len(pose_data), dtype=np.int64)
        else:
            if len(frames) != len(pose_data):
                raise ValueError(
                    f"framesの長さがpose_dataと一致しません。"
                    f"pose_data: {len(pose_data)}, frames: {len(frames)}"
                )
            self.frames = frames.astype(np.int64)

        # ラベル
        if labels is None:
            self.labels = np.zeros(len(pose_data), dtype=np.int64)
        else:
            if len(labels) != len(pose_data):
                raise ValueError(
                    f"labelsの長さがpose_dataと一致しません。"
                    f"pose_data: {len(pose_data)}, labels: {len(labels)}"
                )
            self.labels = labels.astype(np.int64)

        # 特徴量の正規化
        if self.normalize_features:
            if feature_mean is not None and feature_std is not None:
                # 学習時のパラメータを使用
                self.feature_mean = feature_mean
                self.feature_std = feature_std
            else:
                # 新規に計算
                self.feature_mean = np.mean(self.features, axis=0)
                self.feature_std = np.std(self.features, axis=0) + 1e-8

            self.features = (self.features - self.feature_mean) / self.feature_std
        else:
            self.feature_mean = None
            self.feature_std = None

        # シーケンスのインデックスを作成
        self.sequence_indices = self._create_sequence_indices()

        print(f"InMemoryPoseSequenceDataset初期化完了:")
        print(f"  総フレーム数: {len(self.features)}")
        print(f"  シーケンス数: {len(self.sequence_indices)}")
        print(f"  シーケンス長: {self.sequence_length}")
        print(f"  特徴量次元: {self.features.shape[1]}")
        print(f"  プレー中フレーム: {np.sum(self.labels == 1)} / {len(self.labels)}")

    def _create_sequence_indices(self) -> List[Tuple[int, int]]:
        """
        シーケンスの開始・終了インデックスのリストを作成

        Returns:
            [(start_idx, end_idx), ...] のリスト
        """
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
            'start_frame': frames[0],
            'end_frame': frames[-1],
            'start_idx': start_idx,
            'end_idx': end_idx
        }

        return features_tensor, labels_tensor, metadata

    def get_statistics(self) -> Dict:
        """データセットの統計情報を取得"""
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


class MultiVideoPoseDataset(Dataset):
    """
    複数動画のデータを扱うデータセット

    複数のCSVファイルから統合されたデータセットを作成
    """

    def __init__(
        self,
        csv_dir: str,
        label_dir: Optional[str] = None,
        sequence_length: int = 30,
        stride: int = 1,
        normalize_features: bool = True
    ):
        """
        初期化

        Args:
            csv_dir: 正規化された骨格データCSVが入っているディレクトリ
            label_dir: ラベルファイルが入っているディレクトリ
            sequence_length: シーケンスの長さ
            stride: ストライド
            normalize_features: 特徴量を正規化するか
        """
        self.csv_dir = Path(csv_dir)
        self.label_dir = Path(label_dir) if label_dir else None
        self.sequence_length = sequence_length
        self.stride = stride
        self.normalize_features = normalize_features

        # 各動画のデータセットを作成
        self.datasets = []
        csv_files = sorted(self.csv_dir.glob("*_normalized.csv"))

        if len(csv_files) == 0:
            raise ValueError(f"No CSV files found in {self.csv_dir}")

        for csv_file in csv_files:
            # 対応するラベルファイルを探す
            label_file = None
            if self.label_dir:
                video_name = csv_file.stem.replace("_normalized", "")
                label_file = self.label_dir / f"{video_name}_labels.csv"
                if not label_file.exists():
                    label_file = None

            # データセット作成
            dataset = PoseSequenceDataset(
                csv_path=str(csv_file),
                label_path=str(label_file) if label_file else None,
                sequence_length=sequence_length,
                stride=stride,
                normalize_features=False  # 後でまとめて正規化
            )
            self.datasets.append(dataset)

        # 全データを結合
        self._combine_datasets()

        print(f"\n統合データセット:")
        print(f"  動画数: {len(csv_files)}")
        print(f"  総シーケンス数: {len(self)}")

    def _combine_datasets(self):
        """複数のデータセットを結合"""
        # 各データセットのシーケンス数を記録
        self.dataset_lengths = [len(ds) for ds in self.datasets]
        self.cumulative_lengths = np.cumsum([0] + self.dataset_lengths)

        # 全データの特徴量を結合して正規化
        if self.normalize_features:
            all_features = np.concatenate(
                [ds.features for ds in self.datasets],
                axis=0
            )
            self.feature_mean = np.mean(all_features, axis=0)
            self.feature_std = np.std(all_features, axis=0) + 1e-8

            # 各データセットに正規化パラメータを適用
            for ds in self.datasets:
                ds.features = (ds.features - self.feature_mean) / self.feature_std

    def __len__(self) -> int:
        """データセットのサイズ"""
        return sum(self.dataset_lengths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        データの取得

        Args:
            idx: グローバルインデックス

        Returns:
            features: 特徴量テンソル
            labels: ラベルテンソル
            metadata: メタデータ
        """
        # どのデータセットに属するか判定
        dataset_idx = np.searchsorted(self.cumulative_lengths[1:], idx, side='right')
        local_idx = idx - self.cumulative_lengths[dataset_idx]

        # 該当するデータセットから取得
        features, labels, metadata = self.datasets[dataset_idx][local_idx]
        metadata['dataset_idx'] = dataset_idx

        return features, labels, metadata


def collate_fn(batch):
    """
    カスタムcollate関数（バッチ作成用）

    可変長シーケンスに対応する場合に使用
    """
    features, labels, metadata = zip(*batch)

    # テンソルをスタック
    features_batch = torch.stack(features)
    labels_batch = torch.stack(labels)

    return features_batch, labels_batch, metadata


def create_dataloaders(
    train_csv: str,
    val_csv: Optional[str] = None,
    train_labels: Optional[str] = None,
    val_labels: Optional[str] = None,
    batch_size: int = 32,
    sequence_length: int = 30,
    stride: int = 5,
    num_workers: int = 4
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    訓練・検証用DataLoaderを作成

    Args:
        train_csv: 訓練データCSV
        val_csv: 検証データCSV（Noneの場合は検証なし）
        train_labels: 訓練ラベルCSV
        val_labels: 検証ラベルCSV
        batch_size: バッチサイズ
        sequence_length: シーケンス長
        stride: ストライド
        num_workers: ワーカー数

    Returns:
        train_loader: 訓練用DataLoader
        val_loader: 検証用DataLoader（val_csvがNoneの場合はNone）
    """
    # 訓練データ
    train_dataset = PoseSequenceDataset(
        csv_path=train_csv,
        label_path=train_labels,
        sequence_length=sequence_length,
        stride=stride
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # 検証データ
    val_loader = None
    if val_csv:
        val_dataset = PoseSequenceDataset(
            csv_path=val_csv,
            label_path=val_labels,
            sequence_length=sequence_length,
            stride=stride
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )

    return train_loader, val_loader


def test_dataset():
    """データセットのテスト"""
    print("PoseSequenceDataset テスト")
    print("=" * 60)

    # ダミーCSVを作成
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        # ダミーデータ
        num_frames = 100
        data = {
            'frame': range(num_frames),
            'timestamp': np.linspace(0, 3.33, num_frames),
            'track_id': [1] * num_frames,
            'hip_center_x': np.random.rand(num_frames) * 640,
            'hip_center_y': np.random.rand(num_frames) * 480,
            'scale_factor': np.random.rand(num_frames) * 100 + 50,
            'confidence': np.random.rand(num_frames) * 0.5 + 0.5,
            'is_valid': [True] * num_frames
        }

        # 17キーポイント×2座標
        for kp_name in KEYPOINT_NAMES:
            data[f'{kp_name}_norm_x'] = np.random.randn(num_frames)
            data[f'{kp_name}_norm_y'] = np.random.randn(num_frames)
            data[f'{kp_name}_conf'] = np.random.rand(num_frames)

        df = pd.DataFrame(data)
        csv_path = os.path.join(tmpdir, "test_data.csv")
        df.to_csv(csv_path, index=False)

        # ダミーラベル
        labels = {
            'frame': range(num_frames),
            'label': [1 if 20 <= i < 60 else 0 for i in range(num_frames)]
        }
        labels_df = pd.DataFrame(labels)
        label_path = os.path.join(tmpdir, "test_labels.csv")
        labels_df.to_csv(label_path, index=False)

        # データセット作成
        dataset = PoseSequenceDataset(
            csv_path=csv_path,
            label_path=label_path,
            sequence_length=30,
            stride=5
        )

        print(f"\nデータセット統計:")
        stats = dataset.get_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")

        # サンプル取得
        print(f"\n最初のサンプル:")
        features, labels, metadata = dataset[0]
        print(f"  特徴量shape: {features.shape}")
        print(f"  ラベルshape: {labels.shape}")
        print(f"  メタデータ: {metadata}")

        # DataLoader作成
        print(f"\nDataLoaderテスト:")
        loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
        for batch_features, batch_labels, batch_metadata in loader:
            print(f"  バッチ特徴量shape: {batch_features.shape}")
            print(f"  バッチラベルshape: {batch_labels.shape}")
            print(f"  バッチサイズ: {len(batch_metadata)}")
            break

    print("\n" + "=" * 60)
    print("テスト完了")


if __name__ == "__main__":
    test_dataset()
