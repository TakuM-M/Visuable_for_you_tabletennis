"""
メモリベースの骨格シーケンスデータセット

メモリ上の骨格シーケンスデータ(numpy配列)から直接読み込む
推論やリアルタイム処理で使用
"""
import numpy as np
from typing import Optional, List

from src.datasets.base_dataset import BasePoseSequenceDataset


class MemoryPoseSequenceDataset(BasePoseSequenceDataset):
    """
    メモリベースの骨格シーケンスデータセット

    メモリ上の骨格データから直接データセットを作成
    CSV I/Oをスキップ
    """

    def __init__(
        self,
        pose_data: np.ndarray,
        frames: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        sequence_length: int = 30,
        stride: int = 1,
        keypoint_features: Optional[List[str]] = None
    ):
        """
        メモリ骨格シーケンスデータセットを初期化

        Args:
            pose_data: 正規化された骨格データ (num_frames, num_features)
                      num_features = len(keypoint_names) * 2 (x, y座標)
            frames: フレーム番号配列 (num_frames,)。Noneの場合は0から連番を使用
            labels: ラベル配列 (num_frames,)。Noneの場合は全フレームを0としてラベル付け
            sequence_length: シーケンス長（フレーム数）
            stride: シーケンス抽出時のストライド
            keypoint_features: 使用するキーポイント名のリスト（Noneの場合は全て使用）

        Note:
            入力pose_dataは既に正規化されていることを想定
        """
        self.pose_data = pose_data
        self.input_frames = frames
        self.input_labels = labels

        # 基底クラスを初期化
        super().__init__(
            sequence_length=sequence_length,
            stride=stride,
            keypoint_features=keypoint_features
        )

        # データを読み込み
        self._load_data()

        # シーケンスインデックスを作成
        self.sequence_indices = self._create_sequence_indices()

        print(f"MemoryPoseSequenceDataset initialized:")
        print(f"  Total frames: {len(self.features)}")
        print(f"  Total sequences: {len(self.sequence_indices)}")
        print(f"  Sequence length: {self.sequence_length}")
        print(f"  Feature dimensions: {self.features.shape[1]}")
        print(f"  Play frames: {np.sum(self.labels == 1)} / {len(self.labels)}")

    def _load_data(self):
        """メモリ内データを検証して設定"""
        # データを検証
        expected_features = len(self.keypoint_names) * 2
        if self.pose_data.shape[1] != expected_features:
            raise ValueError(
                f"pose_dataの特徴量次元が無効です。"
                f"期待値: {expected_features} (num_keypoints: {len(self.keypoint_names)} x 2), "
                f"実際: {self.pose_data.shape[1]}"
            )

        # 特徴量を保存
        self.features = self.pose_data.astype(np.float32)

        # フレーム番号
        if self.input_frames is None:
            self.frames = np.arange(len(self.pose_data), dtype=np.int64)
        else:
            if len(self.input_frames) != len(self.pose_data):
                raise ValueError(
                    f"framesの長さがpose_dataと一致しません。"
                    f"pose_data: {len(self.pose_data)}, frames: {len(self.input_frames)}"
                )
            self.frames = self.input_frames.astype(np.int64)

        # ラベル
        if self.input_labels is None:
            self.labels = np.zeros(len(self.pose_data), dtype=np.int64)
        else:
            if len(self.input_labels) != len(self.pose_data):
                raise ValueError(
                    f"labelsの長さがpose_dataと一致しません。"
                    f"pose_data: {len(self.pose_data)}, labels: {len(self.input_labels)}"
                )
            self.labels = self.input_labels.astype(np.int64)
