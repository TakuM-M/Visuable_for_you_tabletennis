"""
オンラインデータ拡張モジュール

訓練時にシーケンス単位でリアルタイムに適用するnumpy配列ベースのAugmentor
既存のPoseAugmentationPipeline（CSV/DataFrameベースのオフライン拡張）とは別に、
DataLoaderの__getitem__内で呼ばれることを想定

入力形状: (sequence_length, num_features)
  - num_features=34: 17キーポイント × 2座標 (x, y)
  - num_features=102: 34座標 + 34速度 + 34加速度（motion features有効時）

座標配列のレイアウト:
  [nose_x, nose_y, left_eye_x, left_eye_y, ..., right_ankle_x, right_ankle_y]
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


# COCO形式の左右キーポイントペア（インデックス）
# キーポイント順: nose(0), left_eye(1), right_eye(2), left_ear(3), right_ear(4),
#   left_shoulder(5), right_shoulder(6), left_elbow(7), right_elbow(8),
#   left_wrist(9), right_wrist(10), left_hip(11), right_hip(12),
#   left_knee(13), right_knee(14), left_ankle(15), right_ankle(16)
LEFT_RIGHT_PAIRS = [
    (1, 2),    # left_eye, right_eye
    (3, 4),    # left_ear, right_ear
    (5, 6),    # left_shoulder, right_shoulder
    (7, 8),    # left_elbow, right_elbow
    (9, 10),   # left_wrist, right_wrist
    (11, 12),  # left_hip, right_hip
    (13, 14),  # left_knee, right_knee
    (15, 16),  # left_ankle, right_ankle
]


@dataclass
class OnlineAugmentationConfig:
    """オンラインデータ拡張の設定"""

    # 左右反転
    horizontal_flip: bool = True
    horizontal_flip_prob: float = 0.5

    # ガウシアンノイズ
    add_noise: bool = True
    noise_std: float = 0.02

    # スケーリング
    scaling: bool = True
    scale_range: Tuple[float, float] = (0.9, 1.1)

    # 関節ドロップアウト（ランダムに座標を0にする）
    keypoint_dropout: bool = True
    dropout_prob: float = 0.05

    # 時間マスク（連続フレームをマスク）
    temporal_mask: bool = True
    temporal_mask_max_frames: int = 5

    def __post_init__(self):
        if not 0.0 <= self.horizontal_flip_prob <= 1.0:
            raise ValueError("horizontal_flip_prob must be between 0.0 and 1.0")
        if self.noise_std < 0:
            raise ValueError("noise_std must be non-negative")
        if len(self.scale_range) != 2 or self.scale_range[0] > self.scale_range[1]:
            raise ValueError("scale_range must be (min, max) with min <= max")
        if not 0.0 <= self.dropout_prob <= 1.0:
            raise ValueError("dropout_prob must be between 0.0 and 1.0")
        if self.temporal_mask_max_frames < 1:
            raise ValueError("temporal_mask_max_frames must be at least 1")


class OnlineAugmentor:
    """
    オンラインデータ拡張

    __getitem__内で呼ばれ、シーケンス単位で拡張を適用する。
    毎回異なるランダム拡張が適用されるため、エポックごとに異なる
    訓練データを生成できる。
    """

    def __init__(self, config: OnlineAugmentationConfig):
        self.config = config

    def __call__(self, features: np.ndarray) -> np.ndarray:
        """
        シーケンスにデータ拡張を適用

        Args:
            features: (sequence_length, num_features) の配列
                      num_features は 34 または 102

        Returns:
            拡張された (sequence_length, num_features) の配列
        """
        features = features.copy()
        num_features = features.shape[1]

        # motion features（速度・加速度）の有無を判定
        # 102次元の場合: [0:34]=座標, [34:68]=速度, [68:102]=加速度
        has_motion = num_features == 102

        if self.config.horizontal_flip and np.random.rand() < self.config.horizontal_flip_prob:
            features = self._horizontal_flip(features, has_motion)

        if self.config.scaling:
            features = self._scaling(features, has_motion)

        if self.config.add_noise:
            features = self._add_noise(features, has_motion)

        if self.config.keypoint_dropout:
            features = self._keypoint_dropout(features, has_motion)

        if self.config.temporal_mask:
            features = self._temporal_mask(features)

        return features

    def _horizontal_flip(self, features: np.ndarray, has_motion: bool) -> np.ndarray:
        """
        左右反転

        正規化座標のx成分を反転し、左右のキーポイントを入れ替える。
        motion featuresがある場合は速度・加速度にも同様に適用。
        """
        # 座標部分のスライス群を特定
        slices = [slice(0, 34)]  # 座標
        if has_motion:
            slices.append(slice(34, 68))   # 速度
            slices.append(slice(68, 102))  # 加速度

        for s in slices:
            block = features[:, s]  # (seq, 34)

            # x座標を反転（偶数インデックスがx）
            block[:, 0::2] = -block[:, 0::2]

            # 左右キーポイントを入れ替え
            for left_idx, right_idx in LEFT_RIGHT_PAIRS:
                left_x, left_y = left_idx * 2, left_idx * 2 + 1
                right_x, right_y = right_idx * 2, right_idx * 2 + 1
                # swap
                block[:, [left_x, left_y, right_x, right_y]] = \
                    block[:, [right_x, right_y, left_x, left_y]]

            features[:, s] = block

        return features

    def _scaling(self, features: np.ndarray, has_motion: bool) -> np.ndarray:
        """スケーリング（全座標に同一スケールを適用）"""
        scale = np.random.uniform(*self.config.scale_range)

        # 座標にスケール適用
        features[:, :34] *= scale

        if has_motion:
            # 速度・加速度にも同じスケールを適用（微分はスケールに対して線形）
            features[:, 34:68] *= scale
            features[:, 68:102] *= scale

        return features

    def _add_noise(self, features: np.ndarray, has_motion: bool) -> np.ndarray:
        """ガウシアンノイズを座標に追加"""
        # 座標にのみノイズを追加（速度・加速度は座標から再計算すべきだが、
        # 小さなノイズなので近似的にそのまま使う）
        noise = np.random.normal(0, self.config.noise_std, features[:, :34].shape)
        features[:, :34] += noise.astype(features.dtype)

        return features

    def _keypoint_dropout(self, features: np.ndarray, has_motion: bool) -> np.ndarray:
        """ランダムにキーポイントを0にする"""
        seq_len = features.shape[0]

        for kp_idx in range(17):
            mask = np.random.rand(seq_len) < self.config.dropout_prob
            if not mask.any():
                continue

            x_idx = kp_idx * 2
            y_idx = kp_idx * 2 + 1

            # 座標を0に
            features[mask, x_idx] = 0.0
            features[mask, y_idx] = 0.0

            if has_motion:
                # 速度・加速度も0に
                features[mask, 34 + x_idx] = 0.0
                features[mask, 34 + y_idx] = 0.0
                features[mask, 68 + x_idx] = 0.0
                features[mask, 68 + y_idx] = 0.0

        return features

    def _temporal_mask(self, features: np.ndarray) -> np.ndarray:
        """連続フレームをマスク（SpecAugment的手法）"""
        seq_len = features.shape[0]
        mask_len = np.random.randint(1, self.config.temporal_mask_max_frames + 1)
        mask_start = np.random.randint(0, max(1, seq_len - mask_len))

        features[mask_start:mask_start + mask_len] = 0.0

        return features
