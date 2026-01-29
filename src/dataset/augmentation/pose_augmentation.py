"""
骨格データのデータ拡張モジュール

卓球の動作データに特化したデータ拡張手法を提供します。
正規化された骨格データ (NormalizedPoseData) に対して適用できます。

主な拡張手法:
1. 左右反転 (Horizontal Flip) - 右利き/左利きの変換
2. ガウシアンノイズ (Gaussian Noise) - 検出誤差のシミュレーション
3. 回転 (Rotation) - 体の向きの変化
4. 関節ドロップアウト (Keypoint Dropout) - オクルージョンのシミュレーション
5. 時間スケーリング (Temporal Scaling) - 動作速度の変更
6. 時間的ジッター (Temporal Jitter) - 微小な時間変動

Usage:
    from src.dataset.augmentation import PoseAugmentation, AugmentationConfig

    # 設定を作成
    config = AugmentationConfig(
        horizontal_flip=True,
        rotation_range=15.0,
        noise_std=0.02
    )

    # 拡張器を作成
    augmentor = PoseAugmentation(config)

    # 単一のポーズデータを拡張
    augmented_pose = augmentor.augment(normalized_pose_data)

    # 時系列データを拡張（時間的拡張を含む）
    augmented_sequence = augmentor.augment_sequence(pose_sequence)
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.dataset.annotation.processors.pose_normalizer import NormalizedPoseData


@dataclass
class AugmentationConfig:
    """データ拡張の設定"""

    # 左右反転
    horizontal_flip: bool = False
    horizontal_flip_prob: float = 0.5

    # ガウシアンノイズ
    add_noise: bool = False
    noise_std: float = 0.02  # 正規化座標系でのノイズ標準偏差

    # 回転
    rotation: bool = False
    rotation_range: float = 15.0  # 回転角度の範囲（度）

    # スケーリング
    scaling: bool = False
    scale_range: Tuple[float, float] = (0.9, 1.1)  # スケール範囲

    # 関節ドロップアウト
    keypoint_dropout: bool = False
    dropout_prob: float = 0.1  # 各関節がドロップアウトする確率

    # 時間的ジッター（系列データ用）
    temporal_jitter: bool = False
    jitter_std: float = 0.5  # フレーム単位でのジッター標準偏差

    # 時間スケーリング（系列データ用）
    temporal_scaling: bool = False
    temporal_scale_range: Tuple[float, float] = (0.8, 1.2)

    # ランダムシード
    random_seed: Optional[int] = None

    def __post_init__(self):
        """初期化後の検証"""
        if self.random_seed is not None:
            np.random.seed(self.random_seed)


class PoseAugmentation:
    """
    骨格データのデータ拡張クラス

    正規化された骨格データに対して各種データ拡張を適用します。
    """

    # COCO形式のキーポイントインデックス
    KEYPOINT_NAMES = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]

    # 左右のキーポイントのペア（反転時に入れ替える）
    LEFT_RIGHT_PAIRS = [
        (1, 2),   # left_eye <-> right_eye
        (3, 4),   # left_ear <-> right_ear
        (5, 6),   # left_shoulder <-> right_shoulder
        (7, 8),   # left_elbow <-> right_elbow
        (9, 10),  # left_wrist <-> right_wrist
        (11, 12), # left_hip <-> right_hip
        (13, 14), # left_knee <-> right_knee
        (15, 16), # left_ankle <-> right_ankle
    ]

    def __init__(self, config: AugmentationConfig):
        """
        初期化

        Args:
            config: データ拡張の設定
        """
        self.config = config

    def augment(self, pose_data: NormalizedPoseData) -> NormalizedPoseData:
        """
        単一のポーズデータを拡張

        Args:
            pose_data: 正規化された骨格データ

        Returns:
            拡張された骨格データ
        """
        # コピーを作成
        keypoints = pose_data.normalized_keypoints.copy()
        confidences = pose_data.keypoint_confidences.copy()

        # 各拡張を適用
        if self.config.horizontal_flip and np.random.rand() < self.config.horizontal_flip_prob:
            keypoints, confidences = self._apply_horizontal_flip(keypoints, confidences)

        if self.config.rotation:
            keypoints = self._apply_rotation(keypoints, confidences)

        if self.config.scaling:
            keypoints = self._apply_scaling(keypoints)

        if self.config.add_noise:
            keypoints = self._apply_noise(keypoints, confidences)

        if self.config.keypoint_dropout:
            keypoints, confidences = self._apply_keypoint_dropout(keypoints, confidences)

        # 新しいNormalizedPoseDataを作成
        augmented = NormalizedPoseData(
            track_id=pose_data.track_id,
            frame=pose_data.frame,
            timestamp=pose_data.timestamp,
            normalized_keypoints=keypoints,
            keypoint_confidences=confidences,
            hip_center=pose_data.hip_center,
            scale_factor=pose_data.scale_factor,
            confidence=pose_data.confidence,
            is_valid=pose_data.is_valid
        )

        return augmented

    def augment_sequence(
        self,
        pose_sequence: List[NormalizedPoseData]
    ) -> List[NormalizedPoseData]:
        """
        時系列の骨格データを拡張

        Args:
            pose_sequence: 正規化された骨格データのリスト

        Returns:
            拡張された骨格データのリスト
        """
        if len(pose_sequence) == 0:
            return []

        # 時間的拡張を適用
        if self.config.temporal_scaling:
            pose_sequence = self._apply_temporal_scaling(pose_sequence)

        if self.config.temporal_jitter:
            pose_sequence = self._apply_temporal_jitter(pose_sequence)

        # 各フレームに空間的拡張を適用
        augmented_sequence = []
        for pose_data in pose_sequence:
            augmented = self.augment(pose_data)
            augmented_sequence.append(augmented)

        return augmented_sequence

    def _apply_horizontal_flip(
        self,
        keypoints: np.ndarray,
        confidences: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        左右反転を適用

        Args:
            keypoints: キーポイント配列 (17, 2)
            confidences: 信頼度配列 (17,)

        Returns:
            反転されたキーポイントと信頼度
        """
        flipped_keypoints = keypoints.copy()
        flipped_confidences = confidences.copy()

        # X座標を反転
        flipped_keypoints[:, 0] = -flipped_keypoints[:, 0]

        # 左右のキーポイントを入れ替え
        for left_idx, right_idx in self.LEFT_RIGHT_PAIRS:
            # キーポイント座標を入れ替え
            flipped_keypoints[[left_idx, right_idx]] = flipped_keypoints[[right_idx, left_idx]]
            # 信頼度も入れ替え
            flipped_confidences[[left_idx, right_idx]] = flipped_confidences[[right_idx, left_idx]]

        return flipped_keypoints, flipped_confidences

    def _apply_rotation(
        self,
        keypoints: np.ndarray,
        confidences: np.ndarray
    ) -> np.ndarray:
        """
        回転を適用

        Args:
            keypoints: キーポイント配列 (17, 2)
            confidences: 信頼度配列 (17,)

        Returns:
            回転されたキーポイント
        """
        # ランダムな回転角度（ラジアン）
        angle_deg = np.random.uniform(-self.config.rotation_range, self.config.rotation_range)
        angle_rad = np.deg2rad(angle_deg)

        # 回転行列
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)
        rotation_matrix = np.array([
            [cos_angle, -sin_angle],
            [sin_angle, cos_angle]
        ])

        rotated_keypoints = keypoints.copy()

        # 信頼度が高いキーポイントのみ回転
        for i in range(17):
            if confidences[i] > 0.3:
                rotated_keypoints[i] = rotation_matrix @ keypoints[i]

        return rotated_keypoints

    def _apply_scaling(self, keypoints: np.ndarray) -> np.ndarray:
        """
        スケーリングを適用

        Args:
            keypoints: キーポイント配列 (17, 2)

        Returns:
            スケーリングされたキーポイント
        """
        scale = np.random.uniform(*self.config.scale_range)
        return keypoints * scale

    def _apply_noise(
        self,
        keypoints: np.ndarray,
        confidences: np.ndarray
    ) -> np.ndarray:
        """
        ガウシアンノイズを適用

        Args:
            keypoints: キーポイント配列 (17, 2)
            confidences: 信頼度配列 (17,)

        Returns:
            ノイズが付加されたキーポイント
        """
        noisy_keypoints = keypoints.copy()

        # 信頼度が高いキーポイントにのみノイズを付加
        for i in range(17):
            if confidences[i] > 0.3:
                noise = np.random.normal(0, self.config.noise_std, size=2)
                noisy_keypoints[i] += noise

        return noisy_keypoints

    def _apply_keypoint_dropout(
        self,
        keypoints: np.ndarray,
        confidences: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        関節ドロップアウトを適用（オクルージョンのシミュレーション）

        Args:
            keypoints: キーポイント配列 (17, 2)
            confidences: 信頼度配列 (17,)

        Returns:
            ドロップアウト適用後のキーポイントと信頼度
        """
        dropout_keypoints = keypoints.copy()
        dropout_confidences = confidences.copy()

        # 各キーポイントに対してドロップアウトを適用
        for i in range(17):
            if np.random.rand() < self.config.dropout_prob:
                dropout_keypoints[i] = 0.0
                dropout_confidences[i] = 0.0

        return dropout_keypoints, dropout_confidences

    def _apply_temporal_scaling(
        self,
        pose_sequence: List[NormalizedPoseData]
    ) -> List[NormalizedPoseData]:
        """
        時間スケーリングを適用（動作速度の変更）

        Args:
            pose_sequence: 骨格データのリスト

        Returns:
            時間スケーリングされた骨格データのリスト
        """
        if len(pose_sequence) <= 1:
            return pose_sequence

        scale = np.random.uniform(*self.config.temporal_scale_range)
        original_length = len(pose_sequence)
        new_length = int(original_length * scale)
        new_length = max(2, new_length)  # 最低2フレーム

        # 線形補間でリサンプリング
        indices = np.linspace(0, original_length - 1, new_length)
        scaled_sequence = []

        for idx in indices:
            idx_floor = int(np.floor(idx))
            idx_ceil = min(int(np.ceil(idx)), original_length - 1)

            if idx_floor == idx_ceil:
                scaled_sequence.append(pose_sequence[idx_floor])
            else:
                # 線形補間
                weight = idx - idx_floor
                pose1 = pose_sequence[idx_floor]
                pose2 = pose_sequence[idx_ceil]

                # キーポイントを補間
                interpolated_keypoints = (
                    (1 - weight) * pose1.normalized_keypoints +
                    weight * pose2.normalized_keypoints
                )

                # 新しいデータを作成
                interpolated_pose = NormalizedPoseData(
                    track_id=pose1.track_id,
                    frame=pose1.frame,
                    timestamp=pose1.timestamp * (1 - weight) + pose2.timestamp * weight,
                    normalized_keypoints=interpolated_keypoints,
                    keypoint_confidences=pose1.keypoint_confidences,
                    hip_center=pose1.hip_center,
                    scale_factor=pose1.scale_factor,
                    confidence=pose1.confidence,
                    is_valid=pose1.is_valid
                )
                scaled_sequence.append(interpolated_pose)

        return scaled_sequence

    def _apply_temporal_jitter(
        self,
        pose_sequence: List[NormalizedPoseData]
    ) -> List[NormalizedPoseData]:
        """
        時間的ジッターを適用（微小な時間変動）

        Args:
            pose_sequence: 骨格データのリスト

        Returns:
            ジッター適用後の骨格データのリスト
        """
        if len(pose_sequence) <= 2:
            return pose_sequence

        jittered_sequence = []

        for i, pose_data in enumerate(pose_sequence):
            # ジッター量を計算（両端は変更しない）
            if i == 0 or i == len(pose_sequence) - 1:
                jittered_sequence.append(pose_data)
                continue

            jitter = np.random.normal(0, self.config.jitter_std)
            jitter = np.clip(jitter, -2, 2)  # 最大±2フレーム

            # ジッター適用後のインデックス
            jittered_idx = i + jitter
            jittered_idx = np.clip(jittered_idx, 0, len(pose_sequence) - 1)

            idx_floor = int(np.floor(jittered_idx))
            idx_ceil = min(int(np.ceil(jittered_idx)), len(pose_sequence) - 1)

            if idx_floor == idx_ceil:
                jittered_sequence.append(pose_sequence[idx_floor])
            else:
                # 線形補間
                weight = jittered_idx - idx_floor
                pose1 = pose_sequence[idx_floor]
                pose2 = pose_sequence[idx_ceil]

                interpolated_keypoints = (
                    (1 - weight) * pose1.normalized_keypoints +
                    weight * pose2.normalized_keypoints
                )

                interpolated_pose = NormalizedPoseData(
                    track_id=pose1.track_id,
                    frame=pose1.frame,
                    timestamp=pose1.timestamp * (1 - weight) + pose2.timestamp * weight,
                    normalized_keypoints=interpolated_keypoints,
                    keypoint_confidences=pose1.keypoint_confidences,
                    hip_center=pose1.hip_center,
                    scale_factor=pose1.scale_factor,
                    confidence=pose1.confidence,
                    is_valid=pose1.is_valid
                )
                jittered_sequence.append(interpolated_pose)

        return jittered_sequence


def augment_normalized_pose(
    pose_data: NormalizedPoseData,
    horizontal_flip: bool = False,
    rotation_range: float = 0.0,
    noise_std: float = 0.0,
    dropout_prob: float = 0.0
) -> NormalizedPoseData:
    """
    正規化された骨格データを拡張する便利関数

    Args:
        pose_data: 正規化された骨格データ
        horizontal_flip: 左右反転を適用するか
        rotation_range: 回転角度の範囲（度）
        noise_std: ノイズの標準偏差
        dropout_prob: ドロップアウト確率

    Returns:
        拡張された骨格データ
    """
    config = AugmentationConfig(
        horizontal_flip=horizontal_flip,
        rotation=rotation_range > 0,
        rotation_range=rotation_range,
        add_noise=noise_std > 0,
        noise_std=noise_std,
        keypoint_dropout=dropout_prob > 0,
        dropout_prob=dropout_prob
    )

    augmentor = PoseAugmentation(config)
    return augmentor.augment(pose_data)


def main():
    """テスト用のメイン関数"""
    print("=" * 70)
    print("骨格データ拡張モジュール - テスト")
    print("=" * 70)

    # ダミーの正規化データを作成
    dummy_keypoints = np.random.randn(17, 2).astype(np.float32) * 0.5
    dummy_confidences = np.random.uniform(0.5, 1.0, 17).astype(np.float32)

    dummy_pose = NormalizedPoseData(
        track_id=1,
        frame=0,
        timestamp=0.0,
        normalized_keypoints=dummy_keypoints,
        keypoint_confidences=dummy_confidences,
        hip_center=(640.0, 480.0),
        scale_factor=200.0,
        confidence=0.9,
        is_valid=True
    )

    print("\n元のデータ:")
    print(f"  腰の中心: {dummy_pose.hip_center}")
    print(f"  スケールファクター: {dummy_pose.scale_factor}")
    print(f"  鼻の座標: {dummy_pose.normalized_keypoints[0]}")
    print(f"  左肩の座標: {dummy_pose.normalized_keypoints[5]}")
    print(f"  右肩の座標: {dummy_pose.normalized_keypoints[6]}")

    # テスト1: 左右反転
    print("\n" + "=" * 70)
    print("テスト1: 左右反転")
    print("=" * 70)

    config = AugmentationConfig(horizontal_flip=True, horizontal_flip_prob=1.0)
    augmentor = PoseAugmentation(config)
    flipped = augmentor.augment(dummy_pose)

    print(f"元の鼻X座標: {dummy_pose.normalized_keypoints[0, 0]:.3f}")
    print(f"反転後の鼻X座標: {flipped.normalized_keypoints[0, 0]:.3f}")
    print(f"元の左肩: {dummy_pose.normalized_keypoints[5]}")
    print(f"反転後の左肩（元の右肩と同じはず）: {flipped.normalized_keypoints[5]}")
    print(f"元の右肩: {dummy_pose.normalized_keypoints[6]}")

    # テスト2: 回転
    print("\n" + "=" * 70)
    print("テスト2: 回転")
    print("=" * 70)

    config = AugmentationConfig(rotation=True, rotation_range=30.0, random_seed=42)
    augmentor = PoseAugmentation(config)
    rotated = augmentor.augment(dummy_pose)

    print(f"元の鼻座標: {dummy_pose.normalized_keypoints[0]}")
    print(f"回転後の鼻座標: {rotated.normalized_keypoints[0]}")

    # テスト3: ノイズ
    print("\n" + "=" * 70)
    print("テスト3: ガウシアンノイズ")
    print("=" * 70)

    config = AugmentationConfig(add_noise=True, noise_std=0.05, random_seed=42)
    augmentor = PoseAugmentation(config)
    noisy = augmentor.augment(dummy_pose)

    print(f"元の鼻座標: {dummy_pose.normalized_keypoints[0]}")
    print(f"ノイズ付加後の鼻座標: {noisy.normalized_keypoints[0]}")
    diff = np.linalg.norm(noisy.normalized_keypoints[0] - dummy_pose.normalized_keypoints[0])
    print(f"差分ノルム: {diff:.4f}")

    # テスト4: ドロップアウト
    print("\n" + "=" * 70)
    print("テスト4: 関節ドロップアウト")
    print("=" * 70)

    config = AugmentationConfig(keypoint_dropout=True, dropout_prob=0.3, random_seed=42)
    augmentor = PoseAugmentation(config)
    dropout = augmentor.augment(dummy_pose)

    dropped_count = np.sum(dropout.keypoint_confidences == 0)
    print(f"ドロップアウトされた関節数: {dropped_count} / 17")

    # テスト5: 時系列データ
    print("\n" + "=" * 70)
    print("テスト5: 時系列データの拡張")
    print("=" * 70)

    # ダミーの時系列データを作成
    sequence = []
    for i in range(10):
        pose = NormalizedPoseData(
            track_id=1,
            frame=i,
            timestamp=i * 0.033,
            normalized_keypoints=np.random.randn(17, 2).astype(np.float32) * 0.5,
            keypoint_confidences=np.random.uniform(0.5, 1.0, 17).astype(np.float32),
            hip_center=(640.0, 480.0),
            scale_factor=200.0,
            confidence=0.9,
            is_valid=True
        )
        sequence.append(pose)

    print(f"元の系列長: {len(sequence)}")

    # 時間スケーリング
    config = AugmentationConfig(temporal_scaling=True, temporal_scale_range=(0.7, 0.7), random_seed=42)
    augmentor = PoseAugmentation(config)
    scaled_sequence = augmentor.augment_sequence(sequence)
    print(f"時間スケーリング後の系列長（0.7倍）: {len(scaled_sequence)}")

    # 時間ジッター
    config = AugmentationConfig(temporal_jitter=True, jitter_std=1.0, random_seed=42)
    augmentor = PoseAugmentation(config)
    jittered_sequence = augmentor.augment_sequence(sequence)
    print(f"時間ジッター後の系列長: {len(jittered_sequence)}")

    print("\n" + "=" * 70)
    print("すべてのテストが完了しました")
    print("=" * 70)


if __name__ == "__main__":
    main()
