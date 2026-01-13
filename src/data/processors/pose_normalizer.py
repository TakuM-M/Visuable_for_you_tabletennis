"""
骨格データの正規化処理モジュール

腰の中心を原点として、体幹の長さでスケールを正規化する
"""
import numpy as np
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.detection.yolo_tracker import PersonTrack, KEYPOINT_NAMES


@dataclass
class NormalizedPoseData:
    """正規化された骨格データ"""
    track_id: int
    frame: int
    timestamp: float

    # 正規化されたキーポイント（腰中心を原点、体幹長で正規化）
    normalized_keypoints: np.ndarray  # (17, 2) [normalized_x, normalized_y]
    keypoint_confidences: np.ndarray  # (17,) 各キーポイントの信頼度

    # 正規化に使用したパラメータ（復元用）
    hip_center: Tuple[float, float]  # 腰の中心座標（元画像上）
    scale_factor: float  # 正規化に使用したスケール（体幹の長さ）

    # メタデータ
    confidence: float  # 全体の検出信頼度
    is_valid: bool  # 正規化が成功したか

    def to_dict(self) -> Dict:
        """辞書形式に変換（CSV出力用）"""
        data = {
            'frame': self.frame,
            'timestamp': self.timestamp,
            'track_id': self.track_id,
            'hip_center_x': self.hip_center[0],
            'hip_center_y': self.hip_center[1],
            'scale_factor': self.scale_factor,
            'confidence': self.confidence,
            'is_valid': self.is_valid
        }

        # 正規化されたキーポイント座標
        for i, keypoint_name in enumerate(KEYPOINT_NAMES):
            data[f'{keypoint_name}_norm_x'] = self.normalized_keypoints[i, 0]
            data[f'{keypoint_name}_norm_y'] = self.normalized_keypoints[i, 1]
            data[f'{keypoint_name}_conf'] = self.keypoint_confidences[i]

        return data


class PoseNormalizer:
    """
    骨格データを正規化するクラス

    正規化の方法:
    1. 腰の中心（左腰と右腰の中点）を原点とする
    2. 体幹の長さ（腰中心から肩中心までの距離）でスケールを正規化
    3. 各キーポイントを相対座標に変換
    """

    def __init__(
        self,
        min_confidence: float = 0.3,
        reference_torso_length: float = 1.0
    ):
        """
        初期化

        Args:
            min_confidence: キーポイントの最小信頼度（これ以下は無効）
            reference_torso_length: 基準体幹長（正規化後のスケール）
        """
        self.min_confidence = min_confidence
        self.reference_torso_length = reference_torso_length

        # キーポイントのインデックス
        self.LEFT_HIP_IDX = KEYPOINT_NAMES.index("left_hip")
        self.RIGHT_HIP_IDX = KEYPOINT_NAMES.index("right_hip")
        self.LEFT_SHOULDER_IDX = KEYPOINT_NAMES.index("left_shoulder")
        self.RIGHT_SHOULDER_IDX = KEYPOINT_NAMES.index("right_shoulder")

    def normalize(
        self,
        person: PersonTrack,
        frame_num: int,
        timestamp: float
    ) -> NormalizedPoseData:
        """
        PersonTrackを正規化する

        Args:
            person: トラッキングされた人物データ
            frame_num: フレーム番号
            timestamp: タイムスタンプ

        Returns:
            正規化された骨格データ
        """
        keypoints = person.keypoints  # (17, 3) [x, y, confidence]

        # 腰の中心を計算
        hip_center, hip_valid = self._compute_hip_center(keypoints)

        # 体幹の長さを計算
        torso_length, torso_valid = self._compute_torso_length(keypoints, hip_center)

        # 正規化が有効かチェック
        is_valid = hip_valid and torso_valid and torso_length > 0

        if not is_valid:
            # 正規化できない場合は元の座標をそのまま使用
            normalized_kps = keypoints[:, :2].copy()
            scale_factor = 1.0
            if not hip_valid:
                hip_center = (0.0, 0.0)
        else:
            # スケールファクターを計算
            scale_factor = torso_length / self.reference_torso_length

            # 各キーポイントを正規化
            normalized_kps = np.zeros((17, 2), dtype=np.float32)
            for i in range(17):
                kp_x, kp_y, conf = keypoints[i]

                if conf >= self.min_confidence:
                    # 腰中心を原点とした相対座標
                    relative_x = kp_x - hip_center[0]
                    relative_y = kp_y - hip_center[1]

                    # スケール正規化
                    normalized_kps[i, 0] = relative_x / scale_factor
                    normalized_kps[i, 1] = relative_y / scale_factor
                else:
                    # 信頼度が低い場合は0に設定
                    normalized_kps[i, 0] = 0.0
                    normalized_kps[i, 1] = 0.0

        return NormalizedPoseData(
            track_id=person.track_id,
            frame=frame_num,
            timestamp=timestamp,
            normalized_keypoints=normalized_kps,
            keypoint_confidences=keypoints[:, 2].copy(),
            hip_center=hip_center,
            scale_factor=scale_factor,
            confidence=person.confidence,
            is_valid=is_valid
        )

    def _compute_hip_center(
        self,
        keypoints: np.ndarray
    ) -> Tuple[Tuple[float, float], bool]:
        """
        腰の中心座標を計算

        Args:
            keypoints: キーポイント配列 (17, 3)

        Returns:
            ((center_x, center_y), is_valid)
        """
        left_hip = keypoints[self.LEFT_HIP_IDX]
        right_hip = keypoints[self.RIGHT_HIP_IDX]

        # 両方の腰が検出されているかチェック
        if (left_hip[2] >= self.min_confidence and
            right_hip[2] >= self.min_confidence):
            center_x = (left_hip[0] + right_hip[0]) / 2.0
            center_y = (left_hip[1] + right_hip[1]) / 2.0
            return ((center_x, center_y), True)

        # 片方だけ検出されている場合はその座標を使用
        elif left_hip[2] >= self.min_confidence:
            return ((left_hip[0], left_hip[1]), True)
        elif right_hip[2] >= self.min_confidence:
            return ((right_hip[0], right_hip[1]), True)

        # 両方とも検出されていない
        return ((0.0, 0.0), False)

    def _compute_torso_length(
        self,
        keypoints: np.ndarray,
        hip_center: Tuple[float, float]
    ) -> Tuple[float, bool]:
        """
        体幹の長さを計算（腰中心から肩中心までの距離）

        Args:
            keypoints: キーポイント配列 (17, 3)
            hip_center: 腰の中心座標

        Returns:
            (torso_length, is_valid)
        """
        left_shoulder = keypoints[self.LEFT_SHOULDER_IDX]
        right_shoulder = keypoints[self.RIGHT_SHOULDER_IDX]

        # 肩の中心を計算
        shoulder_center = None
        if (left_shoulder[2] >= self.min_confidence and
            right_shoulder[2] >= self.min_confidence):
            shoulder_center = (
                (left_shoulder[0] + right_shoulder[0]) / 2.0,
                (left_shoulder[1] + right_shoulder[1]) / 2.0
            )
        elif left_shoulder[2] >= self.min_confidence:
            shoulder_center = (left_shoulder[0], left_shoulder[1])
        elif right_shoulder[2] >= self.min_confidence:
            shoulder_center = (right_shoulder[0], right_shoulder[1])

        if shoulder_center is None:
            return (0.0, False)

        # 腰中心から肩中心までの距離
        dx = shoulder_center[0] - hip_center[0]
        dy = shoulder_center[1] - hip_center[1]
        torso_length = np.sqrt(dx * dx + dy * dy)

        # 極端に小さい値は無効とする
        if torso_length < 10.0:  # 10ピクセル以下は異常値
            return (torso_length, False)

        return (torso_length, True)

    def denormalize(
        self,
        normalized_data: NormalizedPoseData
    ) -> np.ndarray:
        """
        正規化されたデータを元の座標系に戻す

        Args:
            normalized_data: 正規化された骨格データ

        Returns:
            元の座標系のキーポイント配列 (17, 3) [x, y, confidence]
        """
        keypoints = np.zeros((17, 3), dtype=np.float32)

        for i in range(17):
            norm_x = normalized_data.normalized_keypoints[i, 0]
            norm_y = normalized_data.normalized_keypoints[i, 1]
            conf = normalized_data.keypoint_confidences[i]

            # スケールを戻す
            original_x = norm_x * normalized_data.scale_factor + normalized_data.hip_center[0]
            original_y = norm_y * normalized_data.scale_factor + normalized_data.hip_center[1]

            keypoints[i] = [original_x, original_y, conf]

        return keypoints

    @staticmethod
    def get_csv_fieldnames() -> list[str]:
        """CSV出力用のフィールド名を取得"""
        fieldnames = [
            'frame', 'timestamp', 'track_id',
            'hip_center_x', 'hip_center_y', 'scale_factor',
            'confidence', 'is_valid'
        ]

        for keypoint_name in KEYPOINT_NAMES:
            fieldnames.extend([
                f'{keypoint_name}_norm_x',
                f'{keypoint_name}_norm_y',
                f'{keypoint_name}_conf'
            ])

        return fieldnames


def main():
    """テスト用のメイン関数"""
    print("PoseNormalizer モジュール")
    print("=" * 60)

    # ダミーデータでテスト
    dummy_keypoints = np.array([
        [640, 300, 0.9],  # nose
        [630, 290, 0.85], # left_eye
        [650, 290, 0.85], # right_eye
        [620, 300, 0.8],  # left_ear
        [660, 300, 0.8],  # right_ear
        [600, 400, 0.9],  # left_shoulder
        [680, 400, 0.9],  # right_shoulder
        [580, 500, 0.85], # left_elbow
        [700, 500, 0.85], # right_elbow
        [560, 600, 0.8],  # left_wrist
        [720, 600, 0.8],  # right_wrist
        [610, 600, 0.9],  # left_hip
        [670, 600, 0.9],  # right_hip
        [600, 800, 0.85], # left_knee
        [680, 800, 0.85], # right_knee
        [590, 1000, 0.8], # left_ankle
        [690, 1000, 0.8], # right_ankle
    ], dtype=np.float32)

    from src.detection.yolo_tracker import PersonTrack

    dummy_person = PersonTrack(
        track_id=1,
        bbox=(550, 250, 750, 1050),
        keypoints=dummy_keypoints,
        confidence=0.9
    )

    # 正規化処理
    normalizer = PoseNormalizer()
    normalized = normalizer.normalize(dummy_person, frame_num=1, timestamp=0.033)

    print("\n正規化結果:")
    print(f"  腰の中心: {normalized.hip_center}")
    print(f"  スケールファクター: {normalized.scale_factor:.2f}px")
    print(f"  正規化成功: {normalized.is_valid}")

    print("\n正規化されたキーポイント（サンプル）:")
    for i, name in enumerate(KEYPOINT_NAMES[:5]):
        x, y = normalized.normalized_keypoints[i]
        conf = normalized.keypoint_confidences[i]
        print(f"  {name:20s}: ({x:7.3f}, {y:7.3f}), conf={conf:.2f}")

    # 復元テスト
    restored = normalizer.denormalize(normalized)
    print("\n復元テスト（元のキーポイントとの差）:")
    max_diff = 0.0
    for i in range(17):
        diff_x = abs(restored[i, 0] - dummy_keypoints[i, 0])
        diff_y = abs(restored[i, 1] - dummy_keypoints[i, 1])
        max_diff = max(max_diff, diff_x, diff_y)
    print(f"  最大誤差: {max_diff:.6f}px (許容範囲内)")

    print("\n" + "=" * 60)
    print("テスト完了")


if __name__ == "__main__":
    main()
