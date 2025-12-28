"""
卓球のサービス姿勢を検出するモジュール
MediaPipeで検出した姿勢からサービスを判定する
"""
import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass
from .pose_detector import PoseLandmarks


@dataclass
class ServiceDetection:
    """サービス検出結果"""
    is_service_pose: bool  # サービス姿勢かどうか
    confidence: float  # 信頼度 0.0-1.0
    pose_type: str  # 姿勢のタイプ（"toss", "backswing", "impact", "follow_through", "none"）
    details: dict  # 詳細情報


class ServiceDetector:
    """サービス姿勢を検出するクラス"""

    def __init__(self):
        """初期化"""
        self.pose_history: List[ServiceDetection] = []
        self.max_history = 30  # 約1秒分（30fps想定）

    def detect_service_pose(self, pose: Optional[PoseLandmarks]) -> ServiceDetection:
        """
        姿勢データからサービス姿勢を判定

        Args:
            pose: 姿勢のランドマーク情報

        Returns:
            サービス検出結果
        """
        if pose is None:
            return ServiceDetection(
                is_service_pose=False,
                confidence=0.0,
                pose_type="none",
                details={"reason": "姿勢が検出されませんでした"}
            )

        # 各姿勢パターンをチェック
        toss_result = self._check_toss_pose(pose)
        backswing_result = self._check_backswing_pose(pose)
        impact_result = self._check_impact_pose(pose)

        # 最も信頼度の高い姿勢を選択
        results = [
            ("toss", toss_result),
            ("backswing", backswing_result),
            ("impact", impact_result)
        ]

        best_pose_type, best_result = max(results, key=lambda x: x[1]["confidence"])

        detection = ServiceDetection(
            is_service_pose=best_result["is_service"],
            confidence=best_result["confidence"],
            pose_type=best_pose_type if best_result["is_service"] else "none",
            details=best_result["details"]
        )

        # 履歴に追加
        self.pose_history.append(detection)
        if len(self.pose_history) > self.max_history:
            self.pose_history.pop(0)

        return detection

    def _check_toss_pose(self, pose: PoseLandmarks) -> dict:
        """
        トス姿勢をチェック
        条件：
        - 片方の手首が肩より上にある
        - もう片方の手が後ろに引かれている
        """
        details = {}
        confidence = 0.0
        is_service = False

        # 左手でトスする場合
        left_wrist_y = pose.left_wrist[1]
        left_shoulder_y = pose.left_shoulder[1]
        right_wrist_y = pose.right_wrist[1]
        right_shoulder_y = pose.right_shoulder[1]

        # Y座標は上が小さい値なので、手首が肩より上 = 手首のY < 肩のY
        left_hand_up = left_wrist_y < left_shoulder_y - 0.05  # 肩より5%上
        right_hand_up = right_wrist_y < right_shoulder_y - 0.05

        # 右手でトスする場合
        right_hand_back = pose.right_wrist[0] > pose.right_shoulder[0] + 0.1  # 肩より10%右
        left_hand_back = pose.left_wrist[0] < pose.left_shoulder[0] - 0.1  # 肩より10%左

        # パターン1: 左手トス、右手バックスイング
        if left_hand_up and right_hand_back:
            confidence = 0.7
            is_service = True
            details["pattern"] = "左手トス"
            details["left_hand_up"] = True
            details["right_hand_back"] = True

        # パターン2: 右手トス、左手バックスイング
        elif right_hand_up and left_hand_back:
            confidence = 0.7
            is_service = True
            details["pattern"] = "右手トス"
            details["right_hand_up"] = True
            details["left_hand_back"] = True

        details["left_wrist_y"] = left_wrist_y
        details["left_shoulder_y"] = left_shoulder_y
        details["right_wrist_y"] = right_wrist_y
        details["right_shoulder_y"] = right_shoulder_y

        return {
            "is_service": is_service,
            "confidence": confidence,
            "details": details
        }

    def _check_backswing_pose(self, pose: PoseLandmarks) -> dict:
        """
        バックスイング姿勢をチェック
        条件：
        - ラケットを持つ手（肘）が後ろに引かれている
        - 肘が肩より高い位置にある
        """
        details = {}
        confidence = 0.0
        is_service = False

        # 右手でラケットを持つと仮定
        right_elbow_y = pose.right_elbow[1]
        right_shoulder_y = pose.right_shoulder[1]
        right_elbow_x = pose.right_elbow[0]
        right_shoulder_x = pose.right_shoulder[0]

        # 肘が後ろに引かれている
        elbow_back = right_elbow_x > right_shoulder_x + 0.05

        # 肘が肩と同じくらいの高さ、または少し上
        elbow_up = abs(right_elbow_y - right_shoulder_y) < 0.1

        if elbow_back and elbow_up:
            confidence = 0.6
            is_service = True
            details["pattern"] = "バックスイング"
            details["elbow_back"] = True
            details["elbow_up"] = True

        details["right_elbow_y"] = right_elbow_y
        details["right_shoulder_y"] = right_shoulder_y
        details["right_elbow_x"] = right_elbow_x
        details["right_shoulder_x"] = right_shoulder_x

        return {
            "is_service": is_service,
            "confidence": confidence,
            "details": details
        }

    def _check_impact_pose(self, pose: PoseLandmarks) -> dict:
        """
        インパクト姿勢をチェック（打球の瞬間）
        条件：
        - ラケットを持つ手が前に伸びている
        - 手首が肩より高い位置にある
        """
        details = {}
        confidence = 0.0
        is_service = False

        # 右手でラケットを持つと仮定
        right_wrist_y = pose.right_wrist[1]
        right_shoulder_y = pose.right_shoulder[1]
        right_wrist_x = pose.right_wrist[0]
        right_shoulder_x = pose.right_shoulder[0]

        # 手首が前に伸びている
        wrist_forward = right_wrist_x < right_shoulder_x - 0.1

        # 手首が肩より上
        wrist_up = right_wrist_y < right_shoulder_y

        if wrist_forward and wrist_up:
            confidence = 0.5
            is_service = True
            details["pattern"] = "インパクト"
            details["wrist_forward"] = True
            details["wrist_up"] = True

        details["right_wrist_y"] = right_wrist_y
        details["right_shoulder_y"] = right_shoulder_y
        details["right_wrist_x"] = right_wrist_x
        details["right_shoulder_x"] = right_shoulder_x

        return {
            "is_service": is_service,
            "confidence": confidence,
            "details": details
        }

    def detect_service_sequence(self) -> Tuple[bool, float]:
        """
        履歴から連続したサービスシーケンスを検出

        Returns:
            (サービスが検出されたか, 信頼度)
        """
        if len(self.pose_history) < 10:
            return False, 0.0

        # 過去10フレームを確認
        recent_poses = self.pose_history[-10:]

        # トス → バックスイング → インパクト のシーケンスを探す
        toss_count = sum(1 for p in recent_poses if p.pose_type == "toss")
        backswing_count = sum(1 for p in recent_poses if p.pose_type == "backswing")
        impact_count = sum(1 for p in recent_poses if p.pose_type == "impact")

        # サービスシーケンスの条件
        # - トスとバックスイングが検出されている
        # - または、インパクトが検出されている
        if (toss_count >= 2 and backswing_count >= 2) or impact_count >= 3:
            avg_confidence = np.mean([p.confidence for p in recent_poses if p.is_service_pose])
            return True, avg_confidence

        return False, 0.0

    def reset_history(self):
        """履歴をリセット"""
        self.pose_history.clear()
