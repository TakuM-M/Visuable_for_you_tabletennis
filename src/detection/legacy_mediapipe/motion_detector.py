"""
動き検出モジュール
フレーム間の差分から動きを検出する
"""
import cv2
import numpy as np
from typing import Optional, List, Tuple


class MotionDetector:
    """動き検出クラス"""

    def __init__(self, threshold: int = 25, min_area: int = 500):
        """
        Args:
            threshold: 動き検出の閾値
            min_area: 動きと判定する最小面積
        """
        self.threshold = threshold
        self.min_area = min_area
        self.prev_frame: Optional[np.ndarray] = None

    def detect(self, frame: np.ndarray) -> Tuple[bool, float]:
        """フレーム内の動きを検出

        Args:
            frame: 入力フレーム

        Returns:
            (動きが検出されたか, 動きの強さ)
        """
        # グレースケール変換
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # 初回フレームの場合
        if self.prev_frame is None:
            self.prev_frame = gray
            return False, 0.0

        # フレーム差分
        frame_diff = cv2.absdiff(self.prev_frame, gray)
        thresh = cv2.threshold(frame_diff, self.threshold, 255, cv2.THRESH_BINARY)[1]

        # 膨張処理でノイズ除去
        thresh = cv2.dilate(thresh, None, iterations=2)

        # 輪郭検出
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 動きの強さを計算
        motion_strength = 0.0
        has_motion = False

        for contour in contours:
            if cv2.contourArea(contour) < self.min_area:
                continue
            has_motion = True
            motion_strength += cv2.contourArea(contour)

        # 正規化
        motion_strength = motion_strength / (frame.shape[0] * frame.shape[1])

        # 前フレームを更新
        self.prev_frame = gray

        return has_motion, motion_strength

    def reset(self):
        """検出器をリセット"""
        self.prev_frame = None
