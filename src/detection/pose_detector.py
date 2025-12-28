"""
MediaPipeを使った姿勢検出モジュール
卓球のサービス姿勢を検出する
"""
import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass


@dataclass
class PoseLandmarks:
    """姿勢のランドマーク情報を保持するクラス"""
    # 主要なランドマーク
    nose: Tuple[float, float, float]  # 鼻
    left_shoulder: Tuple[float, float, float]  # 左肩
    right_shoulder: Tuple[float, float, float]  # 右肩
    left_elbow: Tuple[float, float, float]  # 左肘
    right_elbow: Tuple[float, float, float]  # 右肘
    left_wrist: Tuple[float, float, float]  # 左手首
    right_wrist: Tuple[float, float, float]  # 右手首
    left_hip: Tuple[float, float, float]  # 左腰
    right_hip: Tuple[float, float, float]  # 右腰

    # 検出信頼度
    visibility: float


class PoseDetector:
    """MediaPipeを使った姿勢検出クラス"""

    def __init__(self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        """
        Args:
            min_detection_confidence: 検出の最小信頼度
            min_tracking_confidence: トラッキングの最小信頼度
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=1  # 0: Lite, 1: Full, 2: Heavy
        )

    def detect(self, frame: np.ndarray) -> Tuple[bool, Optional[PoseLandmarks], Optional[any]]:
        """
        フレームから姿勢を検出

        Args:
            frame: 入力フレーム（BGR形式）

        Returns:
            (検出成功, ランドマーク情報, 生のMediaPipe結果)
        """
        # BGRからRGBに変換
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 姿勢検出
        results = self.pose.process(frame_rgb)

        if not results.pose_landmarks:
            return False, None, None

        # ランドマーク情報を取得
        landmarks = results.pose_landmarks.landmark

        # 主要なランドマークを抽出
        try:
            pose_data = PoseLandmarks(
                nose=(landmarks[self.mp_pose.PoseLandmark.NOSE].x,
                      landmarks[self.mp_pose.PoseLandmark.NOSE].y,
                      landmarks[self.mp_pose.PoseLandmark.NOSE].z),
                left_shoulder=(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                              landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y,
                              landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].z),
                right_shoulder=(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                               landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y,
                               landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].z),
                left_elbow=(landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW].x,
                           landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW].y,
                           landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW].z),
                right_elbow=(landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                            landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW].y,
                            landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW].z),
                left_wrist=(landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST].x,
                           landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST].y,
                           landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST].z),
                right_wrist=(landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST].x,
                            landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST].y,
                            landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST].z),
                left_hip=(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].x,
                         landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].y,
                         landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].z),
                right_hip=(landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].x,
                          landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].y,
                          landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].z),
                visibility=landmarks[self.mp_pose.PoseLandmark.NOSE].visibility
            )

            return True, pose_data, results
        except Exception as e:
            print(f"ランドマーク抽出エラー: {e}")
            return False, None, None

    def draw_landmarks(self, frame: np.ndarray, results) -> np.ndarray:
        """
        フレームに姿勢のランドマークを描画

        Args:
            frame: 入力フレーム
            results: MediaPipeの検出結果

        Returns:
            ランドマークが描画されたフレーム
        """
        if results and results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        return frame

    def close(self):
        """リソースを解放"""
        self.pose.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
