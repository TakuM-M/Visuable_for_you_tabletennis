"""
複数人物の検出と選別モジュール
YOLOv8またはMediaPipe Holistic/Poseで複数人物を検出し、
対象の選手（手前のプレイヤー）を特定する
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PersonRegion:
    """検出された人物の領域情報"""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    center_y: float  # 画面上のY座標（画面下側ほど手前）
    area: float  # 領域の面積
    confidence: float  # 検出信頼度


class MultiPersonDetector:
    """複数人物を検出し、対象選手を特定するクラス"""

    def __init__(self, roi_bottom_ratio: float = 0.7):
        """
        Args:
            roi_bottom_ratio: 画面下部の注目領域の割合（0.0-1.0）
                             0.7なら画面下部70%の領域を対象とする
        """
        self.roi_bottom_ratio = roi_bottom_ratio
        self.target_person_history = []  # 過去のターゲット位置を記録
        self.max_history = 10

    def detect_people_regions(self, frame: np.ndarray) -> List[PersonRegion]:
        """
        フレーム内の人物領域を検出（背景差分ベース）

        Args:
            frame: 入力フレーム

        Returns:
            検出された人物領域のリスト
        """
        height, width = frame.shape[:2]

        # グレースケール変換
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # エッジ検出
        edges = cv2.Canny(gray, 50, 150)

        # 膨張処理で輪郭を強調
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)

        # 輪郭検出
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 人物と思われる領域を抽出
        person_regions = []
        min_area = (height * width) * 0.01  # 画面の1%以上
        max_area = (height * width) * 0.5   # 画面の50%以下

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < min_area or area > max_area:
                continue

            # バウンディングボックスを取得
            x, y, w, h = cv2.boundingRect(contour)

            # アスペクト比チェック（人物は縦長）
            aspect_ratio = h / w if w > 0 else 0
            if aspect_ratio < 0.5 or aspect_ratio > 4:
                continue

            center_y = y + h / 2
            confidence = min(area / max_area, 1.0)

            person_regions.append(PersonRegion(
                bbox=(x, y, x + w, y + h),
                center_y=center_y,
                area=area,
                confidence=confidence
            ))

        return person_regions

    def select_target_person(
        self,
        frame: np.ndarray,
        person_regions: Optional[List[PersonRegion]] = None
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        対象選手（手前のプレイヤー）を選択

        優先順位:
        1. 画面下部に位置する（手前のプレイヤー）
        2. 画面中央に近い
        3. 面積が大きい（カメラに近い）

        Args:
            frame: 入力フレーム
            person_regions: 人物領域のリスト（Noneの場合は自動検出）

        Returns:
            対象選手のバウンディングボックス (x1, y1, x2, y2) または None
        """
        height, width = frame.shape[:2]

        if person_regions is None:
            person_regions = self.detect_people_regions(frame)

        if not person_regions:
            return None

        # ROI（関心領域）を設定：画面下部
        roi_top = int(height * (1 - self.roi_bottom_ratio))

        # スコアリング
        scored_regions = []
        for region in person_regions:
            score = 0.0

            # 1. 画面下部スコア（最重要）
            center_y = region.center_y
            if center_y > roi_top:
                # ROI内の場合、下にあるほど高スコア
                y_score = (center_y - roi_top) / (height - roi_top)
                score += y_score * 3.0  # 重み3倍

            # 2. 画面中央スコア
            x1, y1, x2, y2 = region.bbox
            center_x = (x1 + x2) / 2
            x_distance = abs(center_x - width / 2) / (width / 2)
            x_score = 1.0 - x_distance
            score += x_score * 1.0

            # 3. 面積スコア（大きいほど手前）
            area_score = region.area / (height * width)
            score += area_score * 2.0  # 重み2倍

            # 4. 信頼度
            score += region.confidence * 0.5

            scored_regions.append((region, score))

        # スコアが最も高い領域を選択
        if scored_regions:
            best_region, best_score = max(scored_regions, key=lambda x: x[1])

            # 履歴に追加
            self.target_person_history.append(best_region.bbox)
            if len(self.target_person_history) > self.max_history:
                self.target_person_history.pop(0)

            return best_region.bbox

        return None

    def create_roi_mask(self, frame: np.ndarray, bbox: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """
        対象選手のROIマスクを作成

        Args:
            frame: 入力フレーム
            bbox: バウンディングボックス (x1, y1, x2, y2)

        Returns:
            ROIマスク（対象領域が255、それ以外が0）
        """
        height, width = frame.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)

        if bbox is None:
            # バウンディングボックスが指定されていない場合、画面下部全体
            roi_top = int(height * (1 - self.roi_bottom_ratio))
            mask[roi_top:, :] = 255
        else:
            x1, y1, x2, y2 = bbox
            # 余裕を持たせて拡張（20%）
            margin_x = int((x2 - x1) * 0.2)
            margin_y = int((y2 - y1) * 0.2)

            x1 = max(0, x1 - margin_x)
            y1 = max(0, y1 - margin_y)
            x2 = min(width, x2 + margin_x)
            y2 = min(height, y2 + margin_y)

            mask[y1:y2, x1:x2] = 255

        return mask

    def apply_roi_to_frame(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        フレームにROIマスクを適用

        Args:
            frame: 入力フレーム
            mask: ROIマスク

        Returns:
            マスク適用後のフレーム
        """
        # マスク外を黒くする
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        return masked_frame

    def draw_roi_and_target(
        self,
        frame: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        person_regions: Optional[List[PersonRegion]] = None
    ) -> np.ndarray:
        """
        フレームにROIとターゲット選手を描画

        Args:
            frame: 入力フレーム
            bbox: ターゲット選手のバウンディングボックス
            person_regions: すべての人物領域

        Returns:
            描画後のフレーム
        """
        output = frame.copy()
        height, width = frame.shape[:2]

        # ROI領域を描画（画面下部）
        roi_top = int(height * (1 - self.roi_bottom_ratio))
        cv2.line(output, (0, roi_top), (width, roi_top), (0, 255, 255), 2)
        cv2.putText(output, "ROI (Target Area)", (10, roi_top - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # すべての検出領域を描画（灰色）
        if person_regions:
            for region in person_regions:
                x1, y1, x2, y2 = region.bbox
                cv2.rectangle(output, (x1, y1), (x2, y2), (128, 128, 128), 2)

        # ターゲット選手を描画（緑）
        if bbox:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(output, "TARGET", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return output
