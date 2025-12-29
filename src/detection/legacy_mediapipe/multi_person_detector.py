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
    center_x: float  # 画面上のX座標（水平位置）
    center_y: float  # 画面上のY座標（垂直位置）
    area: float  # 領域の面積
    confidence: float  # 検出信頼度


class MultiPersonDetector:
    """複数人物を検出し、対象選手を特定するクラス（卓球特化版）"""

    def __init__(
        self,
        player_left_ratio: float = 0.0,
        player_right_ratio: float = 0.4,
        table_left_ratio: float = 0.35,
        table_right_ratio: float = 0.65,
        player_vertical_top: float = 0.3,
        player_vertical_bottom: float = 0.85
    ):
        """
        卓球の画面レイアウトを前提とした初期化（水平方向ベース）

        画面構成（横方向）：
        ┌────────┬────────┬────────┐
        │ 手前   │ 卓球台 │ 相手   │
        │プレイヤー│（中央）│選手    │
        │ (左)   │        │ (右)   │
        │ ★検出  │        │        │
        └────────┴────────┴────────┘
         0-40%    35-65%   60-100%
         ↑手前選手の検出領域

        縦方向（上下）：
        - player_vertical_top (30%): 手前選手検出の上端
        - player_vertical_bottom (85%): 手前選手検出の下端

        Args:
            player_left_ratio: 手前選手領域の左端（画面左端からの割合）
            player_right_ratio: 手前選手領域の右端（画面左端からの割合）
            table_left_ratio: 卓球台領域の左端
            table_right_ratio: 卓球台領域の右端
            player_vertical_top: 手前選手検出の上端（縦方向）
            player_vertical_bottom: 手前選手検出の下端（縦方向）
        """
        self.player_left_ratio = player_left_ratio
        self.player_right_ratio = player_right_ratio
        self.table_left_ratio = table_left_ratio
        self.table_right_ratio = table_right_ratio
        self.player_vertical_top = player_vertical_top
        self.player_vertical_bottom = player_vertical_bottom
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

            center_x = x + w / 2
            center_y = y + h / 2
            confidence = min(area / max_area, 1.0)

            person_regions.append(PersonRegion(
                bbox=(x, y, x + w, y + h),
                center_x=center_x,
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

        卓球の画面レイアウトを考慮（水平方向ベース）：
        1. 手前選手エリア（画面左側 0-40%）の領域のみ対象
        2. 縦方向でも範囲を絞る（30-85%）
        3. 卓球台エリアは除外

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

        # ROI（関心領域）を設定：手前選手エリア（画面左側）
        player_left = int(width * self.player_left_ratio)
        player_right = int(width * self.player_right_ratio)
        player_top = int(height * self.player_vertical_top)
        player_bottom = int(height * self.player_vertical_bottom)

        # スコアリング
        scored_regions = []
        for region in person_regions:
            score = 0.0
            center_x = region.center_x
            center_y = region.center_y

            # 1. 手前選手エリア（画面左側）にいるかチェック（最重要）
            in_horizontal_range = player_left <= center_x <= player_right
            in_vertical_range = player_top <= center_y <= player_bottom

            if in_horizontal_range and in_vertical_range:
                # 手前選手エリア内の場合、高スコア
                # 水平方向：左寄りほど高スコア
                x_score = 1.0 - ((center_x - player_left) / (player_right - player_left))
                score += x_score * 5.0  # 重み5倍

                # 垂直方向：中央に近いほど高スコア
                vertical_center = (player_top + player_bottom) / 2
                y_distance = abs(center_y - vertical_center) / ((player_bottom - player_top) / 2)
                y_score = 1.0 - y_distance
                score += y_score * 3.0

                # 3. 面積スコア（大きいほど手前）
                area_score = region.area / (height * width)
                score += area_score * 2.0

                # 4. 信頼度
                score += region.confidence * 0.5
            else:
                # 手前選手エリア外は大幅減点
                score = -10.0

            scored_regions.append((region, score))

        # スコアが最も高い領域を選択
        if scored_regions:
            best_region, best_score = max(scored_regions, key=lambda x: x[1])

            # スコアが正の場合のみ有効な検出とみなす
            if best_score > 0:
                # 履歴に追加
                self.target_person_history.append(best_region.bbox)
                if len(self.target_person_history) > self.max_history:
                    self.target_person_history.pop(0)

                return best_region.bbox

        return None

    def create_roi_mask(self, frame: np.ndarray, bbox: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """
        対象選手のROIマスクを作成（卓球レイアウト対応・水平方向ベース）

        Args:
            frame: 入力フレーム
            bbox: バウンディングボックス (x1, y1, x2, y2)

        Returns:
            ROIマスク（対象領域が255、それ以外が0）
        """
        height, width = frame.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)

        if bbox is None:
            # バウンディングボックスが指定されていない場合、手前選手エリア全体
            player_left = int(width * self.player_left_ratio)
            player_right = int(width * self.player_right_ratio)
            player_top = int(height * self.player_vertical_top)
            player_bottom = int(height * self.player_vertical_bottom)
            mask[player_top:player_bottom, player_left:player_right] = 255
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
        フレームにROIとターゲット選手を描画（卓球レイアウト対応・水平方向ベース）

        Args:
            frame: 入力フレーム
            bbox: ターゲット選手のバウンディングボックス
            person_regions: すべての人物領域

        Returns:
            描画後のフレーム
        """
        output = frame.copy()
        height, width = frame.shape[:2]

        # 手前選手エリアの境界を描画
        player_left = int(width * self.player_left_ratio)
        player_right = int(width * self.player_right_ratio)
        player_top = int(height * self.player_vertical_top)
        player_bottom = int(height * self.player_vertical_bottom)

        # 卓球台エリアの境界を描画
        table_left = int(width * self.table_left_ratio)
        table_right = int(width * self.table_right_ratio)

        # 手前選手エリア（ROI）を矩形で描画（黄色）
        cv2.rectangle(output, (player_left, player_top), (player_right, player_bottom), (0, 255, 255), 3)
        cv2.putText(output, "Player Area (ROI)", (player_left + 10, player_top + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 卓球台エリアを矩形で描画（青）
        cv2.rectangle(output, (table_left, 0), (table_right, height), (255, 0, 0), 2)
        cv2.putText(output, "Table Area", (table_left + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # すべての検出領域を描画（灰色）
        if person_regions:
            for region in person_regions:
                x1, y1, x2, y2 = region.bbox
                cv2.rectangle(output, (x1, y1), (x2, y2), (128, 128, 128), 2)

        # ターゲット選手を描画（緑）
        if bbox:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(output, "TARGET PLAYER", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return output
