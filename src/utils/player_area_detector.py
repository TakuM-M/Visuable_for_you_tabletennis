"""
卓球台の座標情報からプレイヤーのプレー領域を検出するモジュール

卓球台の四隅の座標を基に、プレイヤーが立つ可能性の高い領域を
バウンディングボックスとして推定します。
"""

import cv2
import numpy as np
from typing import Tuple, List, Dict, Optional


class PlayerAreaDetector:
    """卓球台の座標からプレイヤー領域を検出するクラス"""

    def __init__(
        self,
        table_corners: List[Tuple[float, float]],
        margin_factor: float = 0.8,
        side_extension_factor: float = 0.3
    ):
        """
        プレイヤー領域検出器を初期化

        Parameters
        ----------
        table_corners : List[Tuple[float, float]]
            卓球台の四隅の座標 [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
            順番: 左上、右上、右下、左下（反時計回り）
        margin_factor : float, optional
            卓球台の短辺の長さに対する前後マージンの倍率
            デフォルトは0.8（卓球台の短辺の80%の距離）
        side_extension_factor : float, optional
            卓球台の長辺の長さに対する左右マージンの倍率
            デフォルトは0.3（卓球台の長辺の30%の距離）
        """
        if len(table_corners) != 4:
            raise ValueError("table_corners must contain exactly 4 points")

        self.table_corners = np.array(table_corners, dtype=np.float32)
        self.margin_factor = margin_factor
        self.side_extension_factor = side_extension_factor

        # 卓球台の各辺の長さを計算
        self._calculate_table_dimensions()

    def _calculate_table_dimensions(self):
        """卓球台のサイズを計算"""
        # 上辺の長さ（左上→右上）
        top_length = np.linalg.norm(
            self.table_corners[1] - self.table_corners[0]
        )
        # 下辺の長さ（左下→右下）
        bottom_length = np.linalg.norm(
            self.table_corners[2] - self.table_corners[3]
        )
        # 左辺の長さ（左上→左下）
        left_length = np.linalg.norm(
            self.table_corners[3] - self.table_corners[0]
        )
        # 右辺の長さ（右上→右下）
        right_length = np.linalg.norm(
            self.table_corners[2] - self.table_corners[1]
        )

        # 長辺と短辺の平均を取得
        self.table_width = (top_length + bottom_length) / 2  # 長辺（横幅）
        self.table_depth = (left_length + right_length) / 2  # 短辺（奥行き）

    def get_player_area_bbox(
        self,
        player_side: str = "near",
        return_format: str = "xyxy"
    ) -> Tuple[float, float, float, float]:
        """
        プレイヤー領域のバウンディングボックスを取得

        Parameters
        ----------
        player_side : str, optional
            "near" (手前側) または "far" (奥側)
        return_format : str, optional
            "xyxy" (x1, y1, x2, y2) または "xywh" (x, y, width, height)

        Returns
        -------
        Tuple[float, float, float, float]
            バウンディングボックスの座標
        """
        if player_side == "near":
            # 手前側（下側）の領域
            # 下辺を基準にプレイヤー領域を計算
            bbox = self._calculate_near_player_bbox()
        elif player_side == "far":
            # 奥側（上側）の領域
            # 上辺を基準にプレイヤー領域を計算
            bbox = self._calculate_far_player_bbox()
        else:
            raise ValueError("player_side must be 'near' or 'far'")

        if return_format == "xywh":
            # xyxy -> xywh に変換
            x1, y1, x2, y2 = bbox
            return (x1, y1, x2 - x1, y2 - y1)

        return bbox

    def _calculate_near_player_bbox(self) -> Tuple[float, float, float, float]:
        """
        手前側のプレイヤー領域を計算

        Returns
        -------
        Tuple[float, float, float, float]
            (x1, y1, x2, y2) 形式のバウンディングボックス
        """
        # 左下と右下の座標
        bottom_left = self.table_corners[3]
        bottom_right = self.table_corners[2]

        # 下辺の中心
        bottom_center = (bottom_left + bottom_right) / 2

        # 下辺の方向ベクトル
        bottom_vector = bottom_right - bottom_left
        bottom_vector_normalized = bottom_vector / np.linalg.norm(bottom_vector)

        # 下辺に垂直な方向（プレイヤーが立つ方向）
        perpendicular_vector = np.array([
            -bottom_vector_normalized[1],
            bottom_vector_normalized[0]
        ])

        # プレイヤー領域の奥行き（卓球台の短辺に基づく）
        player_depth = self.table_depth * self.margin_factor

        # プレイヤー領域の幅（卓球台の幅 + 左右マージン）
        side_margin = self.table_width * self.side_extension_factor
        player_width = self.table_width + 2 * side_margin

        # バウンディングボックスの4つの角を計算
        # 卓球台の下辺から外側に向かってプレイヤー領域を拡張
        half_width = player_width / 2
        width_vector = bottom_vector_normalized * half_width

        # 下辺の延長線上の左右端点
        left_table_edge = bottom_center - width_vector
        right_table_edge = bottom_center + width_vector

        # プレイヤー領域の外側の端点（卓球台から離れた位置）
        left_player_edge = left_table_edge + perpendicular_vector * player_depth
        right_player_edge = right_table_edge + perpendicular_vector * player_depth

        # 4つの角の座標
        corners = np.array([
            bottom_left,
            bottom_right,
            right_player_edge,
            left_player_edge
        ])

        # バウンディングボックスの計算（最小・最大値）
        x_min = np.min(corners[:, 0])
        y_min = np.min(corners[:, 1])
        x_max = np.max(corners[:, 0])
        y_max = np.max(corners[:, 1])

        return (x_min, y_min, x_max, y_max)

    def _calculate_far_player_bbox(self) -> Tuple[float, float, float, float]:
        """
        奥側のプレイヤー領域を計算

        Returns
        -------
        Tuple[float, float, float, float]
            (x1, y1, x2, y2) 形式のバウンディングボックス
        """
        # 左上と右上の座標
        top_left = self.table_corners[0]
        top_right = self.table_corners[1]

        # 上辺の中心
        top_center = (top_left + top_right) / 2

        # 上辺の方向ベクトル
        top_vector = top_right - top_left
        top_vector_normalized = top_vector / np.linalg.norm(top_vector)

        # 上辺に垂直な方向（プレイヤーが立つ方向、下向き）
        perpendicular_vector = np.array([
            top_vector_normalized[1],
            -top_vector_normalized[0]
        ])

        # プレイヤー領域の奥行き
        player_depth = self.table_depth * self.margin_factor

        # プレイヤー領域の幅
        side_margin = self.table_width * self.side_extension_factor
        player_width = self.table_width + 2 * side_margin

        # バウンディングボックスの4つの角を計算
        half_width = player_width / 2
        width_vector = top_vector_normalized * half_width

        # 上辺の延長線上の左右端点
        left_table_edge = top_center - width_vector
        right_table_edge = top_center + width_vector

        # プレイヤー領域の外側の端点
        left_player_edge = left_table_edge + perpendicular_vector * player_depth
        right_player_edge = right_table_edge + perpendicular_vector * player_depth

        # 4つの角の座標
        corners = np.array([
            top_left,
            top_right,
            right_player_edge,
            left_player_edge
        ])

        # バウンディングボックスの計算
        x_min = np.min(corners[:, 0])
        y_min = np.min(corners[:, 1])
        x_max = np.max(corners[:, 0])
        y_max = np.max(corners[:, 1])

        return (x_min, y_min, x_max, y_max)

    def get_both_player_areas(
        self,
        return_format: str = "xyxy"
    ) -> Dict[str, Tuple[float, float, float, float]]:
        """
        両側のプレイヤー領域を取得

        Parameters
        ----------
        return_format : str, optional
            "xyxy" (x1, y1, x2, y2) または "xywh" (x, y, width, height)

        Returns
        -------
        Dict[str, Tuple[float, float, float, float]]
            {"near": (x1, y1, x2, y2), "far": (x1, y1, x2, y2)}
        """
        return {
            "near": self.get_player_area_bbox("near", return_format),
            "far": self.get_player_area_bbox("far", return_format)
        }

    def visualize_player_areas(
        self,
        image: np.ndarray,
        show_both: bool = True,
        near_color: Tuple[int, int, int] = (0, 255, 0),
        far_color: Tuple[int, int, int] = (255, 0, 0),
        alpha: float = 0.3
    ) -> np.ndarray:
        """
        プレイヤー領域を画像上に可視化

        Parameters
        ----------
        image : np.ndarray
            入力画像
        show_both : bool, optional
            両側のプレイヤー領域を表示するかどうか
        near_color : Tuple[int, int, int], optional
            手前側の領域の色 (BGR)
        far_color : Tuple[int, int, int], optional
            奥側の領域の色 (BGR)
        alpha : float, optional
            領域の透明度 (0.0 - 1.0)

        Returns
        -------
        np.ndarray
            プレイヤー領域を描画した画像
        """
        output = image.copy()
        overlay = image.copy()

        # 卓球台の輪郭を描画
        table_pts = self.table_corners.astype(np.int32)
        cv2.polylines(output, [table_pts], True, (0, 128, 0), 3)

        # プレイヤー領域を描画
        areas = self.get_both_player_areas("xyxy")

        if show_both:
            sides = [("near", near_color), ("far", far_color)]
        else:
            sides = [("near", near_color)]

        for side, color in sides:
            x1, y1, x2, y2 = areas[side]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # 領域を半透明で塗りつぶし
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            # 枠線を描画
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)

            # ラベルを追加
            label = f"Player Area ({side})"
            label_y = y1 - 10 if side == "far" else y2 + 30
            cv2.putText(
                output,
                label,
                (x1 + 10, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

        # 半透明の領域を合成
        output = cv2.addWeighted(output, 1 - alpha, overlay, alpha, 0)

        return output

    def is_point_in_player_area(
        self,
        point: Tuple[float, float],
        player_side: str = "near"
    ) -> bool:
        """
        指定した点がプレイヤー領域内にあるかチェック

        Parameters
        ----------
        point : Tuple[float, float]
            チェックする点の座標 (x, y)
        player_side : str, optional
            "near" (手前側) または "far" (奥側)

        Returns
        -------
        bool
            プレイヤー領域内にある場合True
        """
        bbox = self.get_player_area_bbox(player_side, "xyxy")
        x1, y1, x2, y2 = bbox
        x, y = point

        return x1 <= x <= x2 and y1 <= y <= y2
