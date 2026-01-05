"""
卓球台のホモグラフィ変換を行うモジュール

カメラ視点から見た卓球台の四隅の座標から、
卓球台を真上から見た視点（鳥瞰図）に変換するための機能を提供します。
"""

import cv2
import numpy as np
from typing import Tuple, List


class TableHomography:
    """卓球台のホモグラフィ変換を管理するクラス"""

    def __init__(
        self,
        table_corners: List[Tuple[float, float]],
        table_width_mm: float = 1525.0,  # 卓球台の公式幅 (mm)
        table_height_mm: float = 2740.0,  # 卓球台の公式長さ (mm)
        output_scale: float = 0.5  # 出力画像のスケール (1mm = 0.5px)
    ):
        """
        卓球台のホモグラフィ変換を初期化

        Parameters
        ----------
        table_corners : List[Tuple[float, float]]
            カメラ視点での卓球台の四隅の座標 [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
            順番: 左上、右上、右下、左下（反時計回り）
        table_width_mm : float, optional
            卓球台の実際の幅 (mm)、デフォルトは公式サイズ
        table_height_mm : float, optional
            卓球台の実際の長さ (mm)、デフォルトは公式サイズ
        output_scale : float, optional
            出力画像のスケール (1mm = output_scale px)
        """
        if len(table_corners) != 4:
            raise ValueError("table_corners must contain exactly 4 points")

        self.table_corners = np.array(table_corners, dtype=np.float32)
        self.table_width_mm = table_width_mm
        self.table_height_mm = table_height_mm
        self.output_scale = output_scale

        # 出力画像のサイズを計算
        self.output_width = int(table_width_mm * output_scale)
        self.output_height = int(table_height_mm * output_scale)

        # 変換先の座標（真上から見た卓球台の四隅）
        self.dst_corners = np.array([
            [0, 0],                                    # 左上
            [self.output_width, 0],                    # 右上
            [self.output_width, self.output_height],   # 右下
            [0, self.output_height]                    # 左下
        ], dtype=np.float32)

        # ホモグラフィ行列を計算
        self.homography_matrix = cv2.getPerspectiveTransform(
            self.table_corners,
            self.dst_corners
        )

        # 逆変換用の行列（鳥瞰図→カメラ視点）
        self.inv_homography_matrix = cv2.getPerspectiveTransform(
            self.dst_corners,
            self.table_corners
        )

    def transform_point(self, point: Tuple[float, float]) -> Tuple[float, float]:
        """
        カメラ視点の座標を鳥瞰図の座標に変換

        Parameters
        ----------
        point : Tuple[float, float]
            カメラ視点での座標 (x, y)

        Returns
        -------
        Tuple[float, float]
            鳥瞰図での座標 (x, y)
        """
        point_array = np.array([[point]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point_array, self.homography_matrix)
        return tuple(transformed[0][0])

    def transform_points(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        複数のカメラ視点の座標を鳥瞰図の座標に変換

        Parameters
        ----------
        points : List[Tuple[float, float]]
            カメラ視点での座標のリスト

        Returns
        -------
        List[Tuple[float, float]]
            鳥瞰図での座標のリスト
        """
        if not points:
            return []

        points_array = np.array([points], dtype=np.float32)
        transformed = cv2.perspectiveTransform(points_array, self.homography_matrix)
        return [tuple(p) for p in transformed[0]]

    def inverse_transform_point(self, point: Tuple[float, float]) -> Tuple[float, float]:
        """
        鳥瞰図の座標をカメラ視点の座標に変換

        Parameters
        ----------
        point : Tuple[float, float]
            鳥瞰図での座標 (x, y)

        Returns
        -------
        Tuple[float, float]
            カメラ視点での座標 (x, y)
        """
        point_array = np.array([[point]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point_array, self.inv_homography_matrix)
        return tuple(transformed[0][0])

    def warp_image(self, image: np.ndarray) -> np.ndarray:
        """
        カメラ視点の画像を鳥瞰図に変換

        Parameters
        ----------
        image : np.ndarray
            入力画像

        Returns
        -------
        np.ndarray
            鳥瞰図に変換された画像
        """
        warped = cv2.warpPerspective(
            image,
            self.homography_matrix,
            (self.output_width, self.output_height)
        )
        return warped

    def get_player_region_mask(
        self,
        player_side: str = "near",
        margin_mm: float = 500.0
    ) -> np.ndarray:
        """
        選手がいる領域のマスクを生成

        Parameters
        ----------
        player_side : str, optional
            "near" (手前側) または "far" (奥側)
        margin_mm : float, optional
            卓球台からの距離マージン (mm)

        Returns
        -------
        np.ndarray
            選手領域のマスク画像 (binary)
        """
        margin_px = int(margin_mm * self.output_scale)
        mask = np.zeros((self.output_height, self.output_width), dtype=np.uint8)

        if player_side == "near":
            # 手前側の領域（卓球台の下側）
            cv2.rectangle(
                mask,
                (0, self.output_height),
                (self.output_width, self.output_height + margin_px),
                255,
                -1
            )
        elif player_side == "far":
            # 奥側の領域（卓球台の上側）
            cv2.rectangle(
                mask,
                (0, -margin_px),
                (self.output_width, 0),
                255,
                -1
            )
        else:
            raise ValueError("player_side must be 'near' or 'far'")

        return mask

    def get_output_size(self) -> Tuple[int, int]:
        """
        出力画像のサイズを取得

        Returns
        -------
        Tuple[int, int]
            (width, height)
        """
        return (self.output_width, self.output_height)

    def visualize_transformation(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        元画像と変換後の画像を可視化用に生成

        Parameters
        ----------
        image : np.ndarray
            入力画像

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (元画像with四隅マーク, 鳥瞰図画像)
        """
        # 元画像に四隅をマーク
        img_marked = image.copy()
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]  # BGR
        labels = ["TL", "TR", "BR", "BL"]

        for i, (corner, color, label) in enumerate(zip(self.table_corners, colors, labels)):
            x, y = int(corner[0]), int(corner[1])
            cv2.circle(img_marked, (x, y), 10, color, -1)
            cv2.putText(
                img_marked,
                label,
                (x + 15, y + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2
            )

        # 鳥瞰図を生成
        warped = self.warp_image(image)

        return img_marked, warped
