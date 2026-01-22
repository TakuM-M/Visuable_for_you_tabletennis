"""
プレイヤー-卓球台関係分析コンポーネント

プレイヤーと卓球台の位置関係（前後、左右、距離）を分析する
"""

from typing import Tuple

import numpy as np

from ..detection.data_classes import CameraAngle, TableInfo
from .data_classes import PlayerTableRelation


class PlayerTableAnalyzer:
    """
    プレイヤー-卓球台関係分析コンポーネント

    責務:
    - プレイヤーの位置（near/far または left/right）を判定する
    - 画角に応じた適応的な位置判定を行う
    - 卓球台からの正規化距離を計算する
    - プレイヤーがプレイエリア内にいるかを判定する
    """

    def __init__(self, margin_ratio: float = 0.4):
        """
        Args:
            margin_ratio: プレイエリアの余白比率（卓球台サイズに対する比率）
        """
        self.margin_ratio = margin_ratio

    def analyze(
        self,
        player_bbox: Tuple[float, float, float, float],
        track_id: int,
        table_info: TableInfo
    ) -> PlayerTableRelation:
        """
        プレイヤーと卓球台の関係を分析（画角考慮）

        Args:
            player_bbox: プレイヤーのバウンディングボックス (x1, y1, x2, y2)
            track_id: プレイヤーのtracking ID
            table_info: TableInfo（卓球台情報）

        Returns:
            PlayerTableRelation
        """
        person_center = (
            (player_bbox[0] + player_bbox[2]) / 2,
            (player_bbox[1] + player_bbox[3]) / 2
        )

        # 画角に応じた位置判定
        position = self._get_position(person_center, table_info)
        side = self._get_side(person_center, table_info)

        # 距離計算
        distance = self._calculate_distance(player_bbox, table_info)

        # エリア内判定
        in_play_area = self._is_in_play_area(person_center, table_info)

        return PlayerTableRelation(
            track_id=track_id,
            position=position,
            side=side,
            distance_normalized=distance,
            is_in_play_area=in_play_area,
            camera_angle=table_info.camera_angle
        )

    def _get_position(
        self,
        person_center: Tuple[float, float],
        table_info: TableInfo
    ) -> str:
        """
        画角に応じた前後判定

        Args:
            person_center: プレイヤー中心座標 (x, y)
            table_info: TableInfo

        Returns:
            "near", "far", "left", "right"
        """
        if table_info.camera_angle == CameraAngle.SIDELINE:
            # サイドライン: 左右で判定
            if person_center[0] < table_info.center[0]:
                return "left"
            else:
                return "right"

        else:
            # エンドライン / 斜め上: 前後で判定
            boundary_y = table_info.get_near_far_boundary()

            if person_center[1] > boundary_y:
                return "near"
            else:
                return "far"

    def _get_side(
        self,
        person_center: Tuple[float, float],
        table_info: TableInfo
    ) -> str | None:
        """
        左右判定（エンドライン画角で有効）

        Args:
            person_center: プレイヤー中心座標 (x, y)
            table_info: TableInfo

        Returns:
            "left", "right", or None
        """
        if table_info.camera_angle == CameraAngle.SIDELINE:
            # サイドライン: 左右 = 前後の意味
            return None

        # エンドライン / 斜め上
        if person_center[0] < table_info.center[0]:
            return "left"
        else:
            return "right"

    def _is_in_play_area(
        self,
        person_center: Tuple[float, float],
        table_info: TableInfo
    ) -> bool:
        """
        プレイエリア内判定

        Args:
            person_center: プレイヤー中心座標 (x, y)
            table_info: TableInfo

        Returns:
            プレイエリア内ならTrue
        """
        near_area = table_info.get_near_area(self.margin_ratio)
        far_area = table_info.get_far_area(self.margin_ratio)

        def in_area(area: dict) -> bool:
            return (
                area['x1'] <= person_center[0] <= area['x2'] and
                area['y1'] <= person_center[1] <= area['y2']
            )

        return in_area(near_area) or in_area(far_area)

    def _calculate_distance(
        self,
        player_bbox: Tuple[float, float, float, float],
        table_info: TableInfo
    ) -> float:
        """
        卓球台からの正規化距離

        Args:
            player_bbox: プレイヤーのバウンディングボックス (x1, y1, x2, y2)
            table_info: TableInfo

        Returns:
            正規化距離（0.0 = 卓球台に接触、1.0 = 対角線長分離れている）
        """
        person_center = (
            (player_bbox[0] + player_bbox[2]) / 2,
            (player_bbox[1] + player_bbox[3]) / 2
        )

        # 卓球台バウンディングボックスからの最短距離
        dx = max(table_info.bbox[0] - person_center[0], 0, person_center[0] - table_info.bbox[2])
        dy = max(table_info.bbox[1] - person_center[1], 0, person_center[1] - table_info.bbox[3])

        distance = np.sqrt(dx**2 + dy**2)
        diagonal = np.sqrt(table_info.width**2 + table_info.height**2)

        return distance / diagonal if diagonal > 0 else 0.0
