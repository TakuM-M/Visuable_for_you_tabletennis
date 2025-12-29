"""
選手フィルタリングモジュール
トラッキング結果から手前選手・相手選手を特定し、審判・コーチを除外する
"""
import numpy as np
from typing import List, Optional, Dict
from dataclasses import dataclass
from collections import defaultdict

from .table_detector import TableRegion
from .yolo_tracker import PersonTrack


@dataclass
class PlayerClassification:
    """フレームごとの選手分類結果"""
    frame_num: int
    near_player: Optional[PersonTrack]  # 手前選手（カメラに近い）
    far_player: Optional[PersonTrack]   # 相手選手（カメラから遠い）
    others: List[PersonTrack]           # その他（審判・コーチ等）


class PlayerFilter:
    """選手フィルタリングクラス"""

    def __init__(
        self,
        table_region: TableRegion,
        near_margin: float = 50.0,
        far_margin: float = 50.0,
        min_appearance_ratio: float = 0.3
    ):
        """
        選手フィルタの初期化

        Args:
            table_region: 卓球台領域
            near_margin: 手前選手判定のマージン（ピクセル）
            far_margin: 相手選手判定のマージン（ピクセル）
            min_appearance_ratio: 選手として認定する最小出現率
        """
        self.table_region = table_region
        self.near_margin = near_margin
        self.far_margin = far_margin
        self.min_appearance_ratio = min_appearance_ratio

        # 卓球台の上端Y座標（選手分類の基準）
        self.table_y_threshold = table_region.get_y_threshold()

        # 履歴管理
        self.classification_history: List[PlayerClassification] = []
        self.track_positions: Dict[int, List[float]] = defaultdict(list)  # track_id -> [y座標のリスト]

    def classify_players(
        self,
        frame_num: int,
        persons: List[PersonTrack]
    ) -> PlayerClassification:
        """
        フレーム内の人物を分類（手前選手、相手選手、その他）

        Args:
            frame_num: フレーム番号
            persons: トラッキングされた人物リスト

        Returns:
            選手分類結果
        """
        near_player = None
        far_player = None
        others = []

        # 各人物の位置を記録
        for person in persons:
            body_center_y = person.get_body_center_y()
            self.track_positions[person.track_id].append(body_center_y)

        # 人物を位置で分類
        near_candidates = []  # 卓球台より下（手前）
        far_candidates = []   # 卓球台より上（相手）

        for person in persons:
            body_center_y = person.get_body_center_y()

            # 卓球台より十分下にいる場合 → 手前選手候補
            if body_center_y > self.table_y_threshold + self.near_margin:
                near_candidates.append(person)

            # 卓球台より十分上にいる場合 → 相手選手候補
            elif body_center_y < self.table_y_threshold - self.far_margin:
                far_candidates.append(person)

            # マージン範囲内 → その他（判定保留）
            else:
                others.append(person)

        # 手前選手を選択（最もカメラに近い = Y座標が大きい）
        if near_candidates:
            near_player = max(near_candidates, key=lambda p: p.get_body_center_y())

        # 相手選手を選択（最もカメラから遠い = Y座標が小さい）
        if far_candidates:
            far_player = min(far_candidates, key=lambda p: p.get_body_center_y())

        # 選ばれなかった人物はothersに追加
        for person in near_candidates:
            if near_player is None or person.track_id != near_player.track_id:
                others.append(person)

        for person in far_candidates:
            if far_player is None or person.track_id != far_player.track_id:
                others.append(person)

        classification = PlayerClassification(
            frame_num=frame_num,
            near_player=near_player,
            far_player=far_player,
            others=others
        )

        # 履歴に追加
        self.classification_history.append(classification)

        return classification

    def assign_player_roles(self) -> Dict[int, str]:
        """
        全フレーム処理後、各トラッキングIDに役割を割り当て

        Returns:
            {track_id: role} の辞書
            role: "near_player", "far_player", "other"
        """
        if not self.classification_history:
            return {}

        total_frames = len(self.classification_history)

        # 各トラッキングIDの出現回数と役割カウント
        track_role_counts: Dict[int, Dict[str, int]] = defaultdict(lambda: {
            "near_player": 0,
            "far_player": 0,
            "other": 0
        })

        for classification in self.classification_history:
            if classification.near_player:
                track_id = classification.near_player.track_id
                track_role_counts[track_id]["near_player"] += 1

            if classification.far_player:
                track_id = classification.far_player.track_id
                track_role_counts[track_id]["far_player"] += 1

            for person in classification.others:
                track_id = person.track_id
                track_role_counts[track_id]["other"] += 1

        # 各トラッキングIDに最も多い役割を割り当て
        player_roles = {}

        for track_id, role_counts in track_role_counts.items():
            # 総出現回数
            total_appearances = sum(role_counts.values())

            # 出現率が閾値以下の場合はスキップ
            appearance_ratio = total_appearances / total_frames
            if appearance_ratio < self.min_appearance_ratio:
                continue

            # 最多の役割を選択
            assigned_role = max(role_counts, key=role_counts.get)
            player_roles[track_id] = assigned_role

        return player_roles

    def get_player_tracks(
        self,
        player_roles: Dict[int, str]
    ) -> Dict[str, int]:
        """
        役割からトラッキングIDを取得

        Args:
            player_roles: assign_player_roles() の結果

        Returns:
            {"near_player": track_id, "far_player": track_id}
        """
        result = {}

        for track_id, role in player_roles.items():
            if role == "near_player":
                result["near_player"] = track_id
            elif role == "far_player":
                result["far_player"] = track_id

        return result

    def get_statistics(self, player_roles: Dict[int, str]) -> Dict:
        """
        フィルタリング統計情報を取得

        Args:
            player_roles: assign_player_roles() の結果

        Returns:
            統計情報の辞書
        """
        if not self.classification_history:
            return {}

        total_frames = len(self.classification_history)

        # 各役割の出現フレーム数
        near_player_frames = sum(
            1 for c in self.classification_history if c.near_player is not None
        )
        far_player_frames = sum(
            1 for c in self.classification_history if c.far_player is not None
        )

        # 各役割のトラッキングID
        near_player_ids = [
            track_id for track_id, role in player_roles.items()
            if role == "near_player"
        ]
        far_player_ids = [
            track_id for track_id, role in player_roles.items()
            if role == "far_player"
        ]
        other_ids = [
            track_id for track_id, role in player_roles.items()
            if role == "other"
        ]

        return {
            "total_frames": total_frames,
            "near_player_frames": near_player_frames,
            "far_player_frames": far_player_frames,
            "near_player_ratio": near_player_frames / total_frames if total_frames > 0 else 0,
            "far_player_ratio": far_player_frames / total_frames if total_frames > 0 else 0,
            "near_player_ids": near_player_ids,
            "far_player_ids": far_player_ids,
            "other_ids": other_ids,
            "total_tracks": len(player_roles)
        }

    def draw_classification(
        self,
        frame: np.ndarray,
        classification: PlayerClassification
    ) -> np.ndarray:
        """
        フレームに選手分類結果を描画

        Args:
            frame: 入力フレーム
            classification: 分類結果

        Returns:
            描画後のフレーム
        """
        import cv2
        output = frame.copy()

        # 卓球台の閾値ラインを描画
        height, width = frame.shape[:2]
        threshold_y = int(self.table_y_threshold)
        cv2.line(output, (0, threshold_y), (width, threshold_y), (255, 255, 0), 2)
        cv2.putText(output, "Table Threshold", (10, threshold_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # 手前選手を緑で描画
        if classification.near_player:
            person = classification.near_player
            x1, y1, x2, y2 = person.bbox
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 3)
            label = f"NEAR PLAYER (ID:{person.track_id})"
            cv2.putText(output, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 相手選手を青で描画
        if classification.far_player:
            person = classification.far_player
            x1, y1, x2, y2 = person.bbox
            cv2.rectangle(output, (x1, y1), (x2, y2), (255, 0, 0), 3)
            label = f"FAR PLAYER (ID:{person.track_id})"
            cv2.putText(output, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # その他を灰色で描画
        for person in classification.others:
            x1, y1, x2, y2 = person.bbox
            cv2.rectangle(output, (x1, y1), (x2, y2), (128, 128, 128), 2)
            label = f"OTHER (ID:{person.track_id})"
            cv2.putText(output, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 2)

        return output
