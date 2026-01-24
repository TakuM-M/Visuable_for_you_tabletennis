"""
映像から獲得した全人間の骨格データと卓球台の位置情報をもとに、
トラッキングすべきプレイヤーを選定するコンポーネント
"""

from typing import List, Dict
import numpy as np

from src.detection.data_classes import PersonTrack, TableInfo, PlayerCandidate


# パラメータ定数
NEAR_TABLE_THRESHOLD = 0.3  # 正規化距離0.3以下で「卓球台に近い」と判定（厳qしめに調整）


class PlayerClassifier:
    """
    プレイヤー分類クラス

    複数フレームにわたって検出された人物トラッキング情報から、
    実際のプレイヤーを特定します。

    選定基準:
    1. tracking継続時間が長い
    2. 総運動量が多い
    3. 卓球台の近くにいることが多い
    """

    def __init__(
        self,
        near_table_threshold: float = NEAR_TABLE_THRESHOLD,
        min_tracking_frames: int = 10,
        max_players: int = 2
    ):
        """
        PlayerClassifierの初期化

        Args:
            near_table_threshold: 卓球台との正規化距離の閾値
            min_tracking_frames: プレイヤー候補とみなす最小フレーム数
            max_players: 検出する最大プレイヤー数
        """
        self.near_table_threshold = near_table_threshold
        self.min_tracking_frames = min_tracking_frames
        self.max_players = max_players

        # プレイヤー候補の情報を蓄積
        self.candidates: Dict[int, PlayerCandidate] = {}

    def update(
        self,
        persons: List[PersonTrack],
        table_info: TableInfo,
        frame_idx: int
    ) -> None:
        """
        フレームごとに人物トラッキング情報を更新

        Args:
            persons: 検出された人物のリスト
            table_info: 卓球台情報
            frame_idx: 現在のフレーム番号
        """
        current_track_ids = set()

        for person in persons:
            track_id = person.track_id
            current_track_ids.add(track_id)

            # 卓球台との距離を計算
            distance = self._calculate_normalized_distance(person, table_info)
            is_near = distance < self.near_table_threshold

            # 運動量を計算（前フレームとの比較）
            movement = 0.0
            if track_id in self.candidates:
                prev_keypoints = self.candidates[track_id].keypoints_history[-1]
                movement = self._calculate_movement(prev_keypoints, person.keypoints)

            # 候補が存在しない場合は新規作成
            if track_id not in self.candidates:
                self.candidates[track_id] = PlayerCandidate(
                    track_id=track_id,
                    first_seen_frame=frame_idx,
                    last_seen_frame=frame_idx,
                    positions=["near" if is_near else "far"],
                    keypoints_history=[person.keypoints.copy()],
                    total_movement=movement,
                    near_table_count=1 if is_near else 0,
                    total_frames=1
                )
            else:
                # 既存の候補を更新
                candidate = self.candidates[track_id]
                candidate.last_seen_frame = frame_idx
                candidate.positions.append("near" if is_near else "far")
                candidate.keypoints_history.append(person.keypoints.copy())
                candidate.total_movement += movement
                candidate.near_table_count += 1 if is_near else 0
                candidate.total_frames += 1

    def classify_players(self) -> List[int]:
        """
        蓄積された情報からプレイヤーのtracking IDを決定

        選定基準（優先順位順）:
        1. tracking継続時間が長い（tracking_duration_frames）
        2. 総運動量が多い（total_movement）
        3. 卓球台の近くにいることが多い（near_table_ratio）

        Returns:
            選定されたプレイヤーのtracking IDリスト
        """
        # 最小フレーム数を満たす候補をフィルタリング
        valid_candidates = [
            c for c in self.candidates.values()
            if c.total_frames >= self.min_tracking_frames
        ]

        if not valid_candidates:
            return []

        # スコアリング
        scored_candidates = []
        for candidate in valid_candidates:
            score = self._calculate_player_score(candidate)
            scored_candidates.append((candidate.track_id, score))

        # スコア順にソート（降順）
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        # 上位max_players人を選定
        selected_ids = [
            track_id for track_id, _ in scored_candidates[:self.max_players]
        ]

        return selected_ids

    def _calculate_player_score(self, candidate: PlayerCandidate) -> float:
        """
        プレイヤー候補のスコアを計算

        Args:
            candidate: プレイヤー候補

        Returns:
            スコア（高いほど優先）
        """
        # 正規化用の基準値を計算
        max_frames = max(c.total_frames for c in self.candidates.values())
        max_movement = max(c.total_movement for c in self.candidates.values())

        # 各要素を0-1に正規化してスコア計算
        # 重み調整: 卓球台付近の重みを大幅に上げる
        # - tracking継続時間: 10% (長く映っていることは重要だが決定的ではない)
        # - 総運動量: 30% (動きが激しいことは重要)
        # - 卓球台付近時間: 60% (最も重要: プレイヤーは卓球台の近くにいる)
        tracking_score = candidate.total_frames / max_frames if max_frames > 0 else 0
        movement_score = candidate.total_movement / max_movement if max_movement > 0 else 0
        near_table_score = candidate.near_table_ratio

        score = (
            0.1 * tracking_score +
            0.3 * movement_score +
            0.6 * near_table_score
        )

        return score

    def _calculate_normalized_distance(
        self,
        person: PersonTrack,
        table_info: TableInfo
    ) -> float:
        """
        人物と卓球台の正規化距離を計算

        卓球台の対角線長で正規化することで、画角に依存しない
        相対距離を計算します。

        改善点:
        1. Y座標制約: プレイヤーは卓球台より下側にいるべき
        2. 体の中心位置も考慮: 足元だけでなく体の中心（腰）も使用
        3. プレイエリアチェック: 卓球台の周辺エリア外は大きなペナルティ

        Args:
            person: 人物トラッキング情報
            table_info: 卓球台情報

        Returns:
            正規化距離（0.0 = 卓球台に接触、1.0 = 対角線長分離れている）
            プレイエリア外の場合は大きな値（10.0）を返す
        """
        # 卓球台のバウンディングボックス
        table_x1, table_y1, table_x2, table_y2 = table_info.bbox
        table_width = table_x2 - table_x1
        table_height = table_y2 - table_y1

        # 人物の足元位置を取得（バウンディングボックスの下端中心）
        person_foot_x = (person.bbox[0] + person.bbox[2]) / 2
        person_foot_y = person.bbox[3]  # 下端

        # 体の中心Y座標を取得（腰の位置）
        person_body_y = person.get_body_center_y()

        # ===== 改善1: Y座標制約 =====
        # プレイヤーは卓球台より下側（画面下側、Y座標が大きい）にいるべき
        # 審判や背景の人物は卓球台より上側にいることが多い
        if person_foot_y < table_y1:
            # 足元が卓球台の上端より上にある = 審判や背景の可能性が高い
            return 10.0  # 大きなペナルティ

        # ===== 改善2: プレイエリア定義 =====
        # 卓球台を中心とした拡張エリアを定義
        # X方向: 卓球台の左右に幅の1.5倍まで
        # Y方向: 卓球台の下に高さの3倍まで
        play_area_x1 = table_x1 - table_width * 1.5
        play_area_x2 = table_x2 + table_width * 1.5
        play_area_y1 = table_y1  # 上端は卓球台の上端
        play_area_y2 = table_y2 + table_height * 3.0  # 下端は卓球台の下に3倍

        # プレイエリア外の場合は大きなペナルティ
        if not (play_area_x1 <= person_foot_x <= play_area_x2 and
                play_area_y1 <= person_foot_y <= play_area_y2):
            return 10.0  # プレイエリア外

        # ===== 改善3: 体の中心位置も考慮した距離計算 =====
        # 足元と体の中心の両方から距離を計算し、より小さい方を採用

        # 足元からの距離
        dx_foot = max(table_x1 - person_foot_x, 0, person_foot_x - table_x2)
        dy_foot = max(table_y1 - person_foot_y, 0, person_foot_y - table_y2)
        distance_foot = np.sqrt(dx_foot**2 + dy_foot**2)

        # 体の中心からの距離（Y座標のみ）
        dy_body = max(table_y1 - person_body_y, 0, person_body_y - table_y2)

        # 両方を考慮した距離（足元を重視）
        distance = 0.7 * distance_foot + 0.3 * dy_body

        # 卓球台の対角線長で正規化
        table_diagonal = np.sqrt(
            (table_x2 - table_x1)**2 + (table_y2 - table_y1)**2
        )

        if table_diagonal > 0:
            normalized_distance = distance / table_diagonal
        else:
            normalized_distance = float('inf')

        return normalized_distance

    def _calculate_movement(
        self,
        prev_keypoints: np.ndarray,
        curr_keypoints: np.ndarray
    ) -> float:
        """
        キーポイント間の移動量を計算

        Args:
            prev_keypoints: 前フレームのキーポイント (17, 3)
            curr_keypoints: 現フレームのキーポイント (17, 3)

        Returns:
            移動量（ピクセル単位の平均移動距離）
        """
        # 信頼度が高いキーポイントのみを使用
        valid_mask = (prev_keypoints[:, 2] > 0.5) & (curr_keypoints[:, 2] > 0.5)

        if not np.any(valid_mask):
            return 0.0

        # 有効なキーポイントの移動距離を計算
        prev_points = prev_keypoints[valid_mask, :2]
        curr_points = curr_keypoints[valid_mask, :2]

        distances = np.linalg.norm(curr_points - prev_points, axis=1)
        movement = float(np.mean(distances))

        return movement

    def reset(self) -> None:
        """
        候補情報をリセット
        """
        self.candidates.clear()

    def get_candidate_info(self, track_id: int) -> PlayerCandidate:
        """
        特定のtracking IDの候補情報を取得

        Args:
            track_id: トラッキングID

        Returns:
            プレイヤー候補情報
        """
        return self.candidates.get(track_id)

    def get_all_candidates(self) -> Dict[int, PlayerCandidate]:
        """
        全候補情報を取得

        Returns:
            全プレイヤー候補の辞書
        """
        return self.candidates.copy()
