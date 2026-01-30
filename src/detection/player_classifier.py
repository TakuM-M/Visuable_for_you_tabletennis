"""
映像から獲得した人間骨格データと卓球台情報をもとにトラッキングすべきプレイヤーを選定するコンポーネント
"""

from typing import List, Dict
import numpy as np

from src.core.data_classes import PersonTrack, TableInfo, PlayerCandidate


# パラメータ定数
NEAR_TABLE_THRESHOLD = 0.1  # 正規化距離0.1以下で「卓球台に近い」と判定（厳しめに調整）
MOVEMENT_NOISE_THRESHOLD = 5.0  # 5px以下の動きはYOLOのブレ（ノイズ）として無視
RECENT_FRAMES_WINDOW = 146  # 直近60フレームの運動量を考慮（約2秒 @ 30fps）
MAX_CONSECUTIVE_OTHER_COUNT = 30  # other判定が30回連続したら候補をリセット（約1秒 @ 30fps）


class PlayerClassifier:
    """
    プレイヤー分類クラス
    
    人間情報と卓球台情報から選手であるトラッキングIDを判定

    選定基準:
    1. 総運動量が多い
    2. 卓球台の近くにいることが多い
    """

    def __init__(
        self,
        near_table_threshold: float = NEAR_TABLE_THRESHOLD,
        min_tracking_frames: int = 10,
        max_players: int = 2,
        max_inactive_frames: int = 30,
        min_player_score: float = 0.3,
        recent_frames_window: int = RECENT_FRAMES_WINDOW,
        max_consecutive_other_count: int = MAX_CONSECUTIVE_OTHER_COUNT
    ):
        """
        PlayerClassifierの初期化

        Args:
            near_table_threshold: 卓球台との正規化距離の閾値
            min_tracking_frames: プレイヤー候補とみなす最小フレーム数
            max_players: 検出する最大プレイヤー数
            max_inactive_frames: この期間見られていない候補を削除するフレーム数
            min_player_score: プレイヤーとして判定する最小スコア閾値（0.0-1.0）
                            この値以下のスコアの候補はプレイヤーから除外される
            recent_frames_window: 運動量計算に使用する直近フレーム数（デフォルト: 60フレーム）
            max_consecutive_other_count: other判定が連続でこの回数を超えたら候補をリセット
        """
        self.near_table_threshold = near_table_threshold
        self.min_tracking_frames = min_tracking_frames
        self.max_players = max_players
        self.max_inactive_frames = max_inactive_frames
        self.min_player_score = min_player_score
        self.recent_frames_window = recent_frames_window
        self.max_consecutive_other_count = max_consecutive_other_count

        # プレイヤー候補の情報を蓄積
        self.candidates: Dict[int, PlayerCandidate] = {}

        # 現在のフレーム番号を記録
        self.current_frame_idx: int = 0

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
        self.current_frame_idx = frame_idx

        current_track_ids = set()

        for person in persons:
            track_id = person.track_id
            current_track_ids.add(track_id)

            # 卓球台との距離を計算
            distance = self._calculate_table_distance(person, table_info)
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
                    total_frames=1,
                    movement_history=[movement]
                )
            else:
                candidate = self.candidates[track_id]
                candidate.last_seen_frame = frame_idx
                candidate.positions.append("near" if is_near else "far")
                candidate.keypoints_history.append(person.keypoints.copy())
                candidate.total_movement += movement
                candidate.near_table_count += 1 if is_near else 0
                candidate.total_frames += 1

                candidate.movement_history.append(movement)
                if len(candidate.movement_history) > self.recent_frames_window:
                    candidate.movement_history.pop(0)

        # 古い候補をクリーンアップ
        self._cleanup_old_candidates()

    def _update_other_count(self, selected_player_ids: List[int]) -> List[int]:
        """
        プレイヤーとして選定されなかった候補のother判定カウントを更新し、
        連続other判定が閾値を超えた候補を削除する

        削除された候補は、再び卓球台周辺に来ない限り候補として復活しない。
        これにより、審判などのプレイヤー以外の人物が候補として残り続けるのを防ぐ。

        Args:
            selected_player_ids: プレイヤーとして選定されたtracking IDのリスト

        Returns:
            削除されたtracking IDのリスト
        """
        remove_ids = []

        for track_id, candidate in self.candidates.items():
            if track_id in selected_player_ids:
                # プレイヤーとして選定された場合はカウントをリセット
                candidate.consecutive_other_count = 0
            else:
                # other判定された場合はカウントを増やす
                candidate.consecutive_other_count += 1

                # 連続other判定が閾値を超えた場合は削除対象に追加
                if candidate.consecutive_other_count >= self.max_consecutive_other_count:
                    remove_ids.append(track_id)

        # 削除対象の候補を完全に削除
        for track_id in remove_ids:
            del self.candidates[track_id]

        return remove_ids

    def _cleanup_old_candidates(self) -> None:
        """
        長期間見られていない候補を削除する

        トラッキングIDが変わった場合に古いIDをプレイヤーとして
        保持し続けないようにするための処理
        """
        inactive_ids = []
        for track_id, candidate in self.candidates.items():
            frames_since_last_seen = self.current_frame_idx - candidate.last_seen_frame
            if frames_since_last_seen > self.max_inactive_frames:
                inactive_ids.append(track_id)

        # 削除
        for track_id in inactive_ids:
            del self.candidates[track_id]

    def classify_players(self, max_inactive_frames_for_selection: int = 10) -> tuple[List[int], List[int]]:
        """
        蓄積された情報からプレイヤーのtracking IDを決定

        選定基準:
        1. 総運動量が多い（total_movement）
        2. 卓球台の近くにいることが多い（near_table_ratio）
        3. 最近のフレームで見られていること

        重要: 長期間プレイヤー候補として情報が蓄積されていない場合は削除
        　　  other判定が連続で一定回数を超えた候補は自動的にリセット

        Args:
            max_inactive_frames_for_selection: プレイヤー選定時に許容する最大の非アクティブフレーム数
                                               （デフォルト: 10フレーム以内に見られた候補のみ）

        Returns:
            タプル: (選定されたプレイヤーのtracking IDリスト, 削除されたtracking IDリスト)
        """
        # 最小フレーム数を満たし、かつ最近見られている候補をフィルタリング
        valid_candidates = [
            c for c in self.candidates.values()
            if (c.total_frames >= self.min_tracking_frames and
                self.current_frame_idx - c.last_seen_frame <= max_inactive_frames_for_selection)
        ]

        if not valid_candidates:
            return [], []
        
        scored_candidates = []
        for candidate in valid_candidates:
            score = self._calculate_player_score(candidate)
            scored_candidates.append((candidate.track_id, score))

        # スコア順にソート（降順）
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        # スコア閾値でフィルタリング & 上位max_players人を選定
        # 重要: スコアがmin_player_score以下の候補は除外
        # （例: プレイヤー + 審判の場合、審判は低スコアのため除外される）
        selected_ids = [
            track_id for track_id, score in scored_candidates[:self.max_players]
            if score >= self.min_player_score
        ]

        # プレイヤーとして選定されたIDと選定されなかったIDを追跡
        # 削除されたIDのリストを取得
        removed_ids = self._update_other_count(selected_ids)

        return selected_ids, removed_ids

    def _calculate_player_score(self, candidate: PlayerCandidate) -> float:
        """
        プレイヤー候補のスコアを計算
        1. 運動量スコア: 直近フレームでの平均的な動きの多さ（重み: 0.8）
        2. 卓球台近接率スコア: 卓球台の近くにいた割合（重み: 0.2）

        Args:
            candidate: プレイヤー候補

        Returns:
            スコア（高いほど優先、0.0-1.0）
        """
        # 1. 運動量スコアの計算（直近フレームの平均）
        avg_movements = []
        for c in self.candidates.values():
            if c.movement_history and len(c.movement_history) > 0:
                avg_movements.append(sum(c.movement_history) / len(c.movement_history))
            else:
                avg_movements.append(0.0)

        max_avg_movement = max(avg_movements) if avg_movements else 1.0

        # 候補の直近フレーム平均運動量でスコア計算
        if candidate.movement_history and len(candidate.movement_history) > 0:
            candidate_avg_movement = sum(candidate.movement_history) / len(candidate.movement_history)
        else:
            candidate_avg_movement = 0
        movement_score = candidate_avg_movement / max_avg_movement if max_avg_movement > 0 else 0

        # 2. 卓球台近接率スコアの計算
        near_table_ratio = candidate.near_table_count / candidate.total_frames if candidate.total_frames > 0 else 0

        # 全候補の中での最大近接率を取得（正規化用）
        max_near_ratio = max(
            c.near_table_count / c.total_frames if c.total_frames > 0 else 0
            for c in self.candidates.values()
        )

        # 近接率スコアを正規化（0.0-1.0）
        proximity_score = near_table_ratio / max_near_ratio if max_near_ratio > 0 else 0

        # 3. 重み付きスコアの計算
        # 運動量: 80%, 卓球台近接率: 20%
        score = 0.8 * movement_score + 0.2 * proximity_score

        return score

    def _calculate_table_distance(
        self,
        person: PersonTrack,
        table_info
    ) -> float:
        """
        人物と卓球台の正規化距離を計算

        Args:
            person: 人物トラッキング情報
            table_info: 卓球台情報（TableInfo）

        Returns:
            正規化距離（卓球台の対角線長で正規化）
        """
        table_x1, table_y1, table_x2, table_y2 = table_info.bbox

        person_center_x = (person.bbox[0] + person.bbox[2]) / 2
        person_center_y = (person.bbox[1] + person.bbox[3]) / 2

        dx = max(table_x1 - person_center_x, 0, person_center_x - table_x2)
        dy = max(table_y1 - person_center_y, 0, person_center_y - table_y2)
        distance = np.sqrt(dx**2 + dy**2)

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
        - 下半身（腰、膝、足首）: 75%の重み
        - 上半身（顔、肩、肘、手首）: 25%の重み
        判定を行う：
        審判のように上半身だけ動く人物 or プレイヤーのように全身を使って動く人物

        Args:
            prev_keypoints: 前フレームのキーポイント (17, 3)
            curr_keypoints: 現フレームのキーポイント (17, 3)

        Returns:
            移動量（ピクセル単位の重み付き平均移動距離）
        """
        # 下半身キーポイントのインデックス（腰、膝、足首）
        lower_body_indices = [11, 12, 13, 14, 15, 16]
        # 上半身キーポイントのインデックス（顔、肩、肘、手首）
        upper_body_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        lower_body_movement = 0.0
        lower_body_valid_count = 0
        for idx in lower_body_indices:
            if prev_keypoints[idx, 2] > 0.5 and curr_keypoints[idx, 2] > 0.5:
                distance = np.linalg.norm(curr_keypoints[idx, :2] - prev_keypoints[idx, :2])
                # ノイズ閾値未満の微小な動きは無視（YOLOのブレをフィルタリング）
                if distance >= MOVEMENT_NOISE_THRESHOLD:
                    lower_body_movement += distance
                    lower_body_valid_count += 1

        upper_body_movement = 0.0
        upper_body_valid_count = 0
        for idx in upper_body_indices:
            if prev_keypoints[idx, 2] > 0.5 and curr_keypoints[idx, 2] > 0.5:
                distance = np.linalg.norm(curr_keypoints[idx, :2] - prev_keypoints[idx, :2])
                # ノイズ閾値未満の微小な動きは無視（YOLOのブレをフィルタリング）
                if distance >= MOVEMENT_NOISE_THRESHOLD:
                    upper_body_movement += distance
                    upper_body_valid_count += 1

        if lower_body_valid_count == 0 and upper_body_valid_count == 0:
            return 0.0

        lower_avg = (lower_body_movement / lower_body_valid_count
                     if lower_body_valid_count > 0 else 0.0)
        upper_avg = (upper_body_movement / upper_body_valid_count
                     if upper_body_valid_count > 0 else 0.0)
        movement = 0.75 * lower_avg + 0.25 * upper_avg
        
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
