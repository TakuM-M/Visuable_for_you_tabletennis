"""
バッチ処理版プレイヤー分類コンポーネント

動画全体を処理してから、全データを使ってプレイヤーを特定する。
リアルタイム処理ではなく、オフライン分析に適している。

処理フロー:
1. データ収集: 動画から全骨格データを収集
2. 分析: 収集したデータ全体を使ってプレイヤーを判定
3. 出力: プレイヤーIDのリストを返す
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
from dataclasses import dataclass, field

from src.detection.data_classes import PersonTrack, TableInfo, PlayerCandidate


# パラメータ定数
NEAR_TABLE_THRESHOLD = 0.3  # 正規化距離0.3以下で「卓球台に近い」と判定


@dataclass
class TrackingRecord:
    """トラッキングIDごとの記録"""
    frame_idx: int
    person: PersonTrack
    distance_to_table: float  # 卓球台からの正規化距離
    timestamp: float = 0.0  # オプション: タイムスタンプ


class PlayerClassifierBatch:
    """
    バッチ処理版プレイヤー分類クラス

    動画全体を処理してから、全データを使ってプレイヤーを特定します。

    利点:
    - 動画全体の情報を使えるため、より正確な判定が可能
    - IDが途中で変わっても、全体の傾向から判定できる
    - 複数回の分析やパラメータ調整が容易
    """

    def __init__(
        self,
        near_table_threshold: float = NEAR_TABLE_THRESHOLD,
        min_tracking_frames: int = 10,
        max_players: int = 2
    ):
        """
        初期化

        Args:
            near_table_threshold: 卓球台との正規化距離の閾値
            min_tracking_frames: プレイヤー候補とみなす最小フレーム数
            max_players: 検出する最大プレイヤー数
        """
        self.near_table_threshold = near_table_threshold
        self.min_tracking_frames = min_tracking_frames
        self.max_players = max_players

        # データ収集用
        self.table_info: Optional[TableInfo] = None
        self.tracking_data: Dict[int, List[TrackingRecord]] = {}
        self.total_frames = 0

    def set_table_info(self, table_info: TableInfo) -> None:
        """
        卓球台情報をセット

        Args:
            table_info: 卓球台情報
        """
        self.table_info = table_info

    def add_frame_data(
        self,
        frame_idx: int,
        persons: List[PersonTrack],
        timestamp: float = 0.0
    ) -> None:
        """
        フレームごとのデータを追加

        Args:
            frame_idx: フレーム番号
            persons: 検出された人物のリスト
            timestamp: タイムスタンプ（オプション）
        """
        if self.table_info is None:
            # 卓球台情報がない場合は距離計算できないので警告
            print(f"警告: フレーム {frame_idx} - 卓球台情報が設定されていません")
            return

        self.total_frames = max(self.total_frames, frame_idx + 1)

        for person in persons:
            track_id = person.track_id

            # 卓球台との距離を計算
            distance = self._calculate_normalized_distance(person, self.table_info)

            # 記録を作成
            record = TrackingRecord(
                frame_idx=frame_idx,
                person=person,
                distance_to_table=distance,
                timestamp=timestamp
            )

            # tracking_dataに追加
            if track_id not in self.tracking_data:
                self.tracking_data[track_id] = []
            self.tracking_data[track_id].append(record)

    def analyze_and_classify(self) -> List[int]:
        """
        収集したデータを分析してプレイヤーを分類

        Returns:
            プレイヤーのtracking IDリスト
        """
        if not self.tracking_data:
            print("警告: 分析するデータがありません")
            return []

        # 各tracking IDの候補情報を作成
        candidates = self._build_candidates()

        # 最小フレーム数を満たす候補をフィルタリング
        valid_candidates = [
            c for c in candidates.values()
            if c.total_frames >= self.min_tracking_frames
        ]

        if not valid_candidates:
            print(f"警告: 最小フレーム数({self.min_tracking_frames})を満たす候補がいません")
            return []

        # スコアリング
        scored_candidates = []
        for candidate in valid_candidates:
            score = self._calculate_player_score(candidate, candidates)
            scored_candidates.append((candidate.track_id, score))

        # スコア順にソート（降順）
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        # 上位max_players人を選定
        selected_ids = [
            track_id for track_id, _ in scored_candidates[:self.max_players]
        ]

        return selected_ids

    def get_analysis_summary(self) -> Dict:
        """
        分析結果のサマリーを取得

        Returns:
            分析サマリーの辞書
        """
        candidates = self._build_candidates()

        summary = {
            "total_frames": self.total_frames,
            "total_tracking_ids": len(self.tracking_data),
            "valid_candidates": sum(
                1 for c in candidates.values()
                if c.total_frames >= self.min_tracking_frames
            ),
            "candidates": []
        }

        for track_id, candidate in candidates.items():
            score = self._calculate_player_score(candidate, candidates)
            summary["candidates"].append({
                "track_id": track_id,
                "total_frames": candidate.total_frames,
                "total_movement": candidate.total_movement,
                "near_table_ratio": candidate.near_table_ratio,
                "score": score
            })

        # スコア順にソート
        summary["candidates"].sort(key=lambda x: x["score"], reverse=True)

        return summary

    def _build_candidates(self) -> Dict[int, PlayerCandidate]:
        """
        tracking_dataからPlayerCandidateを構築

        Returns:
            tracking IDをキーとしたPlayerCandidateの辞書
        """
        candidates = {}

        for track_id, records in self.tracking_data.items():
            if not records:
                continue

            # レコードをフレーム順にソート
            records.sort(key=lambda r: r.frame_idx)

            # 基本情報
            first_frame = records[0].frame_idx
            last_frame = records[-1].frame_idx
            total_frames = len(records)

            # 位置履歴
            positions = [
                "near" if r.distance_to_table < self.near_table_threshold else "far"
                for r in records
            ]
            near_table_count = positions.count("near")

            # キーポイント履歴
            keypoints_history = [r.person.keypoints.copy() for r in records]

            # 総運動量を計算
            total_movement = 0.0
            for i in range(1, len(keypoints_history)):
                movement = self._calculate_movement(
                    keypoints_history[i-1],
                    keypoints_history[i]
                )
                total_movement += movement

            # PlayerCandidateを作成
            candidate = PlayerCandidate(
                track_id=track_id,
                first_seen_frame=first_frame,
                last_seen_frame=last_frame,
                positions=positions,
                keypoints_history=keypoints_history,
                total_movement=total_movement,
                near_table_count=near_table_count,
                total_frames=total_frames
            )

            candidates[track_id] = candidate

        return candidates

    def _calculate_player_score(
        self,
        candidate: PlayerCandidate,
        all_candidates: Dict[int, PlayerCandidate]
    ) -> float:
        """
        プレイヤー候補のスコアを計算

        Args:
            candidate: プレイヤー候補
            all_candidates: 全候補の辞書（正規化用）

        Returns:
            スコア（高いほど優先）
        """
        # 正規化用の基準値を計算
        max_frames = max(c.total_frames for c in all_candidates.values())
        max_movement = max(c.total_movement for c in all_candidates.values())

        # 各要素を0-1に正規化してスコア計算
        # 重み: tracking継続時間 10%, 総運動量 30%, 卓球台付近時間 60%
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

        Args:
            person: 人物トラッキング情報
            table_info: 卓球台情報

        Returns:
            正規化距離
        """
        # 卓球台のバウンディングボックス
        table_x1, table_y1, table_x2, table_y2 = table_info.bbox
        table_width = table_x2 - table_x1
        table_height = table_y2 - table_y1

        # 人物の足元位置
        person_foot_x = (person.bbox[0] + person.bbox[2]) / 2
        person_foot_y = person.bbox[3]

        # 体の中心Y座標
        person_body_y = person.get_body_center_y()

        # Y座標制約: 卓球台より上にいる場合はペナルティ
        if person_foot_y < table_y1:
            return 10.0

        # プレイエリア定義
        play_area_x1 = table_x1 - table_width * 1.5
        play_area_x2 = table_x2 + table_width * 1.5
        play_area_y1 = table_y1
        play_area_y2 = table_y2 + table_height * 3.0

        # プレイエリア外の場合はペナルティ
        if not (play_area_x1 <= person_foot_x <= play_area_x2 and
                play_area_y1 <= person_foot_y <= play_area_y2):
            return 10.0

        # 距離計算
        dx_foot = max(table_x1 - person_foot_x, 0, person_foot_x - table_x2)
        dy_foot = max(table_y1 - person_foot_y, 0, person_foot_y - table_y2)
        distance_foot = np.sqrt(dx_foot**2 + dy_foot**2)

        dy_body = max(table_y1 - person_body_y, 0, person_body_y - table_y2)

        distance = 0.7 * distance_foot + 0.3 * dy_body

        # 正規化
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
        """データをリセット"""
        self.table_info = None
        self.tracking_data.clear()
        self.total_frames = 0
