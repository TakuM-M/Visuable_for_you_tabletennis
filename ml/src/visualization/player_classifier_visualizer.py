import cv2

class PlayerClassifierVisualizer:
    """プレイヤー分類結果を可視化するクラス"""

    def __init__(self, table_detector, pose_tracker, player_classifier):
        self.table_detector = table_detector
        self.pose_tracker = pose_tracker
        self.player_classifier = player_classifier

    def draw_results(self, frame, table_info, persons, player_ids):
        """検出結果を描画"""
        output = frame.copy()

        # 卓球台を描画
        if table_info:
            x1, y1, x2, y2 = table_info.bbox
            cv2.rectangle(output, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            cv2.putText(output, "Table", (int(x1), int(y1) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            # 卓球台が検出できていない場合の警告表示
            cv2.putText(output, "WARNING: Table Not Detected", (20, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        # スケルトン接続定義
        skeleton_connections = [
            (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 6),
            (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16)
        ]

        # 人物を描画
        for person in persons:
            is_player = person.track_id in player_ids
            color = (0, 255, 0) if is_player else (0, 0, 255)
            label = "PLAYER" if is_player else "Other"

            # バウンディングボックス
            x1, y1, x2, y2 = person.bbox
            thickness = 3 if is_player else 2
            cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)

            # ラベル
            text = f"{label} ID:{person.track_id}"
            cv2.putText(output, text, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # スケルトン
            for connection in skeleton_connections:
                kp1_idx, kp2_idx = connection
                kp1 = person.keypoints[kp1_idx]
                kp2 = person.keypoints[kp2_idx]

                if kp1[2] > 0.5 and kp2[2] > 0.5:
                    pt1 = (int(kp1[0]), int(kp1[1]))
                    pt2 = (int(kp2[0]), int(kp2[1]))
                    cv2.line(output, pt1, pt2, color, 2)

            # キーポイント
            for kp in person.keypoints:
                if kp[2] > 0.5:
                    pt = (int(kp[0]), int(kp[1]))
                    cv2.circle(output, pt, 3, color, -1)

        return output

    def draw_candidate_info(self, frame, player_ids):
        """候補者情報を描画"""
        output = frame.copy()
        info_x = frame.shape[1] - 400
        info_y = 30
        line_height = 25

        cv2.putText(output, "=== Player Candidates ===", (info_x, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        y_offset = info_y + line_height

        # 候補情報を取得してスコア順にソート
        candidates = []
        for track_id, candidate in self.player_classifier.candidates.items():
            if candidate.total_frames >= self.player_classifier.min_tracking_frames:
                score = self.player_classifier._calculate_player_score(candidate)
                candidates.append((track_id, candidate, score))

        candidates.sort(key=lambda x: x[2], reverse=True)

        # 上位候補を表示
        for track_id, candidate, score in candidates[:5]:
            is_player = track_id in player_ids
            color = (0, 255, 0) if is_player else (200, 200, 200)

            text = f"ID:{track_id} {'[PLAYER]' if is_player else ''}"
            cv2.putText(output, text, (info_x, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += line_height

            text = f"  Score: {score:.3f}"
            cv2.putText(output, text, (info_x, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            y_offset += 18

            text = f"  Frames: {candidate.total_frames}, Move: {candidate.total_movement:.1f}"
            cv2.putText(output, text, (info_x, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            y_offset += 18

            text = f"  Near table: {candidate.near_table_ratio:.1%}"
            cv2.putText(output, text, (info_x, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            y_offset += line_height

        return output