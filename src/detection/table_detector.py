"""
卓球台検出モジュール
動画内の卓球台位置を検出し、固定された座標を提供する
"""
import cv2
import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


@dataclass
class TableRegion:
    """卓球台の領域情報"""
    corners: np.ndarray  # 4隅の座標 (4, 2) - [top_left, top_right, bottom_right, bottom_left]
    center: Tuple[float, float]  # 中心座標
    width: float  # 幅
    height: float  # 高さ

    def get_y_threshold(self) -> float:
        """卓球台の上端Y座標を返す（選手分類の閾値として使用）"""
        return self.corners[0][1]  # top_left のY座標

    def is_above_table(self, y_coord: float) -> bool:
        """指定されたY座標が卓球台より上（カメラから遠い）かどうか"""
        return y_coord < self.get_y_threshold()

    def is_below_table(self, y_coord: float) -> bool:
        """指定されたY座標が卓球台より下（カメラに近い）かどうか"""
        return y_coord > self.get_y_threshold()


class TableDetector:
    """卓球台検出クラス"""

    def __init__(
        self,
        detection_method: str = "yolo",  # "yolo" or "color"
        yolo_model_path: str = "yolo11n.pt",
        color_lower_green: Tuple[int, int, int] = (35, 40, 40),
        color_upper_green: Tuple[int, int, int] = (85, 255, 255),
        color_lower_blue: Tuple[int, int, int] = (90, 40, 40),
        color_upper_blue: Tuple[int, int, int] = (130, 255, 255),
        min_area_ratio: float = 0.1,
        max_area_ratio: float = 0.6
    ):
        """
        卓球台検出器の初期化

        Args:
            detection_method: 検出方法（"yolo" または "color"）
            yolo_model_path: YOLOモデルのパス
            color_lower_green: 緑色検出の下限（HSV）
            color_upper_green: 緑色検出の上限（HSV）
            color_lower_blue: 青色検出の下限（HSV）
            color_upper_blue: 青色検出の上限（HSV）
            min_area_ratio: 画面に対する最小面積比
            max_area_ratio: 画面に対する最大面積比
        """
        self.detection_method = detection_method
        self.color_lower_green = np.array(color_lower_green)
        self.color_upper_green = np.array(color_upper_green)
        self.color_lower_blue = np.array(color_lower_blue)
        self.color_upper_blue = np.array(color_upper_blue)
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio

        # YOLOモデルの初期化
        self.yolo_model = None
        if detection_method == "yolo":
            if not YOLO_AVAILABLE:
                print("警告: ultralytics がインストールされていません。色ベース検出にフォールバックします。")
                self.detection_method = "color"
            else:
                try:
                    from ultralytics import YOLO
                    self.yolo_model = YOLO(yolo_model_path)
                    print(f"YOLOモデルをロードしました: {yolo_model_path}")
                except Exception as e:
                    print(f"警告: YOLOモデルのロードに失敗しました: {e}")
                    print("色ベース検出にフォールバックします。")
                    self.detection_method = "color"

    def detect_table(self, frame: np.ndarray) -> Optional[TableRegion]:
        """
        単一フレームから卓球台を検出

        Args:
            frame: 入力フレーム（BGR形式）

        Returns:
            卓球台領域情報、検出できない場合はNone
        """
        if self.detection_method == "yolo" and self.yolo_model is not None:
            return self._detect_table_yolo(frame)
        else:
            return self._detect_table_color(frame)

    def _detect_table_yolo(self, frame: np.ndarray) -> Optional[TableRegion]:
        """
        YOLOを使って卓球台を検出

        戦略:
        1. YOLO物体検出で全オブジェクトを取得
        2. 画面中央上部に位置する大きな横長オブジェクトを卓球台候補とする
        3. 複数候補がある場合は、最も横長で中央にあるものを選択

        Args:
            frame: 入力フレーム（BGR形式）

        Returns:
            卓球台領域情報、検出できない場合はNone
        """
        height, width = frame.shape[:2]
        frame_area = height * width

        # YOLO推論
        results = self.yolo_model(frame, verbose=False)

        if len(results) == 0 or results[0].boxes is None:
            return None

        result = results[0]
        boxes = result.boxes

        # 卓球台候補をフィルタリング
        table_candidates = []

        for box in boxes:
            bbox = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = bbox
            box_width = x2 - x1
            box_height = y2 - y1
            box_area = box_width * box_height
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            # フィルタリング条件
            area_ratio = box_area / frame_area

            # 1. 面積チェック（画面の10%～60%）
            if area_ratio < self.min_area_ratio or area_ratio > self.max_area_ratio:
                continue

            # 2. アスペクト比チェック（横長: 1.5:1 ～ 4:1）
            if box_width == 0 or box_height == 0:
                continue
            aspect_ratio = box_width / box_height
            if aspect_ratio < 1.5 or aspect_ratio > 4.0:
                continue

            # 3. 位置チェック（画面の上部～中央付近）
            vertical_ratio = center_y / height
            if vertical_ratio < 0.2 or vertical_ratio > 0.7:
                continue

            # 4. 水平位置チェック（画面中央付近）
            horizontal_ratio = abs(center_x - width / 2) / width
            if horizontal_ratio > 0.3:  # 中央から30%以上離れていない
                continue

            # スコアリング（中央に近く、横長ほど高スコア）
            center_score = 1.0 - horizontal_ratio
            aspect_score = min(aspect_ratio / 4.0, 1.0)
            area_score = area_ratio / self.max_area_ratio
            total_score = center_score * 0.4 + aspect_score * 0.4 + area_score * 0.2

            table_candidates.append({
                'bbox': bbox,
                'score': total_score
            })

        if not table_candidates:
            return None

        # 最もスコアの高い候補を選択
        best_candidate = max(table_candidates, key=lambda x: x['score'])
        bbox = best_candidate['bbox']

        # 4隅の座標を作成
        x1, y1, x2, y2 = bbox
        corners = np.array([
            [x1, y1],  # top_left
            [x2, y1],  # top_right
            [x2, y2],  # bottom_right
            [x1, y2]   # bottom_left
        ], dtype=np.int32)

        # 中心座標と幅・高さを計算
        center_x = float((x1 + x2) / 2)
        center_y = float((y1 + y2) / 2)
        table_width = float(x2 - x1)
        table_height = float(y2 - y1)

        return TableRegion(
            corners=corners,
            center=(center_x, center_y),
            width=table_width,
            height=table_height
        )

    def _detect_table_color(self, frame: np.ndarray) -> Optional[TableRegion]:
        """
        色ベースで卓球台を検出（既存の実装）

        Args:
            frame: 入力フレーム（BGR形式）

        Returns:
            卓球台領域情報、検出できない場合はNone
        """
        height, width = frame.shape[:2]
        frame_area = height * width

        # HSV色空間に変換
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 緑色と青色のマスクを作成
        mask_green = cv2.inRange(hsv, self.color_lower_green, self.color_upper_green)
        mask_blue = cv2.inRange(hsv, self.color_lower_blue, self.color_upper_blue)

        # マスクを結合
        mask = cv2.bitwise_or(mask_green, mask_blue)

        # ノイズ除去
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # 輪郭検出
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # 最大の輪郭を卓球台として選択
        best_contour = None
        best_area = 0

        for contour in contours:
            area = cv2.contourArea(contour)

            # 面積チェック
            area_ratio = area / frame_area
            if area_ratio < self.min_area_ratio or area_ratio > self.max_area_ratio:
                continue

            # アスペクト比チェック（卓球台は横長）
            rect = cv2.minAreaRect(contour)
            box_width, box_height = rect[1]
            if box_width == 0 or box_height == 0:
                continue

            aspect_ratio = max(box_width, box_height) / min(box_width, box_height)
            if aspect_ratio < 1.5 or aspect_ratio > 4.0:  # 卓球台は1.5:1 ~ 4:1の範囲
                continue

            if area > best_area:
                best_area = area
                best_contour = contour

        if best_contour is None:
            return None

        # 卓球台の矩形を取得
        rect = cv2.minAreaRect(best_contour)
        box = cv2.boxPoints(rect)
        box = np.int32(box)

        # 4隅を並び替え: [top_left, top_right, bottom_right, bottom_left]
        box = self._order_corners(box)

        # 中心座標を計算
        center_x = float(np.mean(box[:, 0]))
        center_y = float(np.mean(box[:, 1]))

        # 幅と高さを計算
        table_width = float(np.linalg.norm(box[0] - box[1]))
        table_height = float(np.linalg.norm(box[1] - box[2]))

        return TableRegion(
            corners=box,
            center=(center_x, center_y),
            width=table_width,
            height=table_height
        )

    def _order_corners(self, corners: np.ndarray) -> np.ndarray:
        """
        4隅の座標を並び替え: [top_left, top_right, bottom_right, bottom_left]

        Args:
            corners: 4隅の座標 (4, 2)

        Returns:
            並び替えた座標 (4, 2)
        """
        # Y座標でソート
        sorted_by_y = corners[np.argsort(corners[:, 1])]

        # 上側2点と下側2点に分ける
        top_points = sorted_by_y[:2]
        bottom_points = sorted_by_y[2:]

        # X座標でソート
        top_left, top_right = top_points[np.argsort(top_points[:, 0])]
        bottom_left, bottom_right = bottom_points[np.argsort(bottom_points[:, 0])]

        return np.array([top_left, top_right, bottom_right, bottom_left])

    def get_stable_table_region(
        self,
        video_loader,
        num_frames: int = 30,
        sample_interval: int = 10
    ) -> Optional[TableRegion]:
        """
        複数フレームから安定した卓球台領域を取得

        Args:
            video_loader: VideoLoaderインスタンス
            num_frames: サンプリングするフレーム数
            sample_interval: サンプリング間隔

        Returns:
            安定化された卓球台領域、検出できない場合はNone
        """
        detected_regions = []
        frame_count = 0

        # 動画の先頭に戻る
        video_loader.reset()

        while len(detected_regions) < num_frames:
            ret, frame = video_loader.read_frame()
            if not ret:
                break

            # サンプリング
            if frame_count % sample_interval == 0:
                region = self.detect_table(frame)
                if region is not None:
                    detected_regions.append(region)

            frame_count += 1

        if not detected_regions:
            return None

        # 複数の検出結果から中央値を取る
        all_corners = np.array([r.corners for r in detected_regions])
        median_corners = np.median(all_corners, axis=0).astype(np.int32)

        # 中心座標、幅、高さを再計算
        center_x = float(np.mean(median_corners[:, 0]))
        center_y = float(np.mean(median_corners[:, 1]))
        table_width = float(np.linalg.norm(median_corners[0] - median_corners[1]))
        table_height = float(np.linalg.norm(median_corners[1] - median_corners[2]))

        # 動画を先頭に戻す
        video_loader.reset()

        return TableRegion(
            corners=median_corners,
            center=(center_x, center_y),
            width=table_width,
            height=table_height
        )

    def draw_table_region(
        self,
        frame: np.ndarray,
        table_region: TableRegion,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 3
    ) -> np.ndarray:
        """
        フレームに卓球台領域を描画

        Args:
            frame: 入力フレーム
            table_region: 卓球台領域
            color: 描画色（BGR）
            thickness: 線の太さ

        Returns:
            描画後のフレーム
        """
        output = frame.copy()

        # 卓球台の輪郭を描画
        cv2.drawContours(output, [table_region.corners], 0, color, thickness)

        # 中心点を描画
        center = (int(table_region.center[0]), int(table_region.center[1]))
        cv2.circle(output, center, 5, color, -1)

        # ラベルを描画
        label = "Table"
        label_pos = (int(table_region.corners[0][0]), int(table_region.corners[0][1]) - 10)
        cv2.putText(output, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, color, 2)

        return output

