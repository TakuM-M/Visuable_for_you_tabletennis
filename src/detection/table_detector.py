"""
卓球台検出モジュール

このモジュールは、YOLOモデルを使用して動画フレームから卓球台のバウンディングボックスを検出します。
検出の際は画面中央に最も近い卓球台を選択し、基本的な座標情報のみを提供します。
"""
import cv2
import numpy as np
from typing import Optional, Tuple

from src.utils.video_loader import VideoLoader
from src.detection.data_classes import TableInfo

from ultralytics import YOLO


# パラメータ定数
CACHE_VALID_FRAMES = 100  # キャッシュ有効期間（フレーム数）


class TableDetector:
    """
    卓球台検出クラス

    YOLOモデルを使用して動画フレームから卓球台のバウンディングボックスを検出します。
    検出された複数の候補の中から、画面中央に最も近いものを卓球台として選択します。

    デフォルトでは、訓練済みの卓球台検出専用モデルを使用します。
    このモデルは以下のクラスを検出できます：
    - Ping Pong Table (卓球台)
    - Paddle (ラケット)
    - Ping Pong Ball (ボール)
    """

    # クラス名の定数
    TABLE_CLASS_NAME = "Ping Pong Table"

    def __init__(
        self,
        yolo_model_path: str = "models/proto_type02_table_detection_models/best.pt",
    ):
        """
        卓球台検出器の初期化

        Args:
            yolo_model_path: YOLOモデルのパス
                           （デフォルト: models/proto_type02_table_detection_models/best.pt）
        """
        # YOLOモデルの初期化
        self.yolo_model = None
        self.class_names = {}

        if yolo_model_path is None:
            print("警告: YOLOモデルパスが指定されていません")
        else:
            try:
                self.yolo_model = YOLO(yolo_model_path)
                # クラス名を取得
                self.class_names = self.yolo_model.names
                print(f"YOLOモデルをロードしました: {yolo_model_path}")
                print(f"検出可能なクラス: {list(self.class_names.values())}")
            except Exception as e:
                print(f"警告: YOLOモデルのロードに失敗しました: {e}")

        # キャッシュ
        self._cached_table_info: Optional[TableInfo] = None
        self._cache_frame_idx: int = -1

    def detect_table_from_frame(
        self,
        frame: np.ndarray,
        frame_idx: int = 0,
        force_detect: bool = False
    ) -> Optional[TableInfo]:
        """
        単一フレームから卓球台のバウンディングボックスを検出

        処理フロー:
        1. キャッシュチェック（CACHE_VALID_FRAMES以内なら再利用）
        2. YOLO物体検出で全候補を抽出
        3. 画面中央に最も近い候補を選択
        4. バウンディングボックスと基本情報をTableInfoとして返す

        Args:
            frame: 入力フレーム（BGR形式）
            frame_idx: フレーム番号
            force_detect: Trueの場合、キャッシュを無視して再検出

        Returns:
            TableInfo: 卓球台のバウンディングボックス情報、検出失敗時はNone
        """
        # キャッシュチェック
        if (not force_detect and
            self._cached_table_info is not None and
            abs(frame_idx - self._cache_frame_idx) < CACHE_VALID_FRAMES):
            return self._cached_table_info

        if self.yolo_model is None:
            print("警告: YOLOモデルが初期化されていません")
            return None

        # YOLO検出を実行
        results = self.yolo_model(frame, verbose=False)

        if len(results) == 0 or len(results[0].boxes) == 0:
            return None

        # 全検出結果から"Ping Pong Table"クラスのみをフィルタリング
        boxes = results[0].boxes
        frame_height, frame_width = frame.shape[:2]
        frame_center = (frame_width / 2, frame_height / 2)

        best_table = None
        min_distance = float('inf')

        for box in boxes:
            # クラスIDとクラス名を取得
            class_id = int(box.cls[0].cpu().numpy())
            class_name = self.class_names.get(class_id, "Unknown")

            # "Ping Pong Table"クラスのみを処理
            if class_name != self.TABLE_CLASS_NAME:
                continue

            # バウンディングボックス座標を取得
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf[0].cpu().numpy())

            # バウンディングボックスの中心座標を計算
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            # 画面中央からの距離を計算（ユークリッド距離）
            distance = np.sqrt((center_x - frame_center[0])**2 +
                             (center_y - frame_center[1])**2)

            # 最も中央に近いものを卓球台として選択
            if distance < min_distance:
                min_distance = distance
                best_table = TableInfo(
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    confidence=confidence,
                    frame_idx=frame_idx
                )

        if best_table is None:
            return None

        # キャッシュに保存
        self._cached_table_info = best_table
        self._cache_frame_idx = frame_idx

        return best_table

    def detect_table_from_video(
        self,
        video_loader: VideoLoader,
        detect_fps: float,
    ) -> Optional[TableInfo]:
        """
        動画から卓球台のバウンディングボックスを検出

        現在の実装では最初のフレームから検出を試みます。
        カメラが固定されている前提で、最初のフレームの検出結果を
        キャッシュとして利用することで処理を効率化します。

        Args:
            video_loader: VideoLoaderインスタンス
            detect_fps: 検出フレームレート（将来の拡張用、現在は未使用）

        Returns:
            TableInfo: 卓球台のバウンディングボックス情報、検出できない場合はNone
        """
        # 最初のフレームを取得
        frame = video_loader.get_frame(0)
        if frame is None:
            return None

        # 卓球台のバウンディングボックスを検出
        table_info = self.detect_table_from_frame(frame, frame_idx=0, force_detect=True)

        return table_info

    def draw_table_region(
        self,
        frame: np.ndarray,
        table_info: TableInfo,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 3
    ) -> np.ndarray:
        """
        フレームに卓球台のバウンディングボックスを描画

        デバッグや可視化のために、検出された卓球台のバウンディングボックスを
        フレーム上に矩形として描画します。

        Args:
            frame: 入力フレーム（BGR形式）
            table_info: 卓球台のバウンディングボックス情報
            color: 描画色（BGR形式）、デフォルト: 緑(0, 255, 0)
            thickness: 矩形の線の太さ

        Returns:
            バウンディングボックスが描画されたフレーム
        """
        if table_info is None:
            return frame

        result = frame.copy()
        x1, y1, x2, y2 = table_info.bbox

        # バウンディングボックスを矩形として描画
        cv2.rectangle(
            result,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            color,
            thickness
        )

        # 信頼度を表示
        text = f"Table: {table_info.confidence:.2f}"
        cv2.putText(
            result,
            text,
            (int(x1), int(y1) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

        return result