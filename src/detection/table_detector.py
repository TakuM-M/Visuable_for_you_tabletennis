"""
卓球台検出モジュール
"""
import cv2
import numpy as np
from typing import Optional, List, Tuple

from utils.video_loader import VideoLoader
from detection.data_classes import TableInfo, CameraAngle

from ultralytics import YOLO


class TableDetector:
    """卓球台検出クラス"""

    def __init__(
        self,
        yolo_model_path: str = "yolo11n.pt",
    ):
        """
        卓球台検出器の初期化

        Args:
            yolo_model_path: YOLOモデルのパス
        """
        # YOLOモデルの初期化
        self.yolo_model = None
        if yolo_model_path is None:
            print("警告:")
        else:
            try:
                self.yolo_model = YOLO(yolo_model_path)
                print(f"YOLOモデルをロードしました: {yolo_model_path}")
            except Exception as e:
                print(f"警告: YOLOモデルのロードに失敗しました: {e}")

    def detect_table_from_frame(self, frame: np.ndarray) -> Optional[TableInfo]:
        """
        単一フレームから卓球台を検出
        1. YOLO物体検出で全卓球台候補を抽出
        2. 画面中央上部に近い位置にあるものを優先的に選択

        Args:
            frame: 入力フレーム（BGR形式）

        Returns:
            TableInfo: 卓球台の情報を返す
            
        Exception:
            検出に失敗した場合は自動で画面中央の領域を返す    
        
        """

    def detect_table_from_video(
        self,
        video_loader: VideoLoader,
        detect_fps: float,
    ) -> Optional[TableInfo]:
        """
        複数フレームから安定した卓球台領域を取得

        Args:
            video_loader: VideoLoaderインスタンス
            detect_fps: 検出する動画のフレームレート

        Returns:
            安定化された卓球台領域、検出できない場合はNone
        """


    def draw_table_region(
        self,
        frame: np.ndarray,
        table_region: TableInfo,
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