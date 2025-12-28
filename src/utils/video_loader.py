"""
動画読み込みユーティリティ
"""
import cv2
from pathlib import Path
from typing import Optional, Tuple


class VideoLoader:
    """動画ファイルを読み込むクラス"""

    def __init__(self, video_path: str):
        """
        Args:
            video_path: 動画ファイルのパス
        """
        self.video_path = Path(video_path)
        self.cap: Optional[cv2.VideoCapture] = None
        self.fps: float = 0
        self.frame_count: int = 0
        self.width: int = 0
        self.height: int = 0

    def open(self) -> bool:
        """動画ファイルを開く

        Returns:
            成功した場合True
        """
        if not self.video_path.exists():
            raise FileNotFoundError(f"動画ファイルが見つかりません: {self.video_path}")

        self.cap = cv2.VideoCapture(str(self.video_path))

        if not self.cap.isOpened():
            return False

        # 動画情報を取得
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        return True

    def read_frame(self) -> Tuple[bool, Optional[cv2.Mat]]:
        """フレームを読み込む

        Returns:
            (成功フラグ, フレーム画像)
        """
        if self.cap is None:
            return False, None

        return self.cap.read()

    def get_info(self) -> dict:
        """動画情報を取得

        Returns:
            動画情報の辞書
        """
        return {
            'fps': self.fps,
            'frame_count': self.frame_count,
            'width': self.width,
            'height': self.height,
            'duration': self.frame_count / self.fps if self.fps > 0 else 0
        }

    def close(self):
        """動画ファイルを閉じる"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        """コンテキストマネージャー対応"""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """コンテキストマネージャー対応"""
        self.close()
