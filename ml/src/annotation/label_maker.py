"""LabelMaker class for manual annotation of play/non-play scenes in table tennis videos.

This module provides an interactive GUI tool for labeling video frames.
"""
import cv2
import pandas as pd
import numpy as np
from pathlib import Path


class LabelMaker:
    """
    ラベル作成用のインタラクティブツール

    使い方:
    - 'k': 再生/一時停止
    - 's': 現在のフレームをプレー開始として記録
    - 'e': 現在のフレームをプレー終了として記録
    - 'd': 最後に追加したラベルを削除
    - 'l': 次のフレーム（一時停止中）
    - 'j': 前のフレーム（一時停止中）
    - 'q': 保存して終了
    """

    def __init__(self, video_path: str, output_path: str = None, fps_divisor: float = 1.0):
        """
        初期化

        Args:
            video_path: 動画ファイルのパス
            output_path: 出力CSVパス（Noneの場合は自動生成）
            fps_divisor: 表示フレームレートの倍率（2なら半分速度、0.5なら2倍速）
        """
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"動画が見つかりません: {video_path}")

        # 出力パス
        if output_path:
            self.output_path = Path(output_path)
        else:
            self.output_path = self.video_path.parent / f"{self.video_path.stem}_labels.csv"

        # 既存のラベルファイルがあれば読み込む
        self.play_scenes = []  # [(start_frame, end_frame), ...]
        if self.output_path.exists():
            self._load_existing_labels()

        # 動画読み込み
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise ValueError(f"動画を開けません: {video_path}")

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps_divisor = fps_divisor

        # 状態管理
        self.current_frame = 0
        self.is_playing = False
        self.temp_start_frame = None  # 一時的なプレー開始フレーム

        print(f"\n{'='*60}")
        print(f"ラベル作成ツール")
        print(f"{'='*60}")
        print(f"動画: {self.video_path.name}")
        print(f"総フレーム数: {self.total_frames}")
        print(f"FPS: {self.fps}")
        print(f"既存ラベル: {len(self.play_scenes)} シーン")
        print(f"\n操作方法:")
        print(f"  k/スペース: 再生/一時停止")
        print(f"  's': プレー開始フレームを記録")
        print(f"  'e': プレー終了フレームを記録")
        print(f"  'd': 最後のシーンを削除")
        print(f"  'l': 次のフレーム（一時停止中）")
        print(f"  'j': 前のフレーム（一時停止中）")
        print(f"  'q': 保存して終了")
        print(f"{'='*60}\n")

    def _load_existing_labels(self):
        """既存のラベルファイルを読み込む"""
        try:
            df = pd.read_csv(self.output_path)
            if 'start_frame' in df.columns and 'end_frame' in df.columns:
                # シーン形式
                for _, row in df.iterrows():
                    self.play_scenes.append((int(row['start_frame']), int(row['end_frame'])))
            else:
                # フレーム単位形式
                play_frames = df[df['label'] == 1]['frame'].values
                if len(play_frames) > 0:
                    # 連続したフレームをシーンとしてグループ化
                    scenes = []
                    start = play_frames[0]
                    prev = play_frames[0]
                    for frame in play_frames[1:]:
                        if frame - prev > 1:
                            scenes.append((start, prev))
                            start = frame
                        prev = frame
                    scenes.append((start, prev))
                    self.play_scenes = scenes

            print(f"既存のラベルを読み込みました: {len(self.play_scenes)} シーン")
        except Exception as e:
            print(f"既存ラベルの読み込みに失敗: {e}")

    def _draw_frame(self, frame: np.ndarray) -> np.ndarray:
        """フレームに情報を描画"""
        frame = frame.copy()

        # 現在のフレームがプレー中かチェック
        is_in_play = self._is_frame_in_play(self.current_frame)

        # 背景色（プレー中なら緑、そうでないなら黒）
        bg_color = (0, 200, 0) if is_in_play else (0, 0, 0)
        cv2.rectangle(frame, (10, 10), (400, 150), bg_color, -1)

        # フレーム情報
        timestamp = self.current_frame / self.fps
        text_color = (255, 255, 255)

        cv2.putText(frame, f"Frame: {self.current_frame}/{self.total_frames}",
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        cv2.putText(frame, f"Time: {timestamp:.2f}s",
                   (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

        status = "PLAYING" if is_in_play else "NON-PLAY"
        cv2.putText(frame, f"Status: {status}",
                   (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

        cv2.putText(frame, f"Scenes: {len(self.play_scenes)}",
                   (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

        # 一時的な開始フレームがある場合
        if self.temp_start_frame is not None:
            cv2.rectangle(frame, (10, 160), (400, 200), (0, 255, 255), -1)
            cv2.putText(frame, f"Start: {self.temp_start_frame} (Press 'e' to end)",
                       (20, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # 再生状態
        play_status = "PLAYING" if self.is_playing else "PAUSED"
        cv2.putText(frame, play_status,
                   (self.width - 150, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return frame

    def _is_frame_in_play(self, frame_num: int) -> bool:
        """指定フレームがプレー中かチェック"""
        for start, end in self.play_scenes:
            if start <= frame_num <= end:
                return True
        return False

    def _add_scene(self, start: int, end: int):
        """プレーシーンを追加"""
        if start >= end:
            print(f"エラー: 開始フレーム({start})が終了フレーム({end})より後です")
            return

        # 重複チェック
        for existing_start, existing_end in self.play_scenes:
            if not (end < existing_start or start > existing_end):
                print(f"警告: シーンが重複しています ({start}-{end})")

        self.play_scenes.append((start, end))
        self.play_scenes.sort()
        print(f"シーン追加: フレーム {start}-{end} ({(end-start)/self.fps:.1f}秒)")

    def _delete_last_scene(self):
        """最後のシーンを削除"""
        if self.play_scenes:
            deleted = self.play_scenes.pop()
            print(f"シーン削除: フレーム {deleted[0]}-{deleted[1]}")
        else:
            print("削除するシーンがありません")

    def _save_labels(self):
        """ラベルをCSV形式で保存"""
        # フレーム単位のラベルを作成
        labels = np.zeros(self.total_frames, dtype=int)
        for start, end in self.play_scenes:
            labels[start:end+1] = 1

        # DataFrame作成
        df = pd.DataFrame({
            'frame': range(self.total_frames),
            'label': labels
        })

        # 保存
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.output_path, index=False)

        # シーン情報も保存
        scenes_path = self.output_path.parent / f"{self.output_path.stem}_scenes.csv"
        scenes_df = pd.DataFrame(self.play_scenes, columns=['start_frame', 'end_frame'])
        scenes_df['duration_sec'] = (scenes_df['end_frame'] - scenes_df['start_frame']) / self.fps
        scenes_df.to_csv(scenes_path, index=False)

        print(f"\nラベル保存完了:")
        print(f"  フレーム単位: {self.output_path}")
        print(f"  シーン単位: {scenes_path}")
        print(f"  総シーン数: {len(self.play_scenes)}")
        print(f"  プレー中フレーム: {np.sum(labels == 1)} / {self.total_frames}")

    def run(self):
        """メインループ"""
        window_name = "Label Maker - Press 'h' for help"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        while True:
            # フレーム読み込み
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            ret, frame = self.cap.read()

            if not ret:
                print("動画の終わりに到達しました")
                break

            # 情報を描画
            display_frame = self._draw_frame(frame)

            # 表示
            cv2.imshow(window_name, display_frame)

            # キー入力待ち
            wait_time = max(1, int(1000 / self.fps * self.fps_divisor)) if self.is_playing else 0
            key = cv2.waitKey(wait_time) & 0xFF

            # キー操作
            if key == ord('q'):  # 終了
                print("\n終了します...")
                break

            elif key == ord(' ') or key == ord('k'):  # 再生/一時停止
                self.is_playing = not self.is_playing
                status = "再生" if self.is_playing else "一時停止"
                print(f"{status}しました（フレーム: {self.current_frame}）")

            elif key == ord('s'):  # プレー開始
                self.temp_start_frame = self.current_frame
                print(f"プレー開始: フレーム {self.current_frame}")

            elif key == ord('e'):  # プレー終了
                if self.temp_start_frame is not None:
                    self._add_scene(self.temp_start_frame, self.current_frame)
                    self.temp_start_frame = None
                else:
                    print("エラー: 先に's'でプレー開始を記録してください")

            elif key == ord('d'):  # 削除
                self._delete_last_scene()

            elif key == ord('l'):  # 次のフレーム
                if not self.is_playing:
                    self.current_frame = min(self.current_frame + 1, self.total_frames - 1)

            elif key == ord('j'):  # 前のフレーム
                if not self.is_playing:
                    self.current_frame = max(self.current_frame - 1, 0)

            elif key == ord('h'):  # ヘルプ
                print(f"\n{'='*60}")
                print("操作方法:")
                print("  k/スペース: 再生/一時停止")
                print("  's': プレー開始フレームを記録")
                print("  'e': プレー終了フレームを記録")
                print("  'd': 最後のシーンを削除")
                print("  'l': 次のフレーム（一時停止中）")
                print("  'j': 前のフレーム（一時停止中）")
                print("  'q': 保存して終了")
                print(f"{'='*60}\n")

            # フレーム進行
            if self.is_playing:
                # fps_divisorが1未満の場合、フレームをスキップして高速再生
                frame_step = max(1, int(1 / self.fps_divisor)) if self.fps_divisor < 1 else 1
                self.current_frame += frame_step
                if self.current_frame >= self.total_frames:
                    self.current_frame = 0
                    self.is_playing = False
                    print("動画の最後まで到達しました")

        # 終了処理
        self._save_labels()
        self.cap.release()
        cv2.destroyAllWindows()
