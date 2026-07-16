"""LabelMaker: Tkinter + OpenCV ベースのプレーシーンアノテーションツール

動画を再生しながらプレー中/プレー外のラベルを手動で付けるGUIツール。
タイムライン、シーンリスト、マウス操作、キーボードショートカットに対応。
"""

import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

from PIL import Image, ImageTk


# ---------------------------------------------------------------------------
# VideoHandler
# ---------------------------------------------------------------------------
class VideoHandler:
    """OpenCV動画の読み込みとフレーム変換を担当"""

    def __init__(
        self, video_path: str, fps_divisor: float = 1.0, target_fps: float | None = None
    ):
        self.path = Path(video_path)
        if not self.path.exists():
            raise FileNotFoundError(f"動画が見つかりません: {video_path}")

        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            raise ValueError(f"動画を開けません: {video_path}")

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps_divisor = fps_divisor
        self._last_read_pos = -1

        # target_fps に基づくフレームステップ（ポーズ抽出と同じ計算式）
        if target_fps and target_fps < self.fps:
            self.frame_step = max(1, round(self.fps / target_fps))
            self.effective_fps = self.fps / self.frame_step
        else:
            self.frame_step = 1
            self.effective_fps = self.fps

    def seek(self, frame_num: int):
        """指定フレームにシーク"""
        frame_num = max(0, min(frame_num, self.total_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        self._last_read_pos = frame_num - 1

    def read_frame(self, frame_num: int) -> np.ndarray | None:
        """指定フレームを読み込む（シーケンシャルなら高速）"""
        if frame_num != self._last_read_pos + 1:
            self.seek(frame_num)
        ret, frame = self.cap.read()
        if ret:
            self._last_read_pos = frame_num
            return frame
        return None

    def frame_to_photo(
        self, frame: np.ndarray, max_w: int, max_h: int
    ) -> ImageTk.PhotoImage:
        """OpenCVフレームをTkinter PhotoImageに変換（アスペクト比維持でリサイズ）"""
        h, w = frame.shape[:2]
        scale = min(max_w / w, max_h / h, 1.0)
        if scale < 1.0:
            new_w, new_h = int(w * scale), int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return ImageTk.PhotoImage(Image.fromarray(rgb))

    def release(self):
        if self.cap:
            self.cap.release()


# ---------------------------------------------------------------------------
# SceneDataModel
# ---------------------------------------------------------------------------
class SceneDataModel:
    """プレーシーンデータの管理（追加・削除・読み書き）"""

    def __init__(self, output_path: Path, total_frames: int, fps: float):
        self.output_path = output_path
        self.total_frames = total_frames
        self.fps = fps
        self.play_scenes: list[tuple[int, int]] = []
        self.temp_start_frame: int | None = None

        if self.output_path.exists():
            self._load_existing_labels()

    def _load_existing_labels(self):
        try:
            df = pd.read_csv(self.output_path)
            if "start_frame" in df.columns and "end_frame" in df.columns:
                for _, row in df.iterrows():
                    self.play_scenes.append(
                        (int(row["start_frame"]), int(row["end_frame"]))
                    )
            elif "label" in df.columns:
                play_frames = df[df["label"] == 1]["frame"].values
                if len(play_frames) > 0:
                    start = play_frames[0]
                    prev = play_frames[0]
                    for frame in play_frames[1:]:
                        if frame - prev > 1:
                            self.play_scenes.append((start, prev))
                            start = frame
                        prev = frame
                    self.play_scenes.append((start, prev))
            self.play_scenes.sort()
        except Exception as e:
            print(f"既存ラベルの読み込みに失敗: {e}")

    def add_scene(self, start: int, end: int) -> str | None:
        """シーンを追加。エラー時はメッセージを返す"""
        if start >= end:
            return f"開始({start})が終了({end})以降です"
        for s, e in self.play_scenes:
            if not (end < s or start > e):
                return f"既存シーン({s}-{e})と重複しています"
        self.play_scenes.append((start, end))
        self.play_scenes.sort()
        return None

    def delete_scene(self, index: int) -> tuple[int, int] | None:
        if 0 <= index < len(self.play_scenes):
            return self.play_scenes.pop(index)
        return None

    def delete_last_scene(self) -> tuple[int, int] | None:
        if self.play_scenes:
            return self.play_scenes.pop()
        return None

    def set_start(self, frame: int):
        self.temp_start_frame = frame

    def complete_scene(
        self, end_frame: int
    ) -> tuple[str | None, tuple[int, int] | None]:
        """仮開始フレームから終了フレームまでのシーンを完成。(エラー, シーン)を返す"""
        if self.temp_start_frame is None:
            return "先に's'でプレー開始を記録してください", None
        start = self.temp_start_frame
        self.temp_start_frame = None
        err = self.add_scene(start, end_frame)
        if err:
            return err, None
        return None, (start, end_frame)

    def is_frame_in_play(self, frame: int) -> bool:
        for s, e in self.play_scenes:
            if s <= frame <= e:
                return True
        return False

    def save(self):
        """CSV保存（フレーム単位 + シーン単位）"""
        labels = np.zeros(self.total_frames, dtype=int)
        for start, end in self.play_scenes:
            labels[start : end + 1] = 1

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"frame": range(self.total_frames), "label": labels}).to_csv(
            self.output_path, index=False
        )

        scenes_path = self.output_path.parent / f"{self.output_path.stem}_scenes.csv"
        scenes_df = pd.DataFrame(self.play_scenes, columns=["start_frame", "end_frame"])
        if not scenes_df.empty:
            scenes_df["duration_sec"] = (
                scenes_df["end_frame"] - scenes_df["start_frame"]
            ) / self.fps
        scenes_df.to_csv(scenes_path, index=False)

        print(f"ラベル保存: {self.output_path}")
        print(f"シーン保存: {scenes_path} ({len(self.play_scenes)}シーン)")


# ---------------------------------------------------------------------------
# TimelineCanvas
# ---------------------------------------------------------------------------
class TimelineCanvas(tk.Canvas):
    """動画タイムラインを表示するカスタムCanvas"""

    TIMELINE_HEIGHT = 40
    BG_COLOR = "#3a3a3a"
    SCENE_COLOR = "#4CAF50"
    TEMP_COLOR = "#FFC107"
    POS_COLOR = "#F44336"

    def __init__(self, parent, on_seek, **kwargs):
        super().__init__(
            parent,
            height=self.TIMELINE_HEIGHT,
            bg=self.BG_COLOR,
            highlightthickness=0,
            **kwargs,
        )
        self._on_seek = on_seek
        self._fps = 30.0
        self._total_frames = 1
        self.bind("<Button-1>", self._on_click)
        self.bind("<B1-Motion>", self._on_drag)

    def set_fps(self, fps: float):
        self._fps = fps

    def redraw(
        self,
        current_frame: int,
        total_frames: int,
        scenes: list[tuple[int, int]],
        temp_start: int | None,
    ):
        self._total_frames = max(total_frames, 1)
        self.delete("all")
        w = self.winfo_width()
        h = self.TIMELINE_HEIGHT

        # 背景バー
        self.create_rectangle(0, 8, w, h - 8, fill="#555555", outline="")

        # シーン
        for s, e in scenes:
            x1 = self._frame_to_x(s, w)
            x2 = self._frame_to_x(e, w)
            self.create_rectangle(
                x1, 4, max(x2, x1 + 2), h - 4, fill=self.SCENE_COLOR, outline=""
            )

        # 仮開始フレーム〜現在位置
        if temp_start is not None:
            tx = self._frame_to_x(temp_start, w)
            cx = self._frame_to_x(current_frame, w)
            self.create_rectangle(
                tx,
                4,
                max(cx, tx + 2),
                h - 4,
                fill=self.TEMP_COLOR,
                outline="",
                stipple="gray50",
            )
            self.create_line(tx, 0, tx, h, fill=self.TEMP_COLOR, width=2)

        # 時間ラベル
        duration_sec = total_frames / max(self._fps, 1)
        interval = self._tick_interval(duration_sec)
        if interval > 0:
            t = interval
            while t < duration_sec:
                x = self._frame_to_x(int(t * self._fps), w)
                self.create_line(x, h - 8, x, h - 4, fill="#aaaaaa")
                self.create_text(
                    x,
                    h - 2,
                    text=self._format_time(t),
                    fill="#aaaaaa",
                    font=("", 8),
                    anchor="s",
                )
                t += interval

        # 現在位置（最後に描画して最前面に）
        px = self._frame_to_x(current_frame, w)
        self.create_line(px, 0, px, h, fill=self.POS_COLOR, width=2)

    def _tick_interval(self, duration: float) -> float:
        if duration <= 30:
            return 5
        if duration <= 120:
            return 15
        if duration <= 600:
            return 30
        return 60

    @staticmethod
    def _format_time(sec: float) -> str:
        m, s = divmod(int(sec), 60)
        return f"{m}:{s:02d}"

    def _frame_to_x(self, frame: int, width: int) -> int:
        return int((frame / self._total_frames) * width)

    def _x_to_frame(self, x: int) -> int:
        w = max(self.winfo_width(), 1)
        return int((x / w) * self._total_frames)

    def _on_click(self, event):
        self._on_seek(self._x_to_frame(event.x))

    def _on_drag(self, event):
        self._on_seek(self._x_to_frame(event.x))


# ---------------------------------------------------------------------------
# AnnotationGUI
# ---------------------------------------------------------------------------
class AnnotationGUI:
    """Tkinterベースのアノテーション画面"""

    VIDEO_MAX_W = 960
    VIDEO_MAX_H = 540
    SPEEDS = [0.5, 1.0, 2.0, 4.0]

    def __init__(self, video: VideoHandler, model: SceneDataModel):
        self.video = video
        self.model = model
        self.current_frame = 0
        self.is_playing = False
        self.speed = 1.0
        self._photo: ImageTk.PhotoImage | None = None
        self._after_id: str | None = None

        self.root = tk.Tk()
        self.root.title(f"Annotation Tool - {video.path.name}")
        self.root.configure(bg="#2b2b2b")
        self._build_ui()
        self._bind_keys()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # 初期表示
        self.root.after(100, self._update_display)

    # ---- UI構築 ----

    def _build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TButton", padding=4)
        style.configure("TLabel", background="#2b2b2b", foreground="white")
        style.configure("TFrame", background="#2b2b2b")

        main = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True)

        # 左パネル（動画 + コントロール + タイムライン）
        left = ttk.Frame(main)
        main.add(left, weight=3)

        self._build_video_canvas(left)
        self._build_controls(left)
        self._build_timeline(left)
        self._build_status_bar(left)

        # 右パネル（シーンリスト）
        right = ttk.Frame(main)
        main.add(right, weight=1)
        self._build_scene_list(right)

    def _build_video_canvas(self, parent):
        self.video_canvas = tk.Canvas(
            parent,
            width=self.VIDEO_MAX_W,
            height=self.VIDEO_MAX_H,
            bg="black",
            highlightthickness=0,
        )
        self.video_canvas.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    def _build_controls(self, parent):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, padx=4, pady=2)

        # ナビゲーションボタン
        btns = [
            ("|<", lambda: self._seek_to(0)),
            ("-5s", lambda: self._step_seconds(-5)),
            ("-1s", lambda: self._step_seconds(-1)),
            ("<", lambda: self._step_frames(-1)),
        ]
        for text, cmd in btns:
            ttk.Button(frame, text=text, width=4, command=cmd).pack(
                side=tk.LEFT, padx=1
            )

        self.play_btn = ttk.Button(
            frame, text="Play", width=6, command=self._toggle_play
        )
        self.play_btn.pack(side=tk.LEFT, padx=2)

        btns2 = [
            (">", lambda: self._step_frames(1)),
            ("+1s", lambda: self._step_seconds(1)),
            ("+5s", lambda: self._step_seconds(5)),
            (">|", lambda: self._seek_to(self.video.total_frames - 1)),
        ]
        for text, cmd in btns2:
            ttk.Button(frame, text=text, width=4, command=cmd).pack(
                side=tk.LEFT, padx=1
            )

        # スピード
        ttk.Label(frame, text="  Speed:").pack(side=tk.LEFT)
        self.speed_var = tk.StringVar(value="1.0x")
        speed_combo = ttk.Combobox(
            frame,
            textvariable=self.speed_var,
            values=[f"{s}x" for s in self.SPEEDS],
            width=5,
            state="readonly",
        )
        speed_combo.pack(side=tk.LEFT, padx=2)
        speed_combo.bind("<<ComboboxSelected>>", self._on_speed_change)

        # アノテーションボタン
        ttk.Separator(frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        ttk.Button(frame, text="Start [s]", command=self._on_mark_start).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(frame, text="End [e]", command=self._on_mark_end).pack(
            side=tk.LEFT, padx=2
        )

        # フレーム情報
        self.info_var = tk.StringVar(value="")
        ttk.Label(frame, textvariable=self.info_var, font=("Menlo", 11)).pack(
            side=tk.RIGHT, padx=6
        )

    def _build_timeline(self, parent):
        self.timeline = TimelineCanvas(parent, on_seek=self._seek_to)
        self.timeline.set_fps(self.video.fps)
        self.timeline.pack(fill=tk.X, padx=4, pady=2)

    def _build_scene_list(self, parent):
        ttk.Label(parent, text="Play Scenes", font=("", 12, "bold")).pack(pady=(8, 4))

        list_frame = ttk.Frame(parent)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=4)

        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.scene_listbox = tk.Listbox(
            list_frame,
            yscrollcommand=scrollbar.set,
            bg="#1e1e1e",
            fg="white",
            selectbackground="#4CAF50",
            font=("Menlo", 10),
            activestyle="none",
        )
        self.scene_listbox.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.scene_listbox.yview)

        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, padx=4, pady=4)
        ttk.Button(btn_frame, text="Jump", command=self._on_jump_to_scene).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(btn_frame, text="Delete", command=self._on_delete_selected).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(
            btn_frame, text="Delete Last [d]", command=self._on_delete_last
        ).pack(side=tk.LEFT, padx=2)

    def _build_status_bar(self, parent):
        self.status_var = tk.StringVar(value="Ready")
        status = ttk.Label(
            parent, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W
        )
        status.pack(fill=tk.X, padx=4, pady=(0, 4))

    # ---- キーバインド ----

    def _bind_keys(self):
        self.root.bind("<space>", lambda e: self._toggle_play())
        self.root.bind("k", lambda e: self._toggle_play())
        self.root.bind("s", lambda e: self._on_mark_start())
        self.root.bind("e", lambda e: self._on_mark_end())
        self.root.bind("d", lambda e: self._on_delete_last())
        self.root.bind("q", lambda e: self._on_close())

        # フレーム移動
        self.root.bind("l", lambda e: self._step_frames(1))
        self.root.bind("j", lambda e: self._step_frames(-1))
        self.root.bind("<Right>", lambda e: self._step_frames(1))
        self.root.bind("<Left>", lambda e: self._step_frames(-1))

        # 大ジャンプ
        self.root.bind("<Shift-Right>", lambda e: self._step_seconds(1))
        self.root.bind("<Shift-Left>", lambda e: self._step_seconds(-1))
        self.root.bind("<Control-Right>", lambda e: self._step_seconds(5))
        self.root.bind("<Control-Left>", lambda e: self._step_seconds(-5))

        # Delete キーで選択シーン削除
        self.root.bind("<Delete>", lambda e: self._on_delete_selected())
        self.root.bind("<BackSpace>", lambda e: self._on_delete_selected())

    # ---- 再生制御 ----

    def _toggle_play(self):
        self.is_playing = not self.is_playing
        self.play_btn.config(text="Pause" if self.is_playing else "Play")
        if self.is_playing:
            self._playback_tick()

    def _playback_tick(self):
        if not self.is_playing:
            return
        base_step = self.video.frame_step
        step = base_step * (max(1, int(self.speed)) if self.speed >= 1 else 1)
        self.current_frame = min(self.current_frame + step, self.video.total_frames - 1)
        if self.current_frame >= self.video.total_frames - 1:
            self.is_playing = False
            self.play_btn.config(text="Play")
            self._set_status("動画の最後に到達しました")
        self._update_display()
        if self.is_playing:
            delay = max(1, int(1000 / self.video.effective_fps / max(self.speed, 0.1)))
            self._after_id = self.root.after(delay, self._playback_tick)

    def _snap_to_step(self, frame: int) -> int:
        """フレーム番号をframe_stepの倍数にスナップ"""
        step = self.video.frame_step
        return round(frame / step) * step

    def _step_frames(self, n: int):
        """n ステップ分移動（1ステップ = frame_step フレーム）"""
        if self.is_playing:
            return
        step = n * self.video.frame_step
        self.current_frame = max(
            0, min(self.current_frame + step, self.video.total_frames - 1)
        )
        self._update_display()

    def _step_seconds(self, sec: float):
        frames = int(abs(sec) * self.video.fps)
        if sec < 0:
            frames = -frames
        target = self.current_frame + frames
        self.current_frame = max(
            0, min(self._snap_to_step(target), self.video.total_frames - 1)
        )
        self._update_display()

    def _seek_to(self, frame: int):
        self.current_frame = max(
            0, min(self._snap_to_step(frame), self.video.total_frames - 1)
        )
        self._update_display()

    def _on_speed_change(self, event=None):
        text = self.speed_var.get().rstrip("x")
        try:
            self.speed = float(text)
        except ValueError:
            self.speed = 1.0

    # ---- 表示更新 ----

    def _update_display(self):
        frame = self.video.read_frame(self.current_frame)
        if frame is not None:
            cw = max(self.video_canvas.winfo_width(), 320)
            ch = max(self.video_canvas.winfo_height(), 240)
            self._photo = self.video.frame_to_photo(frame, cw, ch)
            self.video_canvas.delete("all")
            self.video_canvas.create_image(
                cw // 2, ch // 2, image=self._photo, anchor=tk.CENTER
            )

            # プレー中オーバーレイ
            if self.model.is_frame_in_play(self.current_frame):
                self.video_canvas.create_rectangle(
                    cw - 100, 10, cw - 10, 40, fill="#4CAF50", outline=""
                )
                self.video_canvas.create_text(
                    cw - 55, 25, text="PLAY", fill="white", font=("", 12, "bold")
                )
            elif self.model.temp_start_frame is not None:
                self.video_canvas.create_rectangle(
                    cw - 120, 10, cw - 10, 40, fill="#FFC107", outline=""
                )
                self.video_canvas.create_text(
                    cw - 65, 25, text="MARKING", fill="black", font=("", 12, "bold")
                )

        # フレーム情報
        t = self.current_frame / self.video.fps
        total_t = self.video.total_frames / self.video.fps
        self.info_var.set(
            f"Frame {self.current_frame}/{self.video.total_frames}  "
            f"{self._fmt(t)} / {self._fmt(total_t)}"
        )

        # タイムライン
        self.timeline.redraw(
            self.current_frame,
            self.video.total_frames,
            self.model.play_scenes,
            self.model.temp_start_frame,
        )

    @staticmethod
    def _fmt(sec: float) -> str:
        m, s = divmod(sec, 60)
        return f"{int(m)}:{s:05.2f}"

    # ---- シーン操作 ----

    def _on_mark_start(self):
        self.model.set_start(self.current_frame)
        self._set_status(f"プレー開始マーク: Frame {self.current_frame}")
        self._update_display()

    def _on_mark_end(self):
        err, scene = self.model.complete_scene(self.current_frame)
        if err:
            self._set_status(f"Error: {err}")
        else:
            dur = (scene[1] - scene[0]) / self.video.fps
            self._set_status(f"シーン追加: {scene[0]}-{scene[1]} ({dur:.1f}秒)")
            self._refresh_scene_list()
        self._update_display()

    def _on_delete_selected(self):
        sel = self.scene_listbox.curselection()
        if not sel:
            self._set_status("シーンを選択してください")
            return
        idx = sel[0]
        deleted = self.model.delete_scene(idx)
        if deleted:
            self._set_status(f"シーン削除: {deleted[0]}-{deleted[1]}")
            self._refresh_scene_list()
            self._update_display()

    def _on_delete_last(self):
        deleted = self.model.delete_last_scene()
        if deleted:
            self._set_status(f"シーン削除: {deleted[0]}-{deleted[1]}")
            self._refresh_scene_list()
            self._update_display()
        else:
            self._set_status("削除するシーンがありません")

    def _on_jump_to_scene(self):
        sel = self.scene_listbox.curselection()
        if not sel:
            self._set_status("シーンを選択してください")
            return
        idx = sel[0]
        if 0 <= idx < len(self.model.play_scenes):
            start = self.model.play_scenes[idx][0]
            self._seek_to(start)
            self._set_status(f"シーン #{idx + 1} にジャンプ")

    def _refresh_scene_list(self):
        self.scene_listbox.delete(0, tk.END)
        for i, (s, e) in enumerate(self.model.play_scenes):
            dur = (e - s) / self.video.fps
            ts = self._fmt(s / self.video.fps)
            te = self._fmt(e / self.video.fps)
            self.scene_listbox.insert(tk.END, f"#{i + 1}: {ts}-{te} ({dur:.1f}s)")

    def _set_status(self, msg: str):
        self.status_var.set(msg)
        print(msg)

    # ---- 終了処理 ----

    def _on_close(self):
        if self._after_id:
            self.root.after_cancel(self._after_id)
        self.is_playing = False
        self.model.save()
        self.video.release()
        self.root.destroy()


# ---------------------------------------------------------------------------
# LabelMaker (公開API — 既存インターフェース互換)
# ---------------------------------------------------------------------------
class LabelMaker:
    """
    ラベル作成用のインタラクティブGUIツール

    使い方:
        maker = LabelMaker(video_path="video.MOV", output_path="labels.csv")
        maker.run()
    """

    def __init__(
        self,
        video_path: str,
        output_path: str = None,
        fps_divisor: float = 1.0,
        target_fps: float | None = None,
    ):
        self.video = VideoHandler(video_path, fps_divisor, target_fps=target_fps)

        if output_path:
            out = Path(output_path)
        else:
            vp = Path(video_path)
            out = vp.parent / f"{vp.stem}_labels.csv"

        self.model = SceneDataModel(out, self.video.total_frames, self.video.fps)
        self.gui = AnnotationGUI(self.video, self.model)

        # 既存シーンがあればリストを初期化
        if self.model.play_scenes:
            self.gui._refresh_scene_list()

        print(f"\n{'=' * 60}")
        print(f"アノテーションツール")
        print(f"動画: {self.video.path.name}")
        print(f"総フレーム数: {self.video.total_frames} | FPS: {self.video.fps:.1f}")
        if self.video.frame_step > 1:
            print(
                f"target_fps: {self.video.effective_fps:.1f} "
                f"(frame_step={self.video.frame_step})"
            )
        print(f"既存シーン: {len(self.model.play_scenes)}")
        print(f"{'=' * 60}\n")

    def run(self):
        self.gui.root.mainloop()
