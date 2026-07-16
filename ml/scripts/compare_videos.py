#!/usr/bin/env python3
"""動画比較ビューア

元動画とYOLOポーズ検出動画を左右に並べて再生し、
予測確率(probability)をリアルタイムでグラフ表示する。

2つの動画はFPSが異なる場合でも時間ベースで同期する。
predictions.csvのフレーム番号はposes_csvのtimestampで時間に変換する。

Usage:
    python scripts/compare_videos.py data/sample_video_07_01 \
        --original data/sample_video_07_01/data_raw_01_test_sample_video_07_01_MOV_highlights.mp4

    # 元動画を自動検出（_poses.mp4 でないmp4を使用）
    python scripts/compare_videos.py data/sample_video_07_01

操作方法:
    - スペース/k : 再生/一時停止
    - l          : 次のフレーム（一時停止中、poses基準）
    - j          : 前のフレーム（一時停止中、poses基準）
    - →          : 2秒先へスキップ
    - ←          : 2秒前へスキップ
    - q          : 終了
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def find_videos(result_dir: Path, original_path: str | None = None):
    """結果ディレクトリから元動画とposes動画を探す"""
    name = result_dir.name

    # poses動画
    poses_video = result_dir / f"{name}_poses.mp4"
    if not poses_video.exists():
        raise FileNotFoundError(f"Poses動画が見つかりません: {poses_video}")

    # 元動画
    if original_path:
        orig_video = Path(original_path)
    else:
        # {name}.MOV or {name}.mp4 を探す
        candidates = [
            result_dir / f"{name}.MOV",
            result_dir / f"{name}.mov",
            result_dir / f"{name}.mp4",
        ]
        orig_video = None
        for c in candidates:
            if c.exists():
                orig_video = c
                break
        if orig_video is None:
            raise FileNotFoundError(
                f"元動画が見つかりません（{name}.MOV/.mp4）。--original で指定してください"
            )
        print(f"元動画を自動検出: {orig_video.name}")

    if not orig_video.exists():
        raise FileNotFoundError(f"元動画が見つかりません: {orig_video}")

    return orig_video, poses_video


def load_predictions(result_dir: Path) -> pd.DataFrame | None:
    """predictions.csvを読み込む"""
    name = result_dir.name
    csv_path = result_dir / f"{name}_predictions.csv"
    if not csv_path.exists():
        print(f"警告: predictions CSVが見つかりません: {csv_path}")
        return None
    df = pd.read_csv(csv_path)
    return df


def build_frame_to_time_map(result_dir: Path) -> dict[int, float] | None:
    """poses.csvからフレーム番号→タイムスタンプのマッピングを構築"""
    name = result_dir.name
    csv_path = result_dir / f"{name}_poses.csv"
    if not csv_path.exists():
        print(f"警告: poses CSVが見つかりません: {csv_path}")
        return None
    df = pd.read_csv(csv_path, usecols=["frame", "timestamp"])
    # フレームごとにユニークなタイムスタンプを取得
    frame_time = df.drop_duplicates("frame").set_index("frame")["timestamp"].to_dict()
    return frame_time


def estimate_source_fps(frame_to_time: dict[int, float]) -> float:
    """poses.csvのframe/timestampからソース動画のFPSを推定"""
    frames = sorted(frame_to_time.keys())
    if len(frames) < 2:
        return 30.0
    # 連続フレームペアからFPSを推定
    diffs = []
    for i in range(min(50, len(frames) - 1)):
        df = frames[i + 1] - frames[i]
        dt = frame_to_time[frames[i + 1]] - frame_to_time[frames[i]]
        if dt > 0:
            diffs.append(df / dt)
    return float(np.median(diffs)) if diffs else 30.0


def pred_frame_to_time(
    pred_frame: int,
    frame_to_time: dict[int, float] | None,
    source_fps: float,
) -> float:
    """predictionsのフレーム番号を時間(秒)に変換"""
    if frame_to_time is not None and pred_frame in frame_to_time:
        return frame_to_time[pred_frame]
    # マップにない場合はFPSから推定
    return pred_frame / source_fps


def draw_probability_graph(
    width: int,
    height: int,
    predictions: pd.DataFrame,
    current_time: float,
    pred_times: np.ndarray,
    window_sec: float = 10.0,
) -> np.ndarray:
    """現在時刻付近のprobabilityグラフを描画"""
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (30, 30, 30)

    if predictions is None or predictions.empty:
        return canvas

    margin_left = 50
    margin_right = 20
    margin_top = 30
    margin_bottom = 40
    graph_w = width - margin_left - margin_right
    graph_h = height - margin_top - margin_bottom

    # グラフ領域の枠
    cv2.rectangle(
        canvas,
        (margin_left, margin_top),
        (margin_left + graph_w, margin_top + graph_h),
        (80, 80, 80),
        1,
    )

    # Y軸ラベル (0.0 ~ 1.0)
    for i in range(11):
        y_val = i / 10.0
        y_px = margin_top + graph_h - int(y_val * graph_h)
        cv2.line(
            canvas,
            (margin_left - 5, y_px),
            (margin_left + graph_w, y_px),
            (60, 60, 60),
            1,
        )
        if i % 2 == 0:
            cv2.putText(
                canvas,
                f"{y_val:.1f}",
                (5, y_px + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (180, 180, 180),
                1,
            )

    # 閾値ライン (0.5)
    threshold_y = margin_top + graph_h - int(0.5 * graph_h)
    cv2.line(
        canvas,
        (margin_left, threshold_y),
        (margin_left + graph_w, threshold_y),
        (0, 200, 200),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "0.5",
        (margin_left + graph_w + 2, threshold_y + 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.3,
        (0, 200, 200),
        1,
    )

    # 表示範囲（時間ベース）
    half_window = window_sec / 2
    time_min = current_time - half_window
    time_max = current_time + half_window

    # データをフィルタ
    mask = (pred_times >= time_min) & (pred_times <= time_max)
    visible_idx = np.where(mask)[0]

    if len(visible_idx) == 0:
        return canvas

    # probabilityを描画
    prev_pt = None
    for idx in visible_idx:
        t = pred_times[idx]
        prob = float(predictions.iloc[idx]["probability"])
        pred = int(predictions.iloc[idx]["prediction"])

        x_px = margin_left + int((t - time_min) / (time_max - time_min) * graph_w)
        y_px = margin_top + graph_h - int(np.clip(prob, 0, 1) * graph_h)

        color = (0, 200, 0) if pred == 1 else (0, 0, 200)

        if prev_pt is not None:
            cv2.line(canvas, prev_pt, (x_px, y_px), color, 1, cv2.LINE_AA)
        prev_pt = (x_px, y_px)

    # 現在時刻の縦線
    cur_x = margin_left + int(
        (current_time - time_min) / (time_max - time_min) * graph_w
    )
    cur_x = int(np.clip(cur_x, margin_left, margin_left + graph_w))
    cv2.line(
        canvas,
        (cur_x, margin_top),
        (cur_x, margin_top + graph_h),
        (255, 255, 0),
        2,
    )

    # 現在時刻に最も近いpredictionの情報を表示
    nearest_idx = int(np.argmin(np.abs(pred_times - current_time)))
    nearest_t = pred_times[nearest_idx]
    if abs(nearest_t - current_time) < 1.0:
        cur_prob = float(predictions.iloc[nearest_idx]["probability"])
        cur_pred = int(predictions.iloc[nearest_idx]["prediction"])
        cur_frame = int(predictions.iloc[nearest_idx]["frame"])
        pred_label = "Play" if cur_pred == 1 else "No Play"
        info_text = (
            f"Time:{current_time:.2f}s  "
            f"SrcFrame:{cur_frame}  "
            f"Prob:{cur_prob:.4f}  "
            f"Pred:{pred_label}"
        )
        text_color = (0, 255, 0) if cur_pred == 1 else (0, 100, 255)
    else:
        info_text = f"Time:{current_time:.2f}s  (no prediction data)"
        text_color = (180, 180, 180)

    cv2.putText(
        canvas,
        info_text,
        (margin_left, height - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        text_color,
        1,
        cv2.LINE_AA,
    )

    # タイトル
    cv2.putText(
        canvas,
        "Prediction Probability",
        (margin_left, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    return canvas


def main():
    parser = argparse.ArgumentParser(
        description="元動画とYOLOポーズ検出動画の比較ビューア（時間同期）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("result_dir", type=str, help="結果ディレクトリのパス")
    parser.add_argument(
        "--original", type=str, default=None, help="元動画のパス（省略時は自動検出）"
    )
    parser.add_argument(
        "--width", type=int, default=640, help="各動画の表示幅（デフォルト: 640）"
    )
    parser.add_argument(
        "--graph-height",
        type=int,
        default=200,
        help="グラフの高さ（デフォルト: 200）",
    )
    parser.add_argument(
        "--window-sec",
        type=float,
        default=10.0,
        help="グラフに表示する時間幅（秒、デフォルト: 10.0）",
    )

    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    if not result_dir.is_dir():
        print(f"エラー: ディレクトリが存在しません: {result_dir}", file=sys.stderr)
        sys.exit(1)

    orig_path, poses_path = find_videos(result_dir, args.original)
    predictions = load_predictions(result_dir)
    frame_to_time = build_frame_to_time_map(result_dir)

    # ソース動画のFPSを推定（poses.csvから）
    source_fps = estimate_source_fps(frame_to_time) if frame_to_time else 30.0
    print(f"ソース動画の推定FPS: {source_fps:.2f}")

    # predictions の各フレームに対応する時間を事前計算
    pred_times = None
    if predictions is not None:
        pred_times = np.array(
            [
                pred_frame_to_time(int(f), frame_to_time, source_fps)
                for f in predictions["frame"]
            ]
        )

    cap_orig = cv2.VideoCapture(str(orig_path))
    cap_poses = cv2.VideoCapture(str(poses_path))

    if not cap_orig.isOpened():
        print(f"エラー: 元動画を開けません: {orig_path}", file=sys.stderr)
        sys.exit(1)
    if not cap_poses.isOpened():
        print(f"エラー: Poses動画を開けません: {poses_path}", file=sys.stderr)
        sys.exit(1)

    fps_orig = cap_orig.get(cv2.CAP_PROP_FPS)
    fps_poses = cap_poses.get(cv2.CAP_PROP_FPS)
    total_orig = int(cap_orig.get(cv2.CAP_PROP_FRAME_COUNT))
    total_poses = int(cap_poses.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_orig = total_orig / fps_orig if fps_orig > 0 else 0
    duration_poses = total_poses / fps_poses if fps_poses > 0 else 0

    print(
        f"元動画   : {orig_path.name} ({total_orig} frames, {fps_orig:.1f} fps, {duration_orig:.1f}s)"
    )
    print(
        f"Poses動画: {poses_path.name} ({total_poses} frames, {fps_poses:.1f} fps, {duration_poses:.1f}s)"
    )
    if predictions is not None:
        print(
            f"Predictions: {len(predictions)} rows, "
            f"frames {predictions['frame'].min()}-{predictions['frame'].max()}, "
            f"time {pred_times.min():.1f}-{pred_times.max():.1f}s"
        )

    display_w = args.width
    paused = False
    # posesの動画フレームインデックスをマスターとする
    poses_frame_idx = 0
    delay = int(1000 / fps_poses) if fps_poses > 0 else 66

    window_name = "Video Comparison"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    def get_current_time(p_idx: int) -> float:
        """posesフレームインデックスから現在時刻を算出"""
        return p_idx / fps_poses

    def seek_and_read(current_time: float, p_idx: int):
        """時間ベースで両動画のフレームを取得"""
        # poses動画: フレームインデックスで直接シーク
        cap_poses.set(cv2.CAP_PROP_POS_FRAMES, p_idx)
        ret_p, frame_p = cap_poses.read()

        # 元動画: 時間から対応フレームを計算してシーク
        orig_frame_idx = int(current_time * fps_orig)
        orig_frame_idx = min(orig_frame_idx, total_orig - 1)
        cap_orig.set(cv2.CAP_PROP_POS_FRAMES, orig_frame_idx)
        ret_o, frame_o = cap_orig.read()

        return ret_o, frame_o, ret_p, frame_p, orig_frame_idx

    while True:
        current_time = get_current_time(poses_frame_idx)
        ret_o, frame_o, ret_p, frame_p, orig_fidx = seek_and_read(
            current_time, poses_frame_idx
        )

        if not ret_o or not ret_p:
            print("動画の末尾に到達しました")
            paused = True
            key = cv2.waitKey(0) & 0xFF
            if key == ord("q"):
                break
            continue

        # リサイズ
        h_orig, w_orig = frame_o.shape[:2]
        h_poses, w_poses = frame_p.shape[:2]
        scale_orig = display_w / w_orig
        scale_poses = display_w / w_poses
        display_h = int(max(h_orig * scale_orig, h_poses * scale_poses))

        resized_orig = cv2.resize(frame_o, (display_w, display_h))
        resized_poses = cv2.resize(frame_p, (display_w, display_h))

        # ラベル描画
        time_str = f"{current_time:.2f}s"
        cv2.putText(
            resized_orig,
            f"Original (frame {orig_fidx}, {time_str})",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            resized_poses,
            f"YOLO Poses (frame {poses_frame_idx}, {time_str})",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        # 左右結合
        combined = np.hstack([resized_orig, resized_poses])

        # グラフ描画
        graph = draw_probability_graph(
            combined.shape[1],
            args.graph_height,
            predictions,
            current_time,
            pred_times,
            args.window_sec,
        )

        # 上下結合
        final = np.vstack([combined, graph])

        cv2.imshow(window_name, final)

        key = cv2.waitKey(0 if paused else delay) & 0xFF

        if key == ord("q"):
            break
        elif key == ord(" ") or key == ord("k"):
            paused = not paused
        elif key == ord("l") and paused:
            poses_frame_idx = min(poses_frame_idx + 1, total_poses - 1)
        elif key == ord("j") and paused:
            poses_frame_idx = max(poses_frame_idx - 1, 0)
        elif key == 3:  # → 矢印キー: 2秒先
            skip_frames = int(2.0 * fps_poses)
            poses_frame_idx = min(poses_frame_idx + skip_frames, total_poses - 1)
        elif key == 2:  # ← 矢印キー: 2秒前
            skip_frames = int(2.0 * fps_poses)
            poses_frame_idx = max(poses_frame_idx - skip_frames, 0)
        else:
            if not paused:
                poses_frame_idx += 1

    cap_orig.release()
    cap_poses.release()
    cv2.destroyAllWindows()
    print("終了しました")


if __name__ == "__main__":
    main()
