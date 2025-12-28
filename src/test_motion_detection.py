"""
動き検出のテストスクリプト
サンプル動画で動き検出を実行し、結果を分析する
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# matplotlibで日本語を表示できるように設定
plt.rcParams['font.family'] = ['Hiragino Sans']
plt.rcParams['axes.unicode_minus'] = False  # マイナス記号の文字化け対策

from utils.video_loader import VideoLoader
from detection.motion_detector import MotionDetector


def test_motion_detection(video_path: str, threshold: int = 25, min_area: int = 500, sample_rate: int = 1):
    """
    動き検出のテストを実行

    Args:
        video_path: 動画ファイルのパス
        threshold: 動き検出の閾値
        min_area: 動きと判定する最小面積
        sample_rate: フレームのサンプリングレート（1=全フレーム、2=2フレームに1回）
    """
    print(f"動画を読み込んでいます: {video_path}")

    # 動画読み込み
    video = VideoLoader(video_path)
    if not video.open():
        print("エラー: 動画ファイルを開けませんでした")
        sys.exit(1)

    info = video.get_info()
    print(f"\n動画情報:")
    print(f"  解像度: {info['width']}x{info['height']}")
    print(f"  FPS: {info['fps']:.2f}")
    print(f"  フレーム数: {info['frame_count']}")
    print(f"  長さ: {info['duration']:.2f}秒")

    # 動き検出器の初期化
    detector = MotionDetector(threshold=threshold, min_area=min_area)

    # 結果を格納するリスト
    motion_data = []
    frame_numbers = []
    timestamps = []

    print(f"\n動き検出を実行中...")
    print(f"  閾値: {threshold}")
    print(f"  最小面積: {min_area}")
    print(f"  サンプリングレート: {sample_rate}")

    frame_count = 0
    processed_count = 0

    # プログレスバー付きで処理
    with tqdm(total=info['frame_count'], desc="処理中") as pbar:
        while True:
            ret, frame = video.read_frame()
            if not ret:
                break

            # サンプリング
            if frame_count % sample_rate == 0:
                # 動き検出
                has_motion, motion_strength = detector.detect(frame)

                # 結果を保存
                motion_data.append(motion_strength)
                frame_numbers.append(frame_count)
                timestamps.append(frame_count / info['fps'])
                processed_count += 1

            frame_count += 1
            pbar.update(1)

    video.close()

    print(f"\n処理完了: {processed_count}フレームを分析しました")

    # 統計情報を表示
    motion_array = np.array(motion_data)
    print(f"\n動き検出の統計:")
    print(f"  平均: {motion_array.mean():.6f}")
    print(f"  最大: {motion_array.max():.6f}")
    print(f"  最小: {motion_array.min():.6f}")
    print(f"  標準偏差: {motion_array.std():.6f}")

    # 閾値を設定して動きのある区間を特定
    motion_threshold = motion_array.mean() + motion_array.std()
    motion_frames = motion_array > motion_threshold
    motion_ratio = motion_frames.sum() / len(motion_frames)

    print(f"\n動き検出の判定:")
    print(f"  動き閾値: {motion_threshold:.6f}")
    print(f"  動きがあるフレーム: {motion_frames.sum()}/{len(motion_frames)} ({motion_ratio*100:.1f}%)")

    # グラフ化
    print(f"\nグラフを生成しています...")
    plot_motion_graph(timestamps, motion_data, motion_threshold, info, video_path)

    return motion_data, timestamps


def plot_motion_graph(timestamps, motion_data, motion_threshold, video_info, video_path):
    """
    動きの強さをグラフ化

    Args:
        timestamps: タイムスタンプのリスト
        motion_data: 動きの強さのリスト
        motion_threshold: 動き判定の閾値
        video_info: 動画情報
        video_path: 動画ファイルのパス
    """
    plt.figure(figsize=(15, 6))

    # 動きの強さをプロット
    plt.subplot(2, 1, 1)
    plt.plot(timestamps, motion_data, linewidth=0.5, alpha=0.7)
    plt.axhline(y=motion_threshold, color='r', linestyle='--', label=f'閾値: {motion_threshold:.6f}')
    plt.xlabel('時間 (秒)')
    plt.ylabel('動きの強さ')
    plt.title(f'動き検出結果: {Path(video_path).name}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 動きの有無を二値化してプロット
    plt.subplot(2, 1, 2)
    motion_binary = [1 if m > motion_threshold else 0 for m in motion_data]
    plt.fill_between(timestamps, motion_binary, alpha=0.5, color='green')
    plt.xlabel('時間 (秒)')
    plt.ylabel('動きの検出')
    plt.title('プレー区間の推定（緑=動きあり）')
    plt.yticks([0, 1], ['待機', 'プレー中'])
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存
    output_path = Path('output') / f"motion_detection_{Path(video_path).stem}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"グラフを保存しました: {output_path}")

    # 表示はしない（CLI環境のため）
    plt.close()


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description='動き検出のテストを実行'
    )
    parser.add_argument(
        'input',
        type=str,
        help='入力動画ファイルのパス'
    )
    parser.add_argument(
        '-t', '--threshold',
        type=int,
        default=25,
        help='動き検出の閾値（デフォルト: 25）'
    )
    parser.add_argument(
        '-a', '--min-area',
        type=int,
        default=500,
        help='動きと判定する最小面積（デフォルト: 500）'
    )
    parser.add_argument(
        '-s', '--sample-rate',
        type=int,
        default=3,
        help='フレームのサンプリングレート（デフォルト: 3 = 3フレームに1回）'
    )

    args = parser.parse_args()

    # 入力ファイルチェック
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"エラー: 入力ファイルが見つかりません: {input_path}")
        sys.exit(1)

    # 動き検出テストを実行
    test_motion_detection(
        str(input_path),
        threshold=args.threshold,
        min_area=args.min_area,
        sample_rate=args.sample_rate
    )


if __name__ == "__main__":
    main()
