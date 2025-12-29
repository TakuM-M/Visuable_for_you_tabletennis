"""
動画からアノテーション用のフレームを抽出するスクリプト
卓球台検出モデルの学習用データセット作成
"""
import argparse
import sys
from pathlib import Path
import cv2
from tqdm import tqdm

# プロジェクトのルートディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.video_loader import VideoLoader


def extract_frames(
    video_path: str,
    output_dir: str = "data/annotations/images",
    interval: int = 30,
    max_frames: int = 200,
    resize_width: int = 640
):
    """
    動画からフレームを抽出してアノテーション用に保存

    Args:
        video_path: 入力動画ファイルのパス
        output_dir: 出力ディレクトリ
        interval: フレーム抽出間隔（フレーム数）
        max_frames: 最大抽出フレーム数
        resize_width: リサイズ後の幅（アスペクト比は維持）
    """
    print(f"=== フレーム抽出ツール ===")
    print(f"動画: {video_path}")
    print(f"抽出間隔: {interval}フレーム")
    print(f"最大フレーム数: {max_frames}")
    print(f"リサイズ幅: {resize_width}px\n")

    # 出力ディレクトリの作成
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 動画読み込み
    video = VideoLoader(video_path)
    if not video.open():
        print("エラー: 動画ファイルを開けませんでした")
        sys.exit(1)

    info = video.get_info()
    print(f"動画情報:")
    print(f"  解像度: {info['width']}x{info['height']}")
    print(f"  FPS: {info['fps']:.2f}")
    print(f"  フレーム数: {info['frame_count']}")
    print(f"  長さ: {info['duration']:.2f}秒\n")

    # アスペクト比を維持してリサイズ比率を計算
    aspect_ratio = info['height'] / info['width']
    resize_height = int(resize_width * aspect_ratio)

    # ビデオファイル名（拡張子なし）
    video_name = Path(video_path).stem

    frame_count = 0
    extracted_count = 0

    print("フレーム抽出中...")
    with tqdm(total=min(info['frame_count'], max_frames * interval), desc="処理中") as pbar:
        while extracted_count < max_frames:
            ret, frame = video.read_frame()
            if not ret:
                break

            # 指定間隔でフレームを抽出
            if frame_count % interval == 0:
                # リサイズ
                resized_frame = cv2.resize(
                    frame,
                    (resize_width, resize_height),
                    interpolation=cv2.INTER_AREA
                )

                # ファイル名: video_name_frame_XXXXX.jpg
                output_filename = f"{video_name}_frame_{frame_count:05d}.jpg"
                output_filepath = output_path / output_filename

                # 保存
                cv2.imwrite(str(output_filepath), resized_frame)
                extracted_count += 1

                pbar.set_postfix({"抽出済み": extracted_count})

            frame_count += 1
            pbar.update(1)

    video.close()

    print(f"\n完了:")
    print(f"  抽出フレーム数: {extracted_count}")
    print(f"  保存先: {output_path}")
    print(f"\n次のステップ:")
    print(f"  1. Label Studio または CVAT を使ってアノテーション")
    print(f"  2. アノテーション結果をYOLO形式に変換")
    print(f"  3. train/valに分割してモデル学習")


def extract_from_multiple_videos(
    video_dir: str,
    output_dir: str = "data/annotations/images",
    interval: int = 30,
    max_frames_per_video: int = 50,
    resize_width: int = 640
):
    """
    複数の動画からフレームを抽出

    Args:
        video_dir: 動画が格納されているディレクトリ
        output_dir: 出力ディレクトリ
        interval: フレーム抽出間隔（フレーム数）
        max_frames_per_video: 1動画あたりの最大抽出フレーム数
        resize_width: リサイズ後の幅（アスペクト比は維持）
    """
    video_dir_path = Path(video_dir)
    video_extensions = ['.mp4', '.avi', '.mov', '.MP4', '.AVI', '.MOV']

    # 動画ファイルを検索
    video_files = []
    for ext in video_extensions:
        video_files.extend(video_dir_path.glob(f"*{ext}"))

    if not video_files:
        print(f"エラー: {video_dir} に動画ファイルが見つかりませんでした")
        sys.exit(1)

    print(f"見つかった動画: {len(video_files)}本\n")

    total_extracted = 0
    for video_file in video_files:
        print(f"\n{'='*60}")
        print(f"処理中: {video_file.name}")
        print(f"{'='*60}")

        extract_frames(
            str(video_file),
            output_dir=output_dir,
            interval=interval,
            max_frames=max_frames_per_video,
            resize_width=resize_width
        )

        # 実際に抽出されたフレーム数を確認
        output_path = Path(output_dir)
        current_count = len(list(output_path.glob(f"{video_file.stem}_frame_*.jpg")))
        total_extracted += current_count

    print(f"\n{'='*60}")
    print(f"全体の抽出完了:")
    print(f"  処理動画数: {len(video_files)}")
    print(f"  総抽出フレーム数: {total_extracted}")
    print(f"  保存先: {output_dir}")
    print(f"{'='*60}")


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description='動画からアノテーション用フレームを抽出'
    )
    parser.add_argument(
        'input',
        type=str,
        help='入力動画ファイルまたはディレクトリのパス'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='data/annotations/images',
        help='出力ディレクトリ（デフォルト: data/annotations/images）'
    )
    parser.add_argument(
        '-i', '--interval',
        type=int,
        default=30,
        help='フレーム抽出間隔（デフォルト: 30）'
    )
    parser.add_argument(
        '-m', '--max-frames',
        type=int,
        default=200,
        help='最大抽出フレーム数（デフォルト: 200）'
    )
    parser.add_argument(
        '-w', '--width',
        type=int,
        default=640,
        help='リサイズ後の幅（デフォルト: 640）'
    )

    args = parser.parse_args()

    # 入力パスチェック
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"エラー: 入力パスが見つかりません: {input_path}")
        sys.exit(1)

    # ディレクトリの場合は複数動画処理
    if input_path.is_dir():
        extract_from_multiple_videos(
            str(input_path),
            output_dir=args.output,
            interval=args.interval,
            max_frames_per_video=args.max_frames,
            resize_width=args.width
        )
    else:
        # 単一動画処理
        extract_frames(
            str(input_path),
            output_dir=args.output,
            interval=args.interval,
            max_frames=args.max_frames,
            resize_width=args.width
        )


if __name__ == "__main__":
    main()
