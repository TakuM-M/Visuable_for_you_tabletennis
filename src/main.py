"""
メインスクリプト
動画からプレー中の区間を抽出する
"""
import argparse
from pathlib import Path
import sys

from utils.video_loader import VideoLoader
from detection.motion_detector import MotionDetector


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description='卓球動画からプレー中の区間を抽出する'
    )
    parser.add_argument(
        'input',
        type=str,
        help='入力動画ファイルのパス'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='output',
        help='出力ディレクトリ（デフォルト: output）'
    )
    parser.add_argument(
        '-t', '--threshold',
        type=int,
        default=25,
        help='動き検出の閾値（デフォルト: 25）'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='詳細な情報を表示'
    )

    args = parser.parse_args()

    # 入力ファイルチェック
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"エラー: 入力ファイルが見つかりません: {input_path}")
        sys.exit(1)

    # 出力ディレクトリ作成
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"入力動画: {input_path}")
    print(f"出力先: {output_dir}")

    # 動画読み込み
    try:
        with VideoLoader(str(input_path)) as video:
            info = video.get_info()

            if args.verbose:
                print(f"\n動画情報:")
                print(f"  解像度: {info['width']}x{info['height']}")
                print(f"  FPS: {info['fps']:.2f}")
                print(f"  フレーム数: {info['frame_count']}")
                print(f"  長さ: {info['duration']:.2f}秒")

            # TODO: ここに動き検出とプレー区間抽出のロジックを実装
            print("\n処理を開始します...")
            print("（現在はプロトタイプのため、実装が必要です）")

    except Exception as e:
        print(f"エラー: {e}")
        sys.exit(1)

    print("\n完了しました！")


if __name__ == "__main__":
    main()
