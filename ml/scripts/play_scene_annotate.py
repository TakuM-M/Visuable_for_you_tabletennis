#!/usr/bin/env python3
"""アノテーションツールの実行スクリプト

動画を再生しながらプレー中/プレー外のラベルを手動で付けるGUIツール

Usage:
    python scripts/play_scene_annotate.py data/raw/sample_video_01_01.MOV \
        -o data/proceed/sample_video_01_01/play_labels.csv \
        --fps-divisor 1

操作方法:
    - 'k' or スペース: 再生/一時停止
    - 's': 現在のフレームをプレー開始として記録
    - 'e': 現在のフレームをプレー終了として記録
    - 'd': 最後に追加したラベルを削除
    - 'l': 次のフレーム（一時停止中）
    - 'j': 前のフレーム（一時停止中）
    - 'q': 保存して終了
"""
import argparse
import sys
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.annotation.label_maker import LabelMaker


def main():
    parser = argparse.ArgumentParser(
        description='卓球動画のプレーシーンアノテーションツール',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
操作方法:
  k/スペース : 再生/一時停止
  s          : プレー開始フレームを記録
  e          : プレー終了フレームを記録
  d          : 最後のシーンを削除
  l          : 次のフレーム（一時停止中）
  j          : 前のフレーム（一時停止中）
  q          : 保存して終了

使用例:
  python scripts/annotate.py data/raw/video.MOV
  python scripts/annotate.py data/raw/video.MOV -o data/labels/video_labels.csv
  python scripts/annotate.py data/raw/video.MOV --fps-divisor 2  # 半分速度で再生
        """
    )

    parser.add_argument(
        'video',
        type=str,
        help='動画ファイルのパス'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='出力CSVパス（デフォルト: 動画名_labels.csv）'
    )
    parser.add_argument(
        '--fps-divisor',
        type=float,
        default=1.0,
        help='表示速度の倍率（2なら半分速度、0.5なら2倍速）'
    )
    parser.add_argument(
        '--target-fps',
        type=float,
        default=15.0,
        help='アノテーション対象のFPS（デフォルト: 15.0、ポーズ抽出と同じ値にする）'
    )

    args = parser.parse_args()

    try:
        maker = LabelMaker(
            video_path=args.video,
            output_path=args.output,
            fps_divisor=args.fps_divisor,
            target_fps=args.target_fps,
        )
        maker.run()
    except KeyboardInterrupt:
        print("\n\nユーザーによって中断されました")
        sys.exit(1)
    except Exception as e:
        print(f"\nエラーが発生しました: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
