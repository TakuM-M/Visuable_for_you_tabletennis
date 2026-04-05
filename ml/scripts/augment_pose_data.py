#!/usr/bin/env python3
"""
姿勢データ拡張スクリプト

CSVフォーマットの正規化された骨格データに対してデータ拡張を適用します。
コマンドラインから簡単に実行できるスクリプトです。

Usage:
    python scripts/augment_pose_data.py \
        --input data/pose_data.csv \
        --output data/augmented_pose_data.csv \
        --factor 5

    # カスタム設定で実行
    python scripts/augment_pose_data.py \
        --input data/pose_data.csv \
        --output data/augmented_pose_data.csv \
        --factor 10 \
        --flip-prob 0.7 \
        --noise-std 0.03 \
        --rotation-range 20 \
        --no-metadata
"""

import argparse
import sys
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.pose_augmentation import PoseAugmentationPipeline
from src.training.config import AugmentationPipelineConfig, AugmentationConfig
from src.training.exceptions import PipelineError


def parse_args():
    """コマンドライン引数をパース"""
    parser = argparse.ArgumentParser(
        description='姿勢データ拡張スクリプト',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # デフォルト設定で5倍に拡張
  python scripts/augment_pose_data.py -i data/pose.csv -o data/aug_pose.csv

  # 10倍に拡張、カスタム設定
  python scripts/augment_pose_data.py -i data/pose.csv -o data/aug_pose.csv \\
      --factor 10 --flip-prob 0.7 --noise-std 0.03

  # 元データを保持せず拡張のみ出力
  python scripts/augment_pose_data.py -i data/pose.csv -o data/aug_pose.csv \\
      --factor 5 --no-preserve-original
        """
    )

    # 必須引数
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='入力CSVファイルのパス'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='出力CSVファイルのパス'
    )

    # 基本設定
    parser.add_argument(
        '-f', '--factor',
        type=int,
        default=5,
        help='データ拡張倍率 (デフォルト: 5)'
    )
    parser.add_argument(
        '--preserve-original',
        action='store_true',
        default=True,
        help='元データも出力に含める (デフォルト: True)'
    )
    parser.add_argument(
        '--no-preserve-original',
        dest='preserve_original',
        action='store_false',
        help='元データを出力に含めない'
    )

    # データ拡張設定
    aug_group = parser.add_argument_group('データ拡張設定')

    # 左右反転
    aug_group.add_argument(
        '--horizontal-flip',
        action='store_true',
        default=True,
        help='左右反転を有効化 (デフォルト: True)'
    )
    aug_group.add_argument(
        '--no-horizontal-flip',
        dest='horizontal_flip',
        action='store_false',
        help='左右反転を無効化'
    )
    aug_group.add_argument(
        '--flip-prob',
        type=float,
        default=0.5,
        help='左右反転の確率 (デフォルト: 0.5)'
    )

    # ノイズ
    aug_group.add_argument(
        '--add-noise',
        action='store_true',
        default=True,
        help='ガウシアンノイズを有効化 (デフォルト: True)'
    )
    aug_group.add_argument(
        '--no-add-noise',
        dest='add_noise',
        action='store_false',
        help='ガウシアンノイズを無効化'
    )
    aug_group.add_argument(
        '--noise-std',
        type=float,
        default=0.02,
        help='ノイズの標準偏差 (デフォルト: 0.02)'
    )

    # 回転
    aug_group.add_argument(
        '--rotation',
        action='store_true',
        default=True,
        help='回転を有効化 (デフォルト: True)'
    )
    aug_group.add_argument(
        '--no-rotation',
        dest='rotation',
        action='store_false',
        help='回転を無効化'
    )
    aug_group.add_argument(
        '--rotation-range',
        type=float,
        default=15.0,
        help='回転の範囲（度） (デフォルト: 15.0)'
    )

    # スケーリング
    aug_group.add_argument(
        '--scaling',
        action='store_true',
        default=False,
        help='スケーリングを有効化 (デフォルト: False)'
    )
    aug_group.add_argument(
        '--scale-min',
        type=float,
        default=0.9,
        help='スケーリングの最小値 (デフォルト: 0.9)'
    )
    aug_group.add_argument(
        '--scale-max',
        type=float,
        default=1.1,
        help='スケーリングの最大値 (デフォルト: 1.1)'
    )

    # ドロップアウト
    aug_group.add_argument(
        '--keypoint-dropout',
        action='store_true',
        default=False,
        help='関節ドロップアウトを有効化 (デフォルト: False)'
    )
    aug_group.add_argument(
        '--dropout-prob',
        type=float,
        default=0.1,
        help='ドロップアウトの確率 (デフォルト: 0.1)'
    )

    # その他
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='ランダムシード (デフォルト: None)'
    )
    parser.add_argument(
        '--metadata',
        type=str,
        default=None,
        help='メタデータ出力パス (デフォルト: 自動生成)'
    )
    parser.add_argument(
        '--no-metadata',
        action='store_true',
        help='メタデータを保存しない'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='プログレスバーを非表示にする'
    )

    return parser.parse_args()


def main():
    """メイン処理"""
    args = parse_args()

    # 設定を作成
    augmentation_config = AugmentationConfig(
        horizontal_flip=args.horizontal_flip,
        horizontal_flip_prob=args.flip_prob,
        add_noise=args.add_noise,
        noise_std=args.noise_std,
        rotation=args.rotation,
        rotation_range=args.rotation_range,
        scaling=args.scaling,
        scale_range=(args.scale_min, args.scale_max),
        keypoint_dropout=args.keypoint_dropout,
        dropout_prob=args.dropout_prob,
        random_seed=args.seed
    )

    pipeline_config = AugmentationPipelineConfig(
        augmentation=augmentation_config,
        augmentation_factor=args.factor,
        preserve_original=args.preserve_original,
        save_metadata=not args.no_metadata,
        show_progress=not args.quiet
    )

    # パイプラインを作成
    pipeline = PoseAugmentationPipeline(pipeline_config)

    try:
        # データ拡張実行
        results = pipeline.augment_csv(
            input_csv=args.input,
            output_csv=args.output,
            output_metadata=args.metadata
        )

        # 成功メッセージ
        print("\n✓ データ拡張が正常に完了しました")
        print(f"  元データ: {results['original_samples']} サンプル")
        print(f"  拡張後: {results['augmented_samples']} サンプル")
        print(f"  拡張倍率: {results['augmentation_factor']:.1f}x")

        return 0

    except PipelineError as e:
        print(f"\n✗ エラー: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\n✗ 予期しないエラーが発生しました: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
