#!/usr/bin/env python3
"""
プレー検知モデル訓練スクリプト

正規化された骨格データから、プレーシーンを検知するLSTM/CNN-LSTMモデルを訓練します。

Usage:
    # デフォルト設定で訓練
    python scripts/train_play_classifier.py \
        --train-dirs data/video1 data/video2 \
        --val-dirs data/val_video1

    # LSTMモデルをカスタム設定で訓練
    python scripts/train_play_classifier.py \
        --train-dirs data/train \
        --val-dirs data/val \
        --model-type lstm \
        --hidden-size 256 \
        --num-layers 3 \
        --epochs 100 \
        --batch-size 64 \
        --lr 0.001

    # CNN-LSTMモデルを訓練
    python scripts/train_play_classifier.py \
        --train-dirs data/train \
        --val-dirs data/val \
        --model-type cnn_lstm \
        --cnn-channels 128 \
        --hidden-size 256
"""

import argparse
import sys
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.training_pipeline import TrainingPipeline
from src.training.config import (
    TrainingPipelineConfig,
    ModelConfig,
    DatasetConfig,
    OptimizerConfig,
    TrainingConfig
)
from src.core.exceptions import PipelineError


def parse_args():
    """コマンドライン引数をパース"""
    parser = argparse.ArgumentParser(
        description='プレー検知モデル訓練スクリプト',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # デフォルト設定で訓練（LSTMモデル）
  python scripts/train_play_classifier.py \\
      --train-dirs data/video1 data/video2 \\
      --val-dirs data/val_video1

  # カスタム設定でLSTMモデルを訓練
  python scripts/train_play_classifier.py \\
      --train-dirs data/train \\
      --val-dirs data/val \\
      --model-type lstm \\
      --hidden-size 256 \\
      --num-layers 3 \\
      --epochs 100 \\
      --batch-size 64 \\
      --lr 0.001 \\
      --device cuda

  # CNN-LSTMモデルを訓練
  python scripts/train_play_classifier.py \\
      --train-dirs data/train \\
      --val-dirs data/val \\
      --model-type cnn_lstm \\
      --cnn-channels 128 \\
      --hidden-size 256 \\
      --device mps
        """
    )

    # データセット設定
    data_group = parser.add_argument_group('データセット設定')
    data_group.add_argument(
        '--train-dirs',
        type=str,
        nargs='+',
        required=True,
        help='訓練用動画データディレクトリのリスト（複数指定可能）'
    )
    data_group.add_argument(
        '--val-dirs',
        type=str,
        nargs='+',
        default=None,
        help='検証用動画データディレクトリのリスト（複数指定可能）'
    )
    data_group.add_argument(
        '--csv-filename',
        type=str,
        default='original_pose_data.csv',
        help='CSVファイル名 (デフォルト: original_pose_data.csv)'
    )
    data_group.add_argument(
        '--label-filename',
        type=str,
        default='play_labels.csv',
        help='ラベルファイル名 (デフォルト: play_labels.csv)'
    )
    data_group.add_argument(
        '--sequence-length',
        type=int,
        default=30,
        help='シーケンス長（フレーム数） (デフォルト: 30)'
    )
    data_group.add_argument(
        '--stride',
        type=int,
        default=5,
        help='シーケンス抽出のストライド (デフォルト: 5)'
    )
    data_group.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='バッチサイズ (デフォルト: 32)'
    )
    data_group.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='データローダーのワーカー数 (デフォルト: 4)'
    )

    # モデル設定
    model_group = parser.add_argument_group('モデル設定')
    model_group.add_argument(
        '--model-type',
        type=str,
        choices=['lstm', 'cnn_lstm'],
        default='lstm',
        help='モデルタイプ (デフォルト: lstm)'
    )
    model_group.add_argument(
        '--hidden-size',
        type=int,
        default=128,
        help='隠れ層のサイズ (デフォルト: 128)'
    )
    model_group.add_argument(
        '--num-layers',
        type=int,
        default=2,
        help='LSTM/CNN-LSTMのレイヤー数 (デフォルト: 2)'
    )
    model_group.add_argument(
        '--dropout',
        type=float,
        default=0.3,
        help='ドロップアウト率 (デフォルト: 0.3)'
    )
    model_group.add_argument(
        '--use-attention',
        action='store_true',
        default=True,
        help='Attentionを使用（LSTMのみ） (デフォルト: True)'
    )
    model_group.add_argument(
        '--no-attention',
        dest='use_attention',
        action='store_false',
        help='Attentionを使用しない'
    )
    model_group.add_argument(
        '--cnn-channels',
        type=int,
        default=64,
        help='CNNのチャンネル数（CNN-LSTMのみ） (デフォルト: 64)'
    )

    # 最適化設定
    optim_group = parser.add_argument_group('最適化設定')
    optim_group.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='学習率 (デフォルト: 0.001)'
    )
    optim_group.add_argument(
        '--weight-decay',
        type=float,
        default=0.0,
        help='重み減衰 (デフォルト: 0.0)'
    )
    optim_group.add_argument(
        '--scheduler-patience',
        type=int,
        default=5,
        help='学習率スケジューラーのpatience (デフォルト: 5)'
    )
    optim_group.add_argument(
        '--scheduler-factor',
        type=float,
        default=0.5,
        help='学習率スケジューラーの減衰率 (デフォルト: 0.5)'
    )
    optim_group.add_argument(
        '--scheduler-min-lr',
        type=float,
        default=1e-6,
        help='学習率の最小値 (デフォルト: 1e-6)'
    )

    # 訓練設定
    train_group = parser.add_argument_group('訓練設定')
    train_group.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='エポック数 (デフォルト: 50)'
    )
    train_group.add_argument(
        '--save-every',
        type=int,
        default=10,
        help='チェックポイント保存間隔（エポック） (デフォルト: 10)'
    )
    train_group.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu', 'mps'],
        default='cuda',
        help='使用デバイス (デフォルト: cuda)'
    )
    train_group.add_argument(
        '--early-stopping',
        type=int,
        default=None,
        help='Early stoppingのpatience（エポック）'
    )
    train_group.add_argument(
        '--no-tensorboard',
        action='store_true',
        help='TensorBoardを無効化'
    )

    # 出力設定
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output/training',
        help='出力ディレクトリ (デフォルト: output/training)'
    )

    return parser.parse_args()


def main():
    """メイン処理"""
    args = parse_args()

    # 設定を作成
    model_config = ModelConfig(
        model_type=args.model_type,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_attention=args.use_attention,
        cnn_channels=args.cnn_channels
    )

    dataset_config = DatasetConfig(
        train_data_dirs=args.train_dirs,
        val_data_dirs=args.val_dirs,
        csv_filename=args.csv_filename,
        label_filename=args.label_filename,
        sequence_length=args.sequence_length,
        stride=args.stride,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    optimizer_config = OptimizerConfig(
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        scheduler_patience=args.scheduler_patience,
        scheduler_factor=args.scheduler_factor,
        scheduler_min_lr=args.scheduler_min_lr
    )

    training_config = TrainingConfig(
        epochs=args.epochs,
        save_every=args.save_every,
        device=args.device,
        use_tensorboard=not args.no_tensorboard,
        early_stopping_patience=args.early_stopping
    )

    pipeline_config = TrainingPipelineConfig(
        model=model_config,
        dataset=dataset_config,
        optimizer=optimizer_config,
        training=training_config,
        output_dir=args.output_dir
    )

    # パイプラインを作成
    pipeline = TrainingPipeline(pipeline_config)

    try:
        # 訓練実行
        results = pipeline.run()

        # 成功メッセージ
        print("\n" + "="*70)
        print("訓練が正常に完了しました")
        print("="*70)
        print(f"  Best Val F1: {results['best_val_f1']:.4f}")
        print(f"  Best Val Loss: {results['best_val_loss']:.4f}")
        print(f"  Total Epochs: {results['total_epochs']}")
        print(f"\n出力:")
        print(f"  Best Model: {results['best_model_path']}")
        print(f"  Final Model: {results['final_model_path']}")
        print(f"  History: {results['history_path']}")
        print(f"  Output Dir: {results['output_dir']}")
        print("="*70 + "\n")

        # TensorBoard起動方法を表示
        if not args.no_tensorboard:
            log_dir = Path(results['output_dir']) / 'logs'
            print("TensorBoardで学習過程を確認:")
            print(f"  tensorboard --logdir={log_dir}")
            print()

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
