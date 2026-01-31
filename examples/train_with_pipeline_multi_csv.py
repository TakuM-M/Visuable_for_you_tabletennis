"""
新しいTrainingPipelineを使った複数CSV学習の例

MultiCSVPoseDatasetを内部で使用する新しいパイプライン
"""
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipelines import TrainingPipeline, TrainingPipelineConfig
from src.pipelines import ModelConfig, DatasetConfig, OptimizerConfig, TrainingConfig


# ========================================
# 例1: デフォルト設定で学習（最もシンプル）
# ========================================
def example1_default():
    """
    デフォルト設定で学習
    わずか数行で複数動画からの学習が可能
    """
    print("=" * 70)
    print("例1: デフォルト設定で学習")
    print("=" * 70)

    # 訓練用動画のディレクトリリスト
    train_dirs = [
        'data/detect/sample_video_03_short',
        'data/detect/sample_video_04_short',
        'data/detect/sample_video_05_02',
    ]

    # 検証用動画のディレクトリリスト
    val_dirs = [
        'data/detect/sample_video_06_01',
    ]

    # パイプライン作成（これだけ！）
    pipeline = TrainingPipeline.create_default(
        train_data_dirs=train_dirs,
        val_data_dirs=val_dirs,
        output_dir='output/training_pipeline',
        device='cuda'  # 'cuda', 'cpu', 'mps'
    )

    # 学習実行（全自動）
    results = pipeline.run()

    # 結果表示
    print("\n学習完了！")
    print(f"  Best F1: {results['best_val_f1']:.4f}")
    print(f"  Best Model: {results['best_model_path']}")


# ========================================
# 例2: カスタム設定で学習
# ========================================
def example2_custom():
    """
    カスタム設定で学習
    各種パラメータを細かく調整
    """
    print("=" * 70)
    print("例2: カスタム設定で学習")
    print("=" * 70)

    train_dirs = [
        'data/detect/sample_video_03_short',
        'data/detect/sample_video_04_short',
        'data/detect/sample_video_05_02',
    ]

    val_dirs = [
        'data/detect/sample_video_06_01',
    ]

    # モデル設定
    model_config = ModelConfig(
        model_type='lstm',      # 'lstm' or 'cnn_lstm'
        hidden_size=256,        # より大きな隠れ層
        num_layers=3,           # より深いネットワーク
        dropout=0.4,
        use_attention=True
    )

    # データセット設定
    dataset_config = DatasetConfig(
        train_data_dirs=train_dirs,
        val_data_dirs=val_dirs,
        csv_filename='original_pose_data.csv',
        label_filename='play_labels.csv',
        sequence_length=45,     # より長いシーケンス
        stride=10,
        batch_size=16,          # より小さいバッチ
        num_workers=4
    )

    # 最適化器設定
    optimizer_config = OptimizerConfig(
        learning_rate=5e-4,     # より小さい学習率
        weight_decay=1e-5,      # L2正則化
        scheduler_patience=10,
        scheduler_factor=0.3
    )

    # 学習設定
    training_config = TrainingConfig(
        epochs=100,
        save_every=20,
        device='cuda',
        use_tensorboard=True,
        early_stopping_patience=20  # Early Stopping
    )

    # パイプライン設定を作成
    config = TrainingPipelineConfig(
        model=model_config,
        dataset=dataset_config,
        optimizer=optimizer_config,
        training=training_config,
        output_dir='output/training_custom'
    )

    # パイプライン実行
    pipeline = TrainingPipeline(config)
    results = pipeline.run()

    print("\n学習完了！")
    print(f"  Best F1: {results['best_val_f1']:.4f}")
    print(f"  Best Model: {results['best_model_path']}")


# ========================================
# 例3: 拡張データを使用
# ========================================
def example3_augmented_data():
    """
    拡張されたデータを使用
    csv_filenameを変更するだけ
    """
    print("=" * 70)
    print("例3: 拡張データを使用")
    print("=" * 70)

    train_dirs = [
        'data/detect/sample_video_03_short',
        'data/detect/sample_video_04_short',
        'data/detect/sample_video_05_02',
    ]

    val_dirs = [
        'data/detect/sample_video_06_01',
    ]

    # データセット設定（拡張データを使用）
    dataset_config = DatasetConfig(
        train_data_dirs=train_dirs,
        val_data_dirs=val_dirs,
        csv_filename='augment_pose_data.csv',  # 拡張データ
        label_filename='play_labels.csv',
        sequence_length=30,
        stride=5,
        batch_size=32,
        num_workers=4
    )

    # その他はデフォルト設定
    config = TrainingPipelineConfig(
        model=ModelConfig(),
        dataset=dataset_config,
        optimizer=OptimizerConfig(),
        training=TrainingConfig(device='cuda'),
        output_dir='output/training_augmented'
    )

    pipeline = TrainingPipeline(config)
    results = pipeline.run()

    print("\n学習完了！")
    print(f"  Best F1: {results['best_val_f1']:.4f}")


# ========================================
# 例4: CNN+LSTMモデルで学習
# ========================================
def example4_cnn_lstm():
    """
    CNN+LSTMモデルで学習
    より高精度が期待できるが学習時間が長い
    """
    print("=" * 70)
    print("例4: CNN+LSTMモデルで学習")
    print("=" * 70)

    train_dirs = [
        'data/detect/sample_video_03_short',
        'data/detect/sample_video_04_short',
        'data/detect/sample_video_05_02',
    ]

    val_dirs = [
        'data/detect/sample_video_06_01',
    ]

    # CNN+LSTM設定
    model_config = ModelConfig(
        model_type='cnn_lstm',  # CNN+LSTM
        cnn_channels=128,
        hidden_size=256,
        num_layers=2,
        dropout=0.3
    )

    config = TrainingPipelineConfig.create_default(
        train_data_dirs=train_dirs,
        val_data_dirs=val_dirs,
        output_dir='output/training_cnn_lstm',
        device='cuda'
    )

    # モデル設定を上書き
    config.model = model_config

    pipeline = TrainingPipeline(config)
    results = pipeline.run()

    print("\n学習完了！")
    print(f"  Best F1: {results['best_val_f1']:.4f}")


# ========================================
# メイン
# ========================================
def main():
    """実行例"""
    print("\n" + "=" * 70)
    print("TrainingPipeline (複数CSV対応) - 使用例")
    print("=" * 70 + "\n")

    print("以下のいずれかの関数を実行してください:\n")
    print("  example1_default()        - デフォルト設定で学習")
    print("  example2_custom()         - カスタム設定で学習")
    print("  example3_augmented_data() - 拡張データを使用")
    print("  example4_cnn_lstm()       - CNN+LSTMモデル")

    # どれか1つを実行（コメントを外す）
    # example1_default()
    # example2_custom()
    # example3_augmented_data()
    # example4_cnn_lstm()

    print("\n上記のコメントを外して実行してください")


if __name__ == "__main__":
    main()
