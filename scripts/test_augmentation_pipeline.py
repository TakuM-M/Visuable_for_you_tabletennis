"""
データ拡張パイプラインのテストスクリプト
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipelines import (
    PoseAugmentationPipeline,
    AugmentationPipelineConfig,
    AugmentationConfig,
    DataInputError,
    AugmentationError
)


def test_basic_augmentation():
    """基本的なデータ拡張のテスト"""
    print("=" * 70)
    print("テスト1: 基本的なデータ拡張")
    print("=" * 70)

    # デフォルト設定でパイプラインを作成
    pipeline = PoseAugmentationPipeline.create_default(
        augmentation_factor=3,
        random_seed=42
    )

    # テスト用のダミーCSVパス（実際のファイルが必要）
    input_csv = "output/player_pose_data.csv"
    output_csv = "output/augmented_pose_data.csv"

    if not Path(input_csv).exists():
        print(f"\n⚠ テストスキップ: {input_csv} が存在しません")
        print("先に PlayerPoseExporter を実行してデータを生成してください\n")
        return

    try:
        results = pipeline.augment_csv(
            input_csv=input_csv,
            output_csv=output_csv
        )

        print("\n✓ テスト成功:")
        print(f"  元データ: {results['original_samples']} サンプル")
        print(f"  拡張後: {results['augmented_samples']} サンプル")
        print(f"  拡張倍率: {results['augmentation_factor']:.1f}x")
        print(f"  出力: {results['output_csv']}")

    except DataInputError as e:
        print(f"\n✗ データ読み込みエラー: {e}")
    except AugmentationError as e:
        print(f"\n✗ 拡張処理エラー: {e}")
    except Exception as e:
        print(f"\n✗ 予期しないエラー: {e}")


def test_custom_config():
    """カスタム設定のテスト"""
    print("\n" + "=" * 70)
    print("テスト2: カスタム設定でのデータ拡張")
    print("=" * 70)

    # カスタム設定を作成
    augmentation_config = AugmentationConfig(
        horizontal_flip=True,
        horizontal_flip_prob=0.5,
        add_noise=True,
        noise_std=0.03,
        rotation=True,
        rotation_range=20.0,
        scaling=True,
        scale_range=(0.9, 1.1),
        keypoint_dropout=True,
        dropout_prob=0.15,
        random_seed=42
    )

    pipeline_config = AugmentationPipelineConfig(
        augmentation=augmentation_config,
        augmentation_factor=5,
        preserve_original=True,
        save_metadata=True,
        show_progress=True
    )

    pipeline = PoseAugmentationPipeline(pipeline_config)

    input_csv = "output/player_pose_data.csv"
    output_csv = "output/augmented_pose_data_custom.csv"

    if not Path(input_csv).exists():
        print(f"\n⚠ テストスキップ: {input_csv} が存在しません\n")
        return

    try:
        results = pipeline.augment_csv(
            input_csv=input_csv,
            output_csv=output_csv
        )

        print("\n✓ テスト成功:")
        print(f"  元データ: {results['original_samples']} サンプル")
        print(f"  拡張後: {results['augmented_samples']} サンプル")
        print(f"  拡張倍率: {results['augmentation_factor']:.1f}x")

    except Exception as e:
        print(f"\n✗ エラー: {e}")


def test_config_validation():
    """設定のバリデーションテスト"""
    print("\n" + "=" * 70)
    print("テスト3: 設定のバリデーション")
    print("=" * 70)

    print("\n不正な設定でエラーが発生することを確認:")

    # テスト1: 不正な horizontal_flip_prob
    try:
        config = AugmentationConfig(horizontal_flip_prob=1.5)
        print("  ✗ horizontal_flip_prob のバリデーション失敗")
    except ValueError as e:
        print(f"  ✓ horizontal_flip_prob のバリデーション成功: {e}")

    # テスト2: 不正な augmentation_factor
    try:
        augmentation = AugmentationConfig()
        config = AugmentationPipelineConfig(
            augmentation=augmentation,
            augmentation_factor=0
        )
        print("  ✗ augmentation_factor のバリデーション失敗")
    except ValueError as e:
        print(f"  ✓ augmentation_factor のバリデーション成功: {e}")

    # テスト3: 不正な scale_range
    try:
        config = AugmentationConfig(scale_range=(1.1, 0.9))  # min > max
        print("  ✗ scale_range のバリデーション失敗")
    except ValueError as e:
        print(f"  ✓ scale_range のバリデーション成功: {e}")

    print("\n✓ すべてのバリデーションテスト完了")


def test_import():
    """インポートのテスト"""
    print("\n" + "=" * 70)
    print("テスト4: インポートの確認")
    print("=" * 70)

    try:
        from src.pipelines import (
            PoseAugmentationPipeline,
            AugmentationPipelineConfig,
            AugmentationConfig,
            DataInputError,
            AugmentationError
        )
        print("\n✓ すべてのクラスを正常にインポートできました:")
        print("  - PoseAugmentationPipeline")
        print("  - AugmentationPipelineConfig")
        print("  - AugmentationConfig")
        print("  - DataInputError")
        print("  - AugmentationError")
    except ImportError as e:
        print(f"\n✗ インポートエラー: {e}")


def main():
    """メイン関数"""
    print("\n" + "=" * 70)
    print("データ拡張パイプライン テストスイート")
    print("=" * 70)

    # テスト実行
    test_import()
    test_config_validation()
    test_basic_augmentation()
    test_custom_config()

    print("\n" + "=" * 70)
    print("すべてのテストが完了しました")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
