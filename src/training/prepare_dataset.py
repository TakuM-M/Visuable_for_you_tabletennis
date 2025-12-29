"""
アノテーション済みデータをYOLOv11形式のデータセットに変換
train/val分割とデータセット構造の準備
"""
import argparse
import shutil
import random
from pathlib import Path
from typing import Tuple


def split_dataset(
    images_dir: str,
    labels_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[int, int]:
    """
    画像とラベルをtrain/valに分割

    Args:
        images_dir: アノテーション済み画像ディレクトリ
        labels_dir: ラベルディレクトリ（YOLO形式）
        output_dir: 出力ディレクトリ（table_dataset）
        train_ratio: 学習データの割合（デフォルト: 0.8）
        seed: ランダムシード

    Returns:
        (train_count, val_count): 学習データ数、検証データ数
    """
    print(f"=== データセット準備ツール ===")
    print(f"画像ディレクトリ: {images_dir}")
    print(f"ラベルディレクトリ: {labels_dir}")
    print(f"出力ディレクトリ: {output_dir}")
    print(f"学習/検証比率: {train_ratio:.1%} / {(1-train_ratio):.1%}\n")

    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    output_path = Path(output_dir)

    # 画像ファイル一覧を取得
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(images_path.glob(f"*{ext}"))

    if not image_files:
        print(f"エラー: {images_dir} に画像ファイルが見つかりませんでした")
        return 0, 0

    # 対応するラベルファイルがあるものだけを使用
    valid_pairs = []
    for img_file in image_files:
        label_file = labels_path / f"{img_file.stem}.txt"
        if label_file.exists():
            valid_pairs.append((img_file, label_file))
        else:
            print(f"警告: ラベルファイルが見つかりません: {label_file.name}")

    if not valid_pairs:
        print("エラー: 有効な画像-ラベルペアが見つかりませんでした")
        return 0, 0

    print(f"有効なデータ数: {len(valid_pairs)}\n")

    # ランダムシャッフル
    random.seed(seed)
    random.shuffle(valid_pairs)

    # train/val分割
    split_idx = int(len(valid_pairs) * train_ratio)
    train_pairs = valid_pairs[:split_idx]
    val_pairs = valid_pairs[split_idx:]

    print(f"学習データ: {len(train_pairs)}")
    print(f"検証データ: {len(val_pairs)}\n")

    # 出力ディレクトリ作成
    train_images_dir = output_path / "train" / "images"
    train_labels_dir = output_path / "train" / "labels"
    val_images_dir = output_path / "val" / "images"
    val_labels_dir = output_path / "val" / "labels"

    for directory in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    # ファイルコピー
    print("ファイルコピー中...")

    # 学習データ
    for img_file, label_file in train_pairs:
        shutil.copy2(img_file, train_images_dir / img_file.name)
        shutil.copy2(label_file, train_labels_dir / label_file.name)

    # 検証データ
    for img_file, label_file in val_pairs:
        shutil.copy2(img_file, val_images_dir / img_file.name)
        shutil.copy2(label_file, val_labels_dir / label_file.name)

    print(f"\n完了:")
    print(f"  学習データ: {train_images_dir}")
    print(f"  検証データ: {val_images_dir}")
    print(f"\n次のステップ:")
    print(f"  python src/training/train_table_detector.py")

    return len(train_pairs), len(val_pairs)


def validate_yolo_labels(labels_dir: str) -> bool:
    """
    YOLOラベル形式を検証

    Args:
        labels_dir: ラベルディレクトリ

    Returns:
        検証成功: True、失敗: False
    """
    labels_path = Path(labels_dir)
    label_files = list(labels_path.glob("*.txt"))

    if not label_files:
        print(f"警告: {labels_dir} にラベルファイルが見つかりませんでした")
        return False

    print(f"ラベルファイル数: {len(label_files)}")
    print("ラベル形式を検証中...\n")

    error_count = 0
    for label_file in label_files[:10]:  # 最初の10ファイルをサンプル検証
        with open(label_file, 'r') as f:
            lines = f.readlines()

        for line_num, line in enumerate(lines, 1):
            parts = line.strip().split()
            if len(parts) != 5:
                print(f"エラー [{label_file.name}:{line_num}]: "
                      f"フォーマットが不正です（5要素必要、{len(parts)}要素検出）")
                error_count += 1
                continue

            try:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                # 値の範囲チェック
                if class_id != 0:
                    print(f"警告 [{label_file.name}:{line_num}]: "
                          f"クラスIDが0ではありません: {class_id}")

                if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and
                        0 <= width <= 1 and 0 <= height <= 1):
                    print(f"エラー [{label_file.name}:{line_num}]: "
                          f"座標が範囲外です（0-1の範囲内である必要があります）")
                    error_count += 1

            except ValueError as e:
                print(f"エラー [{label_file.name}:{line_num}]: "
                      f"数値変換エラー: {e}")
                error_count += 1

    if error_count == 0:
        print("✓ ラベル形式の検証に成功しました\n")
        return True
    else:
        print(f"\n✗ {error_count}個のエラーが見つかりました\n")
        return False


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description='アノテーションデータをYOLOv11データセットに変換'
    )
    parser.add_argument(
        '-i', '--images',
        type=str,
        default='data/annotations/images',
        help='画像ディレクトリ（デフォルト: data/annotations/images）'
    )
    parser.add_argument(
        '-l', '--labels',
        type=str,
        default='data/annotations/labels',
        help='ラベルディレクトリ（デフォルト: data/annotations/labels）'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='data/table_dataset',
        help='出力ディレクトリ（デフォルト: data/table_dataset）'
    )
    parser.add_argument(
        '-r', '--train-ratio',
        type=float,
        default=0.8,
        help='学習データの割合（デフォルト: 0.8）'
    )
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='ラベル検証のみ実行'
    )

    args = parser.parse_args()

    # ラベル検証のみ
    if args.validate_only:
        validate_yolo_labels(args.labels)
        return

    # ラベル検証
    if not validate_yolo_labels(args.labels):
        print("警告: ラベル検証でエラーが見つかりました")
        response = input("続行しますか？ (y/N): ")
        if response.lower() != 'y':
            print("中止しました")
            return

    # データセット分割
    split_dataset(
        images_dir=args.images,
        labels_dir=args.labels,
        output_dir=args.output,
        train_ratio=args.train_ratio
    )


if __name__ == "__main__":
    main()
