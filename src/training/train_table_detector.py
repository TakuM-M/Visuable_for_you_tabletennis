"""
YOLOv11を使った卓球台検出モデルの学習スクリプト
カスタムデータセットで卓球台検出に特化したモデルを作成
"""
import argparse
from pathlib import Path
from ultralytics import YOLO


def train_table_detector(
    data_yaml: str = "data/table_dataset.yaml",
    base_model: str = "yolo11n.pt",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "cpu",
    project: str = "models/table_detector",
    name: str = "train",
    patience: int = 50,
    save_period: int = 10
):
    """
    卓球台検出モデルの学習

    Args:
        data_yaml: データセット設定ファイル
        base_model: ベースモデル（転移学習元）
        epochs: 学習エポック数
        imgsz: 画像サイズ
        batch: バッチサイズ
        device: 使用デバイス（"cpu", "cuda", "mps"）
        project: プロジェクトディレクトリ
        name: 実験名
        patience: Early Stoppingの待機エポック数
        save_period: モデル保存間隔（エポック）
    """
    print("="*60)
    print("卓球台検出モデル学習")
    print("="*60)
    print(f"データセット: {data_yaml}")
    print(f"ベースモデル: {base_model}")
    print(f"エポック数: {epochs}")
    print(f"画像サイズ: {imgsz}")
    print(f"バッチサイズ: {batch}")
    print(f"デバイス: {device}")
    print(f"出力先: {project}/{name}")
    print("="*60)
    print()

    # データセットファイルの存在確認
    data_path = Path(data_yaml)
    if not data_path.exists():
        print(f"エラー: データセット設定ファイルが見つかりません: {data_yaml}")
        print("\n次のコマンドでデータセットを準備してください:")
        print("  python src/training/prepare_dataset.py")
        return

    # YOLOモデルをロード
    model = YOLO(base_model)

    # 学習実行
    results = model.train(
        data=str(data_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        patience=patience,
        save_period=save_period,
        # 最適化設定
        optimizer='AdamW',
        lr0=0.01,  # 初期学習率
        lrf=0.01,  # 最終学習率係数
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        # データ拡張
        hsv_h=0.015,  # 色相の変動
        hsv_s=0.7,    # 彩度の変動
        hsv_v=0.4,    # 明度の変動
        degrees=10.0,  # 回転範囲
        translate=0.1,  # 平行移動範囲
        scale=0.5,     # スケール範囲
        shear=0.0,     # せん断変換
        perspective=0.0,  # 透視変換
        flipud=0.0,    # 上下反転（卓球台では不要）
        fliplr=0.5,    # 左右反転
        mosaic=1.0,    # モザイク拡張
        mixup=0.0,     # ミックスアップ
        copy_paste=0.0,  # コピーペースト拡張
        # その他
        verbose=True,
        save=True,
        plots=True,
        exist_ok=True
    )

    print("\n" + "="*60)
    print("学習完了")
    print("="*60)
    print(f"最良モデル: {project}/{name}/weights/best.pt")
    print(f"最終モデル: {project}/{name}/weights/last.pt")
    print(f"学習結果: {project}/{name}")
    print("="*60)
    print()
    print("次のステップ:")
    print("  1. 結果の確認:")
    print(f"     - 学習曲線: {project}/{name}/results.png")
    print(f"     - 混同行列: {project}/{name}/confusion_matrix.png")
    print(f"     - 検証結果: {project}/{name}/val_batch*_pred.jpg")
    print()
    print("  2. モデルの評価:")
    print(f"     python src/training/evaluate_model.py --model {project}/{name}/weights/best.pt")
    print()
    print("  3. 実際の動画でテスト:")
    print(f"     python src/test_yolo_tracking.py <video_path> --table-model {project}/{name}/weights/best.pt")


def validate_model(
    model_path: str,
    data_yaml: str = "data/table_dataset.yaml",
    imgsz: int = 640,
    device: str = "cpu"
):
    """
    学習済みモデルの検証

    Args:
        model_path: モデルファイルのパス
        data_yaml: データセット設定ファイル
        imgsz: 画像サイズ
        device: 使用デバイス
    """
    print("="*60)
    print("モデル検証")
    print("="*60)
    print(f"モデル: {model_path}")
    print(f"データセット: {data_yaml}")
    print("="*60)
    print()

    model = YOLO(model_path)
    metrics = model.val(
        data=data_yaml,
        imgsz=imgsz,
        device=device,
        plots=True
    )

    print("\n" + "="*60)
    print("検証結果")
    print("="*60)
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    print("="*60)


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description='YOLOv11 卓球台検出モデルの学習'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='data/table_dataset.yaml',
        help='データセット設定ファイル（デフォルト: data/table_dataset.yaml）'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='yolo11n.pt',
        help='ベースモデル（デフォルト: yolo11n.pt）'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='学習エポック数（デフォルト: 100）'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='画像サイズ（デフォルト: 640）'
    )
    parser.add_argument(
        '--batch',
        type=int,
        default=16,
        help='バッチサイズ（デフォルト: 16）'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='使用デバイス（デフォルト: cpu）'
    )
    parser.add_argument(
        '--project',
        type=str,
        default='models/table_detector',
        help='プロジェクトディレクトリ（デフォルト: models/table_detector）'
    )
    parser.add_argument(
        '--name',
        type=str,
        default='train',
        help='実験名（デフォルト: train）'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=50,
        help='Early Stoppingの待機エポック数（デフォルト: 50）'
    )
    parser.add_argument(
        '--validate',
        type=str,
        help='学習済みモデルの検証のみ実行（モデルパスを指定）'
    )

    args = parser.parse_args()

    # 検証のみ
    if args.validate:
        validate_model(
            model_path=args.validate,
            data_yaml=args.data,
            imgsz=args.imgsz,
            device=args.device
        )
        return

    # 学習実行
    train_table_detector(
        data_yaml=args.data,
        base_model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience
    )


if __name__ == "__main__":
    main()
