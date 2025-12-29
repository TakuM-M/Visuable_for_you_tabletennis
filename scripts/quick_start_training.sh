#!/bin/bash
# 卓球台検出モデル学習のクイックスタートスクリプト

set -e  # エラーで停止

echo "========================================="
echo "卓球台検出モデル学習 - クイックスタート"
echo "========================================="
echo ""

# プロジェクトルートに移動
cd "$(dirname "$0")/.."

# ステップ1: フレーム抽出
echo "ステップ1: 動画からフレーム抽出"
echo "========================================="
if [ ! -d "data/annotations/images" ] || [ -z "$(ls -A data/annotations/images)" ]; then
    echo "data/raw ディレクトリから動画フレームを抽出します..."
    python src/training/extract_frames.py data/raw \
        -o data/annotations/images \
        -i 30 \
        -m 50 \
        -w 640
    echo "✓ フレーム抽出完了"
else
    echo "✓ フレームは既に存在します（スキップ）"
fi
echo ""

# ステップ2: アノテーション指示
echo "ステップ2: アノテーション"
echo "========================================="
if [ ! -d "data/annotations/labels" ] || [ -z "$(ls -A data/annotations/labels 2>/dev/null)" ]; then
    echo "⚠️  アノテーションが必要です"
    echo ""
    echo "次の手順でアノテーションを実行してください:"
    echo ""
    echo "【推奨】Label Studio を使用:"
    echo "  1. Label Studio をインストール:"
    echo "     pip install label-studio"
    echo ""
    echo "  2. Label Studio を起動:"
    echo "     label-studio"
    echo ""
    echo "  3. ブラウザで http://localhost:8080 を開く"
    echo ""
    echo "  4. プロジェクト作成:"
    echo "     - Project Name: table_tennis_detection"
    echo "     - Data Import: data/annotations/images をアップロード"
    echo "     - Labeling Setup: Object Detection with Bounding Boxes"
    echo "     - Label名: table_tennis_table"
    echo ""
    echo "  5. アノテーション実行:"
    echo "     - 各画像で卓球台をバウンディングボックスで囲む"
    echo "     - ラベルを table_tennis_table に設定"
    echo "     - Submit で保存"
    echo ""
    echo "  6. エクスポート:"
    echo "     - Export → YOLO 形式"
    echo "     - ダウンロードしたZIPを解凍"
    echo "     - ラベルファイル(.txt)を data/annotations/labels にコピー"
    echo ""
    echo "詳細は docs/annotation_guide.md を参照してください"
    echo ""
    exit 1
else
    echo "✓ アノテーション済みラベルが見つかりました"
fi
echo ""

# ステップ3: ラベル検証
echo "ステップ3: ラベル形式の検証"
echo "========================================="
python src/training/prepare_dataset.py --validate-only
echo ""

# ステップ4: データセット準備
echo "ステップ4: データセット準備（train/val分割）"
echo "========================================="
python src/training/prepare_dataset.py \
    -i data/annotations/images \
    -l data/annotations/labels \
    -o data/table_dataset \
    -r 0.8
echo ""

# ステップ5: 学習実行
echo "ステップ5: モデル学習"
echo "========================================="
echo "学習を開始します..."
echo "（デバイス: cpu、エポック: 100、バッチサイズ: 16）"
echo ""
read -p "学習を開始しますか？ (y/N): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python src/training/train_table_detector.py \
        --data data/table_dataset.yaml \
        --model yolo11n.pt \
        --epochs 100 \
        --batch 16 \
        --device cpu \
        --project models/table_detector \
        --name train

    echo ""
    echo "========================================="
    echo "✓ 学習完了"
    echo "========================================="
    echo ""
    echo "学習済みモデル:"
    echo "  models/table_detector/train/weights/best.pt"
    echo ""
    echo "学習結果:"
    echo "  models/table_detector/train/results.png"
    echo "  models/table_detector/train/confusion_matrix.png"
    echo ""
    echo "次のステップ:"
    echo "  1. 実際の動画でテスト:"
    echo "     python src/test_yolo_tracking.py data/raw/sample_video_01_01.MOV --table-interval 0 -v"
    echo ""
    echo "  2. table_detector.py のモデルパスを更新:"
    echo "     yolo_model_path='models/table_detector/train/weights/best.pt'"
    echo ""
else
    echo "学習をスキップしました"
fi
