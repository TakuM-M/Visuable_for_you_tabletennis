# Visuable for You Table Tennis

卓球のプレー動画から必要な部分（プレー中）のみを自動抽出し、プレー間の不要な時間をカットするアプリケーション

## 概要

このプロジェクトは、卓球の練習や試合の動画を効率的に編集するためのツールです。動画内のサービスやラリーなどのプレー区間を自動検出し、待機時間やボール拾いの時間などをカットして、プレー部分のみを含む動画を生成します。

## 最新機能: YOLOv11-Poseトラッキングシステム

**新実装（2025-12-29）**: YOLOv11-PoseとByteTrackを使った高精度な選手トラッキングシステムを実装しました。

### 主な機能

- **卓球台自動検出**: 動画内の卓球台位置を色ベースで自動検出
- **人物トラッキング**: YOLOv11-PoseでID付きの連続トラッキング
- **選手フィルタリング**: 手前選手・相手選手を自動判別（審判・コーチを除外）
- **姿勢データCSV出力**: 17個のキーポイント座標をフレームごとに記録
- **可視化動画出力**: トラッキング結果を重ねた動画を生成

### クイックスタート

```bash
# YOLOトラッキングシステムの実行
python src/test_yolo_tracking.py data/raw/your_video.mp4 -v

# 出力ファイル:
# - output/tracking_players_your_video.csv  (選手の姿勢データ)
# - output/tracking_your_video.mp4          (トラッキング動画)
```

詳細は[YOLOトラッキングシステムのドキュメント](#yoloトラッキングシステム)を参照してください。

## プロジェクト構造

```
Visuable_for_you_tabletennis/
├── src/                              # ソースコード
│   ├── detection/                   # 検出・トラッキングモジュール
│   │   ├── table_detector.py       # 卓球台検出
│   │   ├── yolo_tracker.py         # YOLOv11-Poseトラッキング
│   │   ├── player_filter.py        # 選手フィルタリング
│   │   ├── tracking_exporter.py    # CSV出力
│   │   └── legacy_mediapipe/       # 旧MediaPipe実装（参考用）
│   ├── training/                    # モデル学習用スクリプト
│   │   ├── extract_frames.py       # 動画からフレーム抽出
│   │   ├── prepare_dataset.py      # データセット準備
│   │   └── train_table_detector.py # モデル学習
│   ├── extraction/                  # 動画抽出モジュール
│   ├── utils/                       # ユーティリティ
│   │   └── video_loader.py         # 動画読み込み
│   ├── test_yolo_tracking.py       # YOLOトラッキングテスト（メイン）
│   └── main.py                     # メインスクリプト（旧版）
├── data/                            # データディレクトリ
│   ├── raw/                        # 元動画
│   ├── processed/                  # 処理済み動画
│   ├── annotations/                # アノテーションデータ
│   │   ├── images/                # 抽出フレーム
│   │   └── labels/                # YOLOラベル
│   ├── table_dataset/              # 学習用データセット
│   │   ├── train/                 # 学習データ
│   │   └── val/                   # 検証データ
│   └── table_dataset.yaml          # データセット設定
├── models/                          # 学習済みモデル
│   └── table_detector/             # 卓球台検出モデル
├── output/                          # 出力ディレクトリ
├── scripts/                         # 便利スクリプト
│   └── quick_start_training.sh     # 学習クイックスタート
├── docs/                            # ドキュメント
│   ├── architecture_yolo_tracking.md  # YOLOシステム設計書
│   └── annotation_guide.md         # アノテーションガイド
├── tests/                           # テストコード
├── notes/                           # 開発メモ
├── requirements.txt                 # 依存ライブラリ
└── README.md                       # このファイル
```

## セットアップ

### 1. リポジトリのクローン

```bash
cd /path/to/your/directory
```

### 2. 仮想環境の作成と有効化

```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# または
venv\Scripts\activate     # Windows
```

### 3. 依存ライブラリのインストール

```bash
pip install -r requirements.txt
# または直接インストール:
pip install opencv-python ultralytics numpy tqdm pandas
```

## YOLOトラッキングシステム

### 使い方

#### 基本的な使用方法

```bash
python src/test_yolo_tracking.py data/raw/your_video.mp4
```

#### オプション付き実行

```bash
python src/test_yolo_tracking.py data/raw/your_video.mp4 \
  -o output \              # 出力ディレクトリを指定
  -s 3 \                   # 3フレームごとにサンプリング（高速化）
  -v \                     # トラッキング動画を出力
  --conf 0.6 \            # YOLO信頼度閾値
  --device cuda            # GPU使用
```

#### オプション一覧

- `-o, --output DIR`: 出力ディレクトリ（デフォルト: output）
- `-s, --sample-rate N`: プレイヤー検出のフレームサンプリングレート（デフォルト: 1）
- `-v, --output-video`: トラッキング結果の動画を出力
- `--conf FLOAT`: YOLO検出信頼度の閾値（デフォルト: 0.5）
- `--device {cpu,cuda}`: 使用デバイス（デフォルト: cpu）
- `--table-interval N`: 卓球台再検出の間隔（フレーム数）。0で初回のみ検出（デフォルト: 30）

### 出力ファイル

```
output/
├── tracking_all_<動画名>.csv          # 全人物のトラッキングデータ
├── tracking_players_<動画名>.csv      # 選手のみのトラッキングデータ
├── tracking_<動画名>_near.csv         # 手前選手のみ
├── tracking_<動画名>_far.csv          # 相手選手のみ
├── tracking_summary_<動画名>.txt      # 統計サマリー
└── tracking_<動画名>.mp4              # トラッキング動画（-v指定時）
```

### CSV形式

各CSVファイルには以下の情報が含まれます：

- `track_id`: トラッキングID
- `frame`: フレーム番号
- `timestamp`: タイムスタンプ（秒）
- `role`: 役割（near_player=手前選手, far_player=相手選手, other=その他）
- `confidence`: 検出信頼度
- `bbox_*`: バウンディングボックス座標
- `nose_x, nose_y, nose_conf`: 鼻のキーポイント座標・信頼度
- その他16個のキーポイント（計17点: COCO形式）

詳細な設計ドキュメントは[docs/architecture_yolo_tracking.md](docs/architecture_yolo_tracking.md)を参照してください。

### パフォーマンス最適化

**卓球台とプレイヤーの検出FPS分離（2025-12-30実装）**

卓球台は静止物のため低FPS検出、プレイヤーは動的なため高FPS検出を行うことでパフォーマンスを最適化しています。

```bash
# 卓球台: 初回のみ検出（最速・推奨）
python src/test_yolo_tracking.py data/raw/your_video.mp4 --table-interval 0 -v

# 卓球台: 30フレームごとに再検出（カメラが動く場合）
python src/test_yolo_tracking.py data/raw/your_video.mp4 --table-interval 30 -v

# プレイヤー検出も2フレームに1回にして高速化
python src/test_yolo_tracking.py data/raw/your_video.mp4 --table-interval 0 -s 2 -v
```

## カスタムモデル学習

卓球台検出の精度を向上させるため、カスタムYOLOモデルを学習できます。

### クイックスタート

```bash
# 全自動スクリプト（アノテーション以外）
bash scripts/quick_start_training.sh
```

### 手動実行

#### 1. フレーム抽出

```bash
# 複数動画から一括抽出（推奨）
python src/training/extract_frames.py data/raw -o data/annotations/images -i 30 -m 50

# 単一動画から抽出
python src/training/extract_frames.py data/raw/sample_video.MOV -o data/annotations/images
```

#### 2. アノテーション

**Label Studio を使用（推奨）:**

```bash
# インストール
pip install label-studio

# 起動
label-studio
```

ブラウザで http://localhost:8080 を開き、以下の手順でアノテーション:
1. プロジェクト作成（Object Detection with Bounding Boxes）
2. `data/annotations/images` をアップロード
3. ラベル名を `table_tennis_table` に設定
4. 各画像で卓球台をバウンディングボックスで囲む
5. YOLO形式でエクスポート
6. ラベルファイル(.txt)を `data/annotations/labels` にコピー

詳細は [docs/annotation_guide.md](docs/annotation_guide.md) を参照。

#### 3. データセット準備

```bash
# ラベル検証
python src/training/prepare_dataset.py --validate-only

# train/val分割
python src/training/prepare_dataset.py -i data/annotations/images -l data/annotations/labels
```

#### 4. モデル学習

```bash
# CPU
python src/training/train_table_detector.py --epochs 100 --batch 16 --device cpu

# GPU (CUDA)
python src/training/train_table_detector.py --epochs 100 --batch 32 --device cuda

# Apple Silicon (MPS)
python src/training/train_table_detector.py --epochs 100 --batch 16 --device mps
```

#### 5. 学習済みモデルの使用

学習完了後、[src/detection/table_detector.py:44](src/detection/table_detector.py#L44) を編集してカスタムモデルを使用:

```python
table_detector = TableDetector(
    detection_method="yolo",
    yolo_model_path="models/table_detector/train/weights/best.pt"  # カスタムモデル
)
```

詳細な学習ガイドは [docs/annotation_guide.md](docs/annotation_guide.md) を参照してください。

## 開発状況

現在はプロトタイプ段階です。

### 完了したタスク

- [x] プロジェクト構造の作成
- [x] Python環境のセットアップ
- [x] 基本モジュールの作成

### 進行中のタスク

- [ ] タスク0: データ処理（動画の収集）
- [ ] タスク1: 動画の中でサービスする時間を抽出する

## 技術スタック

- **Python 3.x**
- **OpenCV**: 動画処理
- **MediaPipe**: 姿勢・動作検出
- **NumPy**: 数値計算
- **scikit-learn**: 機械学習（オプション）

## ライセンス

このプロジェクトは開発中です。

## 貢献

プロジェクトは現在開発中です。

## 注意事項

- このツールは開発中のプロトタイプです
- 動画ファイルは適切な形式（MP4, AVI等）である必要があります
- 処理には時間がかかる場合があります
