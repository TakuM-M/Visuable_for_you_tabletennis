# ML サービス

卓球動画から**プレーシーンのみを自動抽出**する ML 推論サービス。

## 処理の概要

```
動画入力
  ↓
卓球台検出（カスタム YOLO）
  ↓
選手姿勢推定（YOLOv11-Pose）
  ↓
プレー判定（LSTM）
  ↓
動画クリップ出力
```

---

## ディレクトリ構成

```
ml/
├── app.py                # FastAPI 推論サービス（エントリポイント）
├── mock_app.py           # モック推論サービス（開発用）
├── pyproject.toml        # uv パッケージ定義
├── uv.lock               # ロックファイル
├── Dockerfile            # 本番・学習環境イメージ
├── Dockerfile.mock       # モックサービスイメージ
│
├── src/                  # ソースコード
│   ├── core/             # データクラス・共通定義
│   ├── detection/        # 検出・トラッキング
│   ├── models/           # モデルクラス定義（LSTM等）
│   ├── pipelines/        # 推論パイプライン
│   ├── training/         # 学習パイプライン
│   ├── datasets/         # データセット
│   ├── annotation/       # アノテーション補助ツール
│   ├── utils/            # 共通ユーティリティ
│   └── visualization/    # デバッグ用可視化ツール
│
├── models/               # 学習済みモデルの重みファイル
│   ├── table_detection/  # 卓球台検出モデル（YOLO）
│   ├── play_classifier/  # プレー判定モデル（LSTM）
│   └── pretrained/       # 事前学習済みモデル（YOLO Pose 等）
│
├── scripts/              # 学習・アノテーションスクリプト
│   ├── notebooks/        # Jupyter Notebook（実験・分析）
│   ├── train_play_classifier.py
│   ├── augment_pose_data.py
│   └── play_scene_annotate.py
│
├── configs/              # パイプライン設定ファイル
└── data/                 # データ（raw / processed）
```

---

## src/ モジュール詳細

### `pipelines/` — 推論パイプライン
本番で使用するメインパイプライン。

| ファイル | 役割 |
|---------|------|
| `inference_pipeline.py` | エンドツーエンド推論（動画 → クリップ） |
| `play_scene_detector.py` | プレーシーン区間の検出 |
| `player_pose_exporter.py` | 選手姿勢データの抽出・CSV出力 |
| `video_composer.py` | クリップ動画の合成 |
| `config.py` | 推論パイプライン設定クラス |
| `exceptions.py` | 推論パイプライン例外クラス |

### `training/` — 学習パイプライン
プレー判定モデルの学習に使用。

| ファイル | 役割 |
|---------|------|
| `training_pipeline.py` | LSTM モデルの学習パイプライン |
| `pose_augmentation.py` | 姿勢データのデータ拡張 |
| `config.py` | 学習パイプライン設定クラス |
| `exceptions.py` | 学習パイプライン例外クラス |

### `detection/` — 検出・トラッキング

| ファイル | 役割 |
|---------|------|
| `table_detector.py` | 卓球台領域の検出（カスタム YOLO） |
| `yolopose_tracker.py` | YOLOv11-Pose による姿勢推定・トラッキング |
| `player_classifier.py` | 手前選手・相手選手の分類 |
| `tracking_exporter.py` | トラッキング結果の CSV 出力 |

### `models/` — モデルクラス定義

| ファイル | 役割 |
|---------|------|
| `play_classifier_lstm.py` | LSTM / CNN-LSTM プレー判定モデル |

---

## セットアップ

### ローカル開発（macOS 等）

```bash
# 開発環境のセットアップ（torch は CPU 版がインストールされる）
uv sync --dev

# スクリプト実行
uv run python scripts/train_play_classifier.py \
    --train-dirs data/processed/train \
    --val-dirs data/processed/val

# テスト
uv run pytest
```

### Docker（GPU 環境）

```bash
# 推論サービス起動
docker compose up ml-service

# 学習・開発コンテナ（GPU）
docker compose --profile ml up ml
```

> Docker 環境では PyTorch の CUDA 版が使用される（`pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime` ベース）。

---

## インポート例

```python
# 推論パイプライン
from src.pipelines import InferencePipeline, InferencePipelineConfig

# 学習パイプライン
from src.training import TrainingPipeline, TrainingPipelineConfig

# 検出モジュール
from src.detection.table_detector import TableDetector
from src.detection.yolopose_tracker import YOLOPoseTracker

# モデル
from src.models.play_classifier_lstm import PlayClassifierLSTM
```
