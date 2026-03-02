# Visuable for You Table Tennis

卓球のプレー動画から必要な部分（プレー中）のみを自動抽出し、プレー間の不要な時間をカットするWebアプリケーション

## 概要

このプロジェクトは、卓球の練習や試合の動画を効率的に編集するためのツールです。動画内のサービスやラリーなどのプレー区間を自動検出し、待機時間やボール拾いの時間などをカットして、プレー部分のみを含む動画を生成します。

## 処理フロー

```
動画入力
  └─ 卓球台検出（カスタムYOLOモデル）
       └─ 選手姿勢推定（YOLOv11-Pose）
            └─ プレー区間分類（LSTM）
                 └─ 動画切り抜き・出力
```

## プロジェクト構造

```
Visuable_for_you_tabletennis/
├── backend/                         # FastAPI バックエンド（開発予定）
├── frontend/                        # React フロントエンド（開発予定）
├── ml/                              # ML パイプライン
│   ├── src/                        # ソースコード
│   │   ├── detection/              # 検出・トラッキング
│   │   │   ├── table_detector.py   # 卓球台検出
│   │   │   ├── yolopose_tracker.py # YOLOv11-Pose トラッキング
│   │   │   ├── player_classifier.py# 選手判別
│   │   │   └── tracking_exporter.py# CSV 出力
│   │   ├── models/                 # ML モデルクラス
│   │   │   └── play_classifier_lstm.py # LSTM プレー分類器
│   │   ├── pipelines/              # 推論パイプライン
│   │   │   ├── inference_pipeline.py
│   │   │   ├── play_scene_detector.py
│   │   │   ├── player_pose_exporter.py
│   │   │   └── video_composer.py
│   │   ├── datasets/               # データセット管理
│   │   ├── annotation/             # アノテーションツール
│   │   ├── visualization/          # 可視化ツール
│   │   ├── utils/                  # ユーティリティ
│   │   └── main.py                 # エントリポイント
│   ├── models/                     # 学習済みモデルファイル
│   │   ├── pretrained/             # 事前学習済みモデル（YOLO）
│   │   ├── play_classifier/        # LSTMプレー分類モデル
│   │   └── table_detection/        # 卓球台検出モデル
│   ├── data/                       # データ
│   │   ├── raw/                    # 元動画
│   │   └── processed/              # 処理済みデータ（CSVラベル等）
│   ├── scripts/                    # 学習・実験スクリプト
│   │   ├── notebooks/              # Jupyter Notebook（01〜05）
│   │   ├── train_play_classifier.py
│   │   └── augment_pose_data.py
│   ├── configs/                    # 設定ファイル
│   └── requirements.txt
├── docs/                            # ドキュメント
│   ├── architecture/               # アーキテクチャ設計（proto_type_01〜04）
│   ├── components/                 # コンポーネント仕様
│   └── development/                # 開発メモ
├── docker-compose.yml
├── Dockerfile
└── README.md
```

## 技術スタック

### ML パイプライン

| 用途 | ライブラリ |
|---|---|
| 動画処理 | OpenCV |
| 姿勢推定・トラッキング | YOLOv11-Pose (ultralytics) |
| 卓球台検出 | カスタム YOLO モデル |
| プレー区間分類 | LSTM (PyTorch) |
| データ処理 | NumPy, pandas |

### Web アプリ（開発予定）

**Backend**
- Python / FastAPI
- SQLAlchemy + Alembic（ORM・マイグレーション）
- Pydantic（バリデーション）
- PostgreSQL

**Frontend**
- TypeScript / React
- React Router, react-query, react-hook-form
- Zod（バリデーション）
- Tailwind CSS
- Orval（OpenAPI → APIフック自動生成）

**インフラ**
- Docker / Docker Compose

## セットアップ（ML）

### 1. 仮想環境の作成と有効化

```bash
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
```

### 2. 依存ライブラリのインストール

```bash
pip install -r ml/requirements.txt
pip install ultralytics  # YOLOv11
```

### 3. 動画処理の実行

```bash
python ml/src/main.py data/raw/your_video.mp4 -o output -v
```

## 学習フロー

`ml/scripts/notebooks/` に Jupyter Notebook が番号順に用意されています。

| Notebook | 内容 |
|---|---|
| `01_train_table_detector.ipynb` | 卓球台検出モデルの学習 |
| `02_export_player_pose.ipynb` | 選手の姿勢データ抽出 |
| `02b_pose_augmentation.ipynb` | 姿勢データのオーグメンテーション |
| `03_train_lstm_play_classifier.ipynb` | LSTMプレー分類モデルの学習 |
| `04_predict_play_scenes.ipynb` | プレー区間の予測 |
| `05_crip.ipynb` | 動画の切り抜き |

## ライセンス

このプロジェクトは開発中です。
