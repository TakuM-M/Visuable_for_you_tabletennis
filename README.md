# Visuable for You Table Tennis

卓球のプレー映像からプレー中のみを自動抽出し、プレー間の不要な時間をカットする Web アプリケーション

## 概要

卓球の試合動画をアップロードすると、ML パイプラインがサービスやラリーなどのプレー区間を自動検出し、待機時間やボール拾いの時間をカットしたプレー部分のみの動画を生成します。
[Visuable for You](https://visualize-tt.com/login)

## アーキテクチャ

```
ユーザー (ブラウザ)
    ↓
Cloudflare (DNS / CDN)
    ↓
nginx (:443 SSL終端)
    ├── /api/*      → FastAPI Backend (:8000)
    ├── /internal/* → FastAPI Backend (:8000)  ← RunPod からのコールバック
    └── /*          → React Frontend (:80)

Backend → RunPod Serverless API → ML Worker (GPU)
    ↑                                  │
    └──── コールバック (clips 結果) ←───┘

Backend ←→ PostgreSQL (:5432)
```

### ML パイプライン

```
動画入力
  └─ 卓球台検出（カスタム YOLO モデル）
       └─ 選手姿勢推定（YOLOv11-Pose）
            └─ プレー区間分類（LSTM）
                 └─ プレーシーンのタイムスタンプを返却
```

## 技術スタック

### Backend

| 用途 | 技術 |
|---|---|
| Web フレームワーク | FastAPI |
| ORM / マイグレーション | SQLAlchemy 2.0 + Alembic |
| データベース | PostgreSQL 17 |
| 認証 | JWT (python-jose) |
| バリデーション | Pydantic |
| メール | Resend API |
| パッケージ管理 | uv |

### Frontend

| 用途 | 技術 |
|---|---|
| UI | React 19 + TypeScript |
| ビルド | Vite |
| スタイル | Tailwind CSS |
| ルーティング | React Router v7 |
| データ取得 | React Query |
| フォーム | React Hook Form + Zod |
| API クライアント生成 | Orval (OpenAPI → TypeScript) |

### ML パイプライン

| 用途 | 技術 |
|---|---|
| 姿勢推定 | YOLOv11-Pose (ultralytics) |
| 卓球台検出 | カスタム YOLO モデル |
| プレー区間分類 | LSTM (PyTorch 2.1 / CUDA 12.1) |

### インフラ

| 用途 | 技術 |
|---|---|
| コンテナ | Docker / Docker Compose |
| リバースプロキシ / SSL | nginx + Cloudflare Origin Certificate |
| VPS | ConoHa VPS |
| GPU 推論 | RunPod Serverless |

## ディレクトリ構成

```
.
├── backend/           # FastAPI バックエンド
│   ├── app/
│   │   ├── routers/   # API エンドポイント
│   │   ├── models/    # SQLAlchemy モデル
│   │   ├── schemas/   # Pydantic スキーマ
│   │   ├── services/  # ビジネスロジック
│   │   └── repositories/ # データアクセス層
│   └── alembic/       # DB マイグレーション
├── frontend/          # React フロントエンド
│   └── src/
│       ├── pages/     # ページコンポーネント
│       ├── components/
│       └── api/       # Orval 自動生成
├── ml/                # ML サービス
│   ├── src/
│   │   ├── pipelines/      # 推論パイプライン
│   │   ├── detection/      # 検出・トラッキング
│   │   ├── models/         # モデルアーキテクチャ
│   │   ├── datasets/       # データセットローダー
│   │   ├── training/       # 学習パイプライン
│   │   ├── annotation/     # アノテーションツール
│   │   ├── core/           # データクラス・例外定義
│   │   ├── utils/          # ユーティリティ
│   │   └── visualization/  # 可視化ツール
│   ├── scripts/
│   │   ├── notebooks/      # 学習・評価用 Notebook
│   │   └── play_scene_annotate.py  # プレーシーンアノテーション
│   ├── models/             # 学習済みモデル (git 管理外)
│   ├── runpod_handler.py   # RunPod Serverless エントリーポイント
│   └── mock_app.py         # ローカル開発用モックサービス
├── nginx/             # nginx 設定
│   └── nginx.prod.conf
├── docker-compose.yml      # ローカル開発用
└── docker-compose.prod.yml # 本番用
```

## セットアップ（ローカル開発）

### 1. 環境変数の設定

```bash
cp .env.example .env
# .env を編集して必要な値を設定
```

### 2. Docker Compose で起動

```bash
docker compose up
```

以下のサービスが起動します:
- **postgres** — データベース (:5432)
- **backend** — FastAPI (:8000) ホットリロード有効
- **frontend** — React/Vite (:5173) ホットリロード有効
- **ml-mock** — ML モックサービス (:8001)

### 3. DB マイグレーション

```bash
docker compose exec backend alembic upgrade head
```

## 本番デプロイ

### VPS (ConoHa)

```bash
# .env.prod を設定後
docker compose -f docker-compose.prod.yml up -d --build
```

nginx が SSL 終端を行い、ポート 80/443 で外部リクエストを受け付けます。

### ML 推論 (RunPod Serverless)

ML 推論は RunPod Serverless で実行されます。Docker イメージをビルドして DockerHub に push します。

```bash
cd ml
docker build --platform linux/amd64 --target runpod-worker -t takumm/tabletennis-ml:v1.0.0 .
docker push takumm/tabletennis-ml:v1.0.0
```

RunPod エンドポイントでこのイメージを指定して使用します。

## 学習フロー

`ml/scripts/notebooks/` に Jupyter Notebook が番号順に用意されています。

| Notebook | 内容 |
|---|---|
| `01_train_table_detector.ipynb` | 卓球台検出モデルの学習 |
| `02_export_player_pose.ipynb` | 選手の姿勢データ抽出 |
| `03_train_lstm_play_classifier.ipynb` | LSTM プレー分類モデルの学習 |
| `04_crip.ipynb` | 動画の切り抜き |

## ライセンス

このプロジェクトは開発中です。
