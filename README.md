# Visuable for You Table Tennis
-> https://visualize-tt.com/

*卓球映像から選手のプレーシーンを自動検出，プレー間の不要な時間をカットする Web アプリケーション*

## 概要

卓球映像をアップロードするとサービスやラリーなどのプレー区間を自動検出，待機時間やボール拾いの時間をカットしたプレー部分のみの動画を生成します．


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

## MLフロー

`ml/scripts/notebooks/` に Jupyter Notebook が番号順に用意されています．

| Notebook | 内容 |
|---|---|
| `01_train_table_detector.ipynb` | 卓球台検出モデルの学習 |
| `02_export_player_pose.ipynb` | 選手骨格データ抽出 |
| `03_train_lstm_play_classifier.ipynb` | LSTM プレー分類モデルの学習 |
| `04_crip.ipynb` | 動画の切り抜き |

## ライセンス

このプロジェクトは開発中です．