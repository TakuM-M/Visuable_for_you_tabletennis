<div align="center">

# Visuable for You Tabletennis

**卓球の試合動画から「プレー中の区間」だけを自動抽出して、ラリー中だけの動画を生成する Web アプリケーション**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-visualize--tt.com-1f6feb?style=for-the-badge)](https://visualize-tt.com/)

</div>

---
### snap

<div align="center">
<table>
<tr>
<td align="center"><strong>Before（元の試合動画）</strong></td>
<td align="center"><strong>After（プレー区間だけ抽出）</strong></td>
</tr>
<tr>
<td align="center">
  <video src="https://github.com/user-attachments/assets/94b280ad-3f8c-4c40-8d47-bbe9747cb135" controls width="100%"></video><br/>
  ⏱ 1:10
</td>
<td align="center">
  <video src="https://github.com/user-attachments/assets/2b834427-0b67-4f7f-baaf-a49a0f2b77ac" controls width="100%"></video><br/>
  ⏱ 0:12
</td>
</tr>
</table>

<sub>※ デモ動画では 3 ラリー分を抽出しています。プライバシー保護のため、選手の顔など一部映像にモザイク加工を施しています。</sub>

</div>

---

## プロジェクト概要

| 項目 | 内容 |
|------|------|
| **開発期間** | 2025年12月 〜 2026年5月（約6ヶ月） |
| **公開URL** | [https://visualize-tt.com/](https://visualize-tt.com/) |
| **ステータス** | デモ公開中 |



## UI

<div align="center">

<table>
<tr>
<td align="center" width="50%">
  <img src="docs/images/home_list.png" width="100%" /><br/>
  <sub><b>home</b></sub>
</td>
<td align="center" width="50%">
  <img src="docs/images/upload02.png" width="100%" /><br/>
  <sub><b>upload</b></sub>
</td>
</tr>
<tr>
<td align="center" width="50%">
  <img src="docs/images/upload05.png" width="100%" /><br/>
  <sub><b>waiting</b></sub>
</td>
<td align="center" width="50%">
  <img src="docs/images/upload06.png" width="100%" /><br/>
  <sub><b>result（一部加工）</b></sub>
</td>
</tr>
</table>

</div>

---

## ユーザーフロー

1. **アカウント登録** — メールアドレスで新規登録 → 認証メールのリンクをクリック
2. **動画アップロード** — 試合の動画ファイルを画面にアップロード
3. **処理を待つ** — AI が動画を解析してプレー区間を検出（処理完了時にメール通知）
4. **ダウンロード** — プレーシーンだけを連結した動画を取得

---

## 主な機能

- **メール認証付きユーザー登録 / ログイン**（JWT）
- **動画アップロード**（大容量動画にも対応するチャンク分割アップロード）
- **AI による自動プレー区間検出**（卓球台検出 + 姿勢推定 + LSTM 時系列分類）
- **ハイライト動画の自動生成**（FFmpeg によるクリップ結合）
- **処理完了をメールで通知**（Resend API）
- **クラウドストレージ連携**（Cloudflare R2 / S3 互換）

---

## ML パイプライン

```mermaid
flowchart LR
    A[入力動画] --> B[① 卓球台検出<br/>YOLOv11]
    B --> C[② 選手の姿勢推定<br/>YOLOv11-Pose]
    C --> D[③ プレー/非プレー分類<br/>LSTM]
    D --> E[プレー区間<br/>start, end の配列]
```

| ステップ | モデル | 役割 |
|---------|--------|------|
| ① 卓球台検出 | YOLOv11 (転移学習) | フレーム内の卓球台位置を特定し、選手抽出の基準とする |
| ② 姿勢推定 | YOLOv11-Pose | 各選手の骨格 17 キーポイントを抽出 |
| ③ プレー分類 | LSTM | 時系列の姿勢特徴量から「プレー中 / 非プレー中」を分類 |

<details>
<summary><b>技術選定の理由</b></summary>

### なぜ YOLO, LSTMの二段階フローなのか

- **検討した候補**: 
    - 単一モデルで直接プレー区間を検出するアプローチ（例: 時間的畳み込みネットワーク）
    - ボール検出によるプレー区間推定
- **課題**: 
    - **アマチュア層への適用性**: プロの試合映像は画角が固定されているという前提が成り立ちやすい一方，アマチュア層は三脚撮影が中心で，会場の狭さや撮影機材の制約から十分な画角を確保できないケースが多い．
    - **ボール検出の難しさ**: 卓球のボールは小さく，また高速であるため検出の計算量がボトルネックとなる．加えて，画角によってはボールが映らないなどの問題もある．
- **YOLO + LSTM のメリット**: 
    - **均一な特徴量抽出**: YOLO による選手の骨格特徴量抽出は，画角や撮影条件の違いに対して比較的ロバストである．これにより，LSTM は「プレー中 / 非プレー中」の分類に専念できるため，検出の安定性などが見込めると判断した．
</details>


## 技術スタック

<div align="center">

### Frontend
![React](https://img.shields.io/badge/React-61DAFB?style=flat&logo=react&logoColor=black)
![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?style=flat&logo=typescript&logoColor=white)
![Vite](https://img.shields.io/badge/Vite-646CFF?style=flat&logo=vite&logoColor=white)
![TailwindCSS](https://img.shields.io/badge/Tailwind-06B6D4?style=flat&logo=tailwindcss&logoColor=white)
![TanStack Query](https://img.shields.io/badge/TanStack_Query-FF4154?style=flat&logo=reactquery&logoColor=white)
![Orval](https://img.shields.io/badge/Orval-OpenAPI-green?style=flat)

### Backend
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)
![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-ORM-red?style=flat)
![Alembic](https://img.shields.io/badge/Alembic-Migration-blue?style=flat)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?style=flat&logo=postgresql&logoColor=white)
![uv](https://img.shields.io/badge/uv-Package_Manager-orange?style=flat)

### ML
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Ultralytics](https://img.shields.io/badge/YOLO-Ultralytics-yellow?style=flat)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white)
![FFmpeg](https://img.shields.io/badge/FFmpeg-007808?style=flat&logo=ffmpeg&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.1-76B900?style=flat&logo=nvidia&logoColor=white)

### Infra / DevOps
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)
![Nginx](https://img.shields.io/badge/Nginx-009639?style=flat&logo=nginx&logoColor=white)
![Cloudflare R2](https://img.shields.io/badge/Cloudflare_R2-F38020?style=flat&logo=cloudflare&logoColor=white)
![RunPod](https://img.shields.io/badge/RunPod-Serverless_GPU-purple?style=flat)
![Resend](https://img.shields.io/badge/Resend-Email-black?style=flat)

</div>
---

## ディレクトリ構成

```
.
├── backend/           # FastAPI バックエンド
│   └── app/
│       ├── routers/       # API エンドポイント層
│       ├── services/      # ビジネスロジック層
│       ├── repositories/  # データアクセス層
│       ├── models/        # SQLAlchemy モデル
│       └── schemas/       # Pydantic スキーマ
├── frontend/          # React + TypeScript フロントエンド
│   └── src/
│       ├── pages/         # 画面コンポーネント
│       ├── components/
│       └── api/           # Orval 自動生成クライアント
├── ml/                # ML 推論 / 学習サービス
│   ├── src/
│   │   ├── pipelines/     # 推論パイプライン
│   │   ├── detection/     # 検出 / トラッキング
│   │   ├── models/        # モデルアーキテクチャ
│   │   ├── training/      # 学習パイプライン
│   │   └── ...
│   ├── scripts/
│   │   └── notebooks/     # 学習・推論用 Colab Notebook
│   ├── runpod_handler.py  # RunPod Serverless ハンドラ
│   └── mock_app.py        # 開発用モック ML サービス
├── nginx/             # Nginx 設定（本番リバースプロキシ）
├── docs/              # 設計ドキュメント / 画像
├── docker-compose.dev.yml   # ローカル開発用
└── docker-compose.yml       # 本番用
```

---

## ML 学習スクリプトを Colab で動かす

`ml/scripts/notebooks/` には **Google Colab で実行可能な学習・推論ノートブック** が用意されています。

### ノートブック一覧

| # | Notebook | 内容 | Open in Colab |
|---|----------|------|----------------|
| 01 | `01_train_table_detector.ipynb` | 卓球台検出モデル (YOLOv11) の学習 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TakuM-M/Visuable_for_you_tabletennis/blob/main/ml/scripts/notebooks/01_train_table_detector.ipynb) |
| 02 | `02_export_player_pose.ipynb` | 動画から選手の骨格データ (CSV) を抽出 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TakuM-M/Visuable_for_you_tabletennis/blob/main/ml/scripts/notebooks/02_export_player_pose.ipynb) |
| 03 | `03_train_lstm_play_classifier.ipynb` | LSTM プレー分類モデルの学習 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TakuM-M/Visuable_for_you_tabletennis/blob/main/ml/scripts/notebooks/03_train_lstm_play_classifier.ipynb) |
| 04 | `04_crip.ipynb` | 学習済みモデルを使って動画から区間切り抜きを行う | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TakuM-M/Visuable_for_you_tabletennis/blob/main/ml/scripts/notebooks/04_crip.ipynb) |

### Colab 実行のセットアップ手順

ノートブックは Google Drive 上の以下のパスをプロジェクトルートとして読み込みます。

```
/content/drive/MyDrive/Visuable_for_you_tabletennis/
```

#### リポジトリを Google Drive に配置

ローカルで本リポジトリをクローンし、`ml/` ディレクトリ配下を Google Drive にアップロード。

```
MyDrive/
└── Visuable_for_you_tabletennis/
    ├── src/              # ml/src/ をアップロード
    ├── scripts/          # ml/scripts/ をアップロード
    ├── configs/          # API キーや学習設定（後述）
    ├── data/             # データセット・動画
    └── models/           # 学習結果の保存先
```

> `ml/` 直下のフォルダ構成をそのまま Google Drive に置く形になります。

---

## ライセンス

本プロジェクトは現在開発中です。

## Author

**Takumi M.** ([@TakuM-M](https://github.com/TakuM-M))

- お問い合わせ: GitHub Issues にて
