# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

卓球の試合動画から「プレー中の区間」だけを自動抽出して連結動画を生成する Web アプリ。

## Common commands

### Local development (Docker)

```bash
docker compose -f docker-compose.dev.yml --env-file .env.dev up

docker compose up
```

### DB migrations

```bash
docker compose -f docker-compose.dev.yml exec backend alembic upgrade head
docker compose -f docker-compose.dev.yml exec backend alembic revision --autogenerate -m "msg"

docker exec -i tabletennis-postgres psql -U postgres -d tabletennis < backend/app/sql/seed_test_user.sql
```

### CI (GitHub Actions)

| ワークフロー | トリガー | 内容 |
|---|---|---|
| `frontend.yml` | `frontend/**` の PR | `npm ci` → `npm run lint` → `npm run build`（build に `tsc -b` が含まれるので型チェックも兼ねる） |
| `backend.yml` | `backend/**` の PR | `build` ジョブで `ruff check`、`test` ジョブで postgres を立てて `pytest`。2ジョブは並列 |
| `ml-image.yml` | 手動 / `ml/**` の PR | ML イメージのビルドと検証。手動実行時のみ Docker Hub へ push |
| `deploy.yml` | 手動のみ | VPS へ SSH して本番デプロイ |

```bash
cd frontend && npm run lint && npm run build
docker exec tabletennis-backend uv run ruff check .
docker exec tabletennis-backend uv run pytest
```

### Production deploy (VPS)

```bash
gh workflow run deploy.yml     # または Actions タブ → deploy → Run workflow
```

```bash
set -e
cd ~/Visuable_for_you_tabletennis
git fetch origin main
git reset --hard origin/main
docker compose build
docker compose run --rm backend alembic upgrade head   # 1. 先に DB マイグレーション
docker compose up -d                                   # 2. コンテナ入れ替え
docker compose restart nginx                           # 3. 最後に nginx 再起動（必須）
```

### Backend (FastAPI / uv)

```bash
cd backend
uv sync                  # dev 依存込み
uv run pytest            # テスト
uv run uvicorn app.main:app --reload
```

### Frontend (Vite / npm)

```bash
cd frontend
npm run dev              # vite dev server (port 5173)
npm run build            # tsc -b && vite build
npm run lint             # eslint .
npx orval                # backend が立ち上がってる状態で OpenAPI → src/api/generated.ts を再生成
```

### ML (uv / PyTorch)

```bash
cd ml
uv sync                  # macOS は CPU 版 torch、Linux は CUDA 12.4 版（dev + training グループ込み）

# RunPod 用イメージのビルド・push
docker build --platform linux/amd64 --target runpod-worker -t takumm/tabletennis-ml:vX.Y.Z .
docker push takumm/tabletennis-ml:vX.Y.Z
```

## Architecture

### 動画処理フロー（複数サービス・非同期コールバック）

1. **アップロード**: フロントから `POST /videos` または `POST /videos/upload/init|chunk|complete`（チャンク分割版） → `backend/app/services/video_service.py` がローカル一時保存 → **Cloudflare R2** にアップロード → DB に `videos` / `jobs` レコード作成。
2. **ML キック**: `video_service.call_ml_service()` が `BackgroundTasks` で起動。`USE_RUNPOD` フラグで分岐:
   - `true` → RunPod Serverless API (`api.runpod.ai/v2/{ENDPOINT}/run`) に POST。本番経路。
   - `false` → `ml-mock` サービス (`http://ml-mock:8001/process`) に POST。dev compose は常にこちら（`docker-compose.dev.yml` で `USE_RUNPOD=false` 固定）。
3. **ML 処理**: 渡された R2 presigned URL から動画をダウンロードし、卓球台検出（YOLO）→ 姿勢推定（YOLOv11-Pose）→ LSTM プレー分類でシーン区間 `[(start_sec, end_sec), ...]` を算出。
4. **コールバック**: ML 側が `POST {BACKEND_INTERNAL_URL}/internal/jobs/{job_id}/complete` を叩く → `backend/app/services/job_service.py:complete_job()` が R2 から元動画を取得 → FFmpeg でクリップ結合 → R2 に出力動画を put → `videos.output_path` 更新 → Resend でメール通知。
5. **取得**: フロントは `GET /videos/{id}/output` → backend が presigned URL を発行して `RedirectResponse`。


### Backend 階層（`backend/app/`）

`routers/` → `services/` → `repositories/` → `models/` のレイヤード構造。新しいエンドポイントは原則この経路で追加する。

- `routers/`: FastAPI エンドポイント。バリデーションと認証（`Depends(get_current_user)`）のみ。
- `services/`: ビジネスロジック。複数 repository を組み合わせる処理・外部 I/O（R2, ML, メール）を書く層。
- `repositories/`: SQLAlchemy セッションを使った CRUD のみ。
- `schemas/`: Pydantic（API I/O 型）。`models/` は SQLAlchemy（テーブル）。混同しない。
- `core/`: `config.py`（Settings）, `security.py`（JWT）, `deps.py`（`get_current_user`）。
- `db/session.py`: `get_db()` 依存関数のみ。

### Frontend (`frontend/src/`)

- API クライアントは **Orval が `http://localhost:8000/openapi.json` から自動生成**（`src/api/generated.ts`）。手書きしない。backend のスキーマ変更後は `npx orval` で再生成。
- Vite dev サーバの `/api/*` は `backend:8000` にプロキシされる（`vite.config.ts`、`/api` プレフィックスは除去）。Orval も `baseUrl: "/api"` を前提にしているのでセットで考える。
- 認証: JWT を localStorage に保存（`lib/auth.ts`）、`Authorization: Bearer` ヘッダで送信。
- 大きい動画は `lib/chunkedUpload.ts` でチャンク分割アップロード（`/videos/upload/init|chunk|complete`）。

### ML サービス (`ml/`)

エントリーポイントが用途別に分かれている:
- `runpod_handler.py` — RunPod Serverless ワーカー。Docker `runpod-worker` ステージで起動。
- `mock_app.py` — ローカル開発用の偽 ML（5 秒待って動画長の 0–33%, 66–100% を「プレー区間」として返す）。`mock` ステージ。
- `src/pipelines/inference_pipeline.py` — 本物の推論パイプライン。

`Dockerfile` は 6 ステージのマルチビルド（`base` → `system-deps` → `python-deps` → `app` / `mock` / `runpod-worker`）。target を変えて使い分ける。

学習済みモデルは `ml/models/` に置く。**推論が使う 4 ファイル（計 54MB）だけは git 管理下**に置き、イメージに焼き込む。RunPod に Network Volume はマウントしていないので、モデルはイメージに入っていないと動かない。

- `models/table_detection/best.pt` / `models/pretrained/yolo11m-pose.pt` / `models/play_classifier/lstm_model.pth` / `models/play_classifier/lstm_config.json`
- 学習の中間生成物や未使用の事前学習モデルは除外したまま。**追跡対象はルートの `.gitignore` と `ml/.dockerignore` の 2 箇所に許可リストがあるので必ず揃える**。
- モデルを差し替えるときは `ml/configs/runpod_config.json`（これも git 管理下）のパスと実ファイル名の一致を確認する。`PlaySceneDetector._load_config()` は設定ファイルが見つからなくても警告だけで既定値に進むため、パスの綴り違いが黙って通る。

`ml/models/` と `ml/configs/` を丸ごと `.gitignore` していた頃、ローカルビルドでは作業ディレクトリの実ファイルが `COPY . /workspace/` で入るため気づけなかったが、ビルドを CI に移した途端 `actions/checkout` がこれらを持って来ずイメージから欠落し、ワーカーが起動直後に `FileNotFoundError` で落ちて RunPod 上で unhealthy になった（v1.3.2）。CI の「ワーカー起動のスモークテスト」がこの欠落を検知する。

### Infra

- 本番: nginx (SSL 終端) → backend / frontend / ml。`/api/*` と `/internal/*` は backend、それ以外は frontend。
- `.env` (prod) / `.env.dev` (dev) を `--env-file` で読み分ける。両方とも git 管理外。
- R2 認証は `R2_ENDPOINT_URL` / `R2_ACCESS_KEY_ID` / `R2_SECRET_ACCESS_KEY` / `R2_BUCKET_NAME`。
- RunPod 認証は `RUNPOD_API_KEY` / `RUNPOD_ENDPOINT_ID` / `USE_RUNPOD=true`。

## Conventions

- バックエンドのコメント・docstring・例外メッセージ・ログは日本語で書かれている。既存コードに合わせる。
- パッケージ管理は backend / ml ともに **uv**（`pip` を直接使わない）。frontend は npm。
- Python は `>=3.12`（backend）/ `>=3.10`（ml）。バージョンを跨ぐ依存追加には注意。
- 新規 ML 機能は `src/pipelines/` 配下にパイプラインクラスとして追加し、`runpod_handler.py` から呼び出す。`mock_app.py` 側にもダミー応答の整合性が必要なら追従する。
