# feat: ジョブ耐障害性・エラーハンドリング強化

**Status: Done (2026-05-18)**

重要度: 中〜高

## 背景・目的

現状、ML ジョブが失敗してもリトライや状態回復の仕組みがなく、ユーザーは詰む。RunPod からのコールバックが届かないケース、アップロード中断時のゴミファイルなど、障害シナリオへの対応が不足している。耐障害性を上げて運用負荷を下げる。

## 受け入れ条件

- [x] ML ジョブ失敗時に自動リトライされる（回数・バックオフ戦略を明文化）
    - `job_max_retries=2`、指数バックオフ `60s / 600s`。`jobs.retry_count` を DB で保持
- [x] RunPod コールバック未着時にタイムアウト検知して `jobs.status` を failed に遷移できる
    - 実際の処理には時間がかかるため，タイムアウト用の時刻設定は慎重に設定する必要がある．
    - 処理自体に非常に時間がかかる，加えてrunpodserverless自体がGPU割り当てが少ない可能性もあるため，現状は24時間程度を想定している．
    - APScheduler の `reap_timeouts` ジョブが 60 秒ごとに `started_at < now() - 24h` の queued/processing ジョブを failed に遷移させる
- [x] アップロード中断時に `tmp` ディレクトリのゴミファイルが定期 or 起動時にクリーンアップされる
    - 起動時に 1 回 + 1 時間ごとに `clean_tmp_dir` が 24h 経過した項目を削除
- [x] 失敗ジョブに対しユーザー向けの再実行 UI または手動再実行 API がある
    - `POST /jobs/{job_id}/retry`（所有者検証付き）+ `VideoDetailPage` の「再実行」ボタン。手動時は `retry_count=0` にリセット
- [x] エラー時のログ・通知（Resend or その他）が整備されている
    - 標準 `logging` モジュール導入。最終失敗（自動リトライ枠を使い切った時）のみ Resend で `send_clip_failure_email` を送信

## 関連ファイル

- `backend/app/services/job_service.py` — `handle_ml_failure` / `retry_job` 追加
- `backend/app/services/video_service.py` — `call_ml_service` に失敗ハンドリング統合
- `backend/app/services/job_reaper.py`（新規）— APScheduler に登録する 3 ジョブ
- `backend/app/services/email_service.py` — `send_clip_failure_email` 追加
- `backend/app/routers/jobs.py` — `POST /jobs/{job_id}/retry` 追加
- `backend/app/repositories/job.py` — reaper 用クエリ追加
- `backend/app/models/job.py` — `retry_count` / `next_retry_at` / `updated_at` カラム追加
- `backend/app/core/config.py` — リトライ・タイムアウト設定追加
- `backend/app/core/logging.py`（新規）— `print` → `logging` への置換基盤
- `backend/app/main.py` — `lifespan` に APScheduler 組み込み
- `backend/alembic/versions/8c4f2b7e9d51_add_retry_fields_to_jobs.py`（新規）
- `frontend/src/pages/VideoDetailPage.tsx` — 「再実行」ボタンと `useMutation` 追加

## 設計メモ

- リトライは backend 側で管理することにした（RunPod のリトライ機構ではコールバック未着・アプリ層失敗を救えないため）
- タイムアウト検知は APScheduler の常駐ジョブで DB を見るポーリング方式。サーバー再起動を跨ぐため、状態の真実は DB の `jobs.next_retry_at` に置く
- 手動再実行は `retry_count=0` にリセット。自動リトライ枠を作り直す（ユーザーの「もう一度試したい」意図に素直）
- 自動リトライは `60s / 600s` の指数バックオフ × 最大 2 回 → 最終失敗で Resend メール 1 通のみ

## 進捗ログ

- 2026-05-18: タスク移行（個別ファイル化）
- 2026-05-18: 実装完了（マイグレーション・サービス・reaper・API・UI・Orval 再生成まで）。Docker 起動・APScheduler 動作・型チェック OK。完了。
