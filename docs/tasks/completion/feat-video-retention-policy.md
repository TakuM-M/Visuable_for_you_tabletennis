# feat: 動画の保持ポリシー・容量管理

**Status: Done (2026-05-18)** ※「閾値超過時アラート通知」のみスコープ外（後続タスクとして別途切り出し）

重要度: 中

## 背景・目的

R2 にアップロードされた動画は現状無制限に保持され、容量・コストが青天井。ユーザーごとの上限がないため悪用にも弱い。保持期間とクォータを定義し、自動削除と監視を入れる。

## 受け入れ条件

- [x] 元動画 / 出力動画それぞれに保持期間ポリシーが定義されている
    - `settings.video_retention_days = 7.0` で video レコード単位に一括削除
- [x] 期限切れ動画を自動削除する
    - APScheduler ジョブ `cleanup_expired_videos`（1時間ごと）→ 既存 `video_service.delete_video()` で DB + R2 をクリーン
- [x] ユーザーごとのアップロード制限（容量 GB or 本数）
    - 本数制限（`settings.user_video_quota = 10`）。3つのアップロード経路全てで `_ensure_under_quota()` を呼び、超過は 409 を返す
- [x] R2 使用量・DB レコード数の監視メトリクスが取得できる
    - `GET /admin/metrics`（`X-Internal-Api-Key` 必須）+ 1時間ごとの INFO ログ
- [ ] 閾値超過時にアラート通知（メール / ログ）が飛ぶ
    - **スコープ外**: 別タスク（`feat-storage-alert-notification` 等）に切り出す予定

## 関連ファイル

- `backend/app/services/video_service.py`
- `backend/app/repositories/video_repository.py`
- `backend/app/models/video.py`
- R2 ライフサイクルルール設定（infra 側）

## 設計メモ

- R2 のライフサイクルルールで自動削除するか、backend のバッチで削除するかの方針が必要
  → **採用: backend バッチ**。`delete_video()` が DB（jobs / clips / notification_logs / video）と R2（storage_path / output_path）を一括処理してくれるので再利用。R2 ライフサイクルは二重削除フェイルセーフとして将来 infra 側で別途検討
- ユーザー制限は `users` テーブルに quota カラムを追加する形が素直
  → **採用せず**。全ユーザー一律で十分なため `settings.user_video_quota` で運用。個別上書きが必要になったら `users.video_quota_override` を後付け
- 保存期間については1週間を想定する．
  → `settings.video_retention_days = 7.0` で実装

## 進捗ログ

- 2026-05-18: タスク移行（個別ファイル化）
- 2026-05-18: 実装完了
    - 設定追加: `video_retention_days`, `user_video_quota`, `internal_api_key` ほか（`backend/app/core/config.py`）
    - クォータ enforcement: `_ensure_under_quota()` を `upload_video` / `init_chunk_upload` / `complete_chunk_upload` に追加、router で 409 にマップ
    - 自動削除バッチ: `job_reaper.cleanup_expired_videos`（interval=3600s）
    - 監視メトリクス: `metrics_service.collect_storage_metrics`、`GET /admin/metrics`（`X-Internal-Api-Key` 必須）、`job_reaper.log_storage_metrics`（interval=3600s）
    - マイグレーション: `videos.created_at` インデックス追加（`a1f3c7d5e2b9_add_videos_created_at_index.py`）
    - テスト: 11件追加（quota 4件 / cleanup 3件 / admin metrics 4件）すべて pass
    - 手動検証: `/admin/metrics` で `r2_bytes=42353700 r2_objects=4 db_videos=1` を確認、未認証/誤鍵で 401
    - スコープ外: 閾値超過アラートは別タスク化
