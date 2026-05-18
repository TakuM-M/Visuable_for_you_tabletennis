# chore: セキュリティ強化

**Status: Done (2026-05-18)**

重要度: 中

## 背景・目的

公開サービスとして最低限固めるべきセキュリティ項目が未対応。Rate limiting / CORS / 内部 API キーのローテーション / CSRF を整備し、不正アクセス・悪用への耐性を上げる。

## 受け入れ条件

- [x] アップロード系エンドポイントに Rate limiting が掛かっている（per-user / per-IP）
- [x] CORS 設定がレビュー済みで、本番ドメインのみ許可になっている
- [x] `INTERNAL_API_KEY` のローテーション手順が文書化され、無停止で切り替え可能
- [x] CSRF 対策が必要なエンドポイント（特に Cookie ベース部分）を整理し、対応 or 不要判定が済んでいる
- [x] セキュリティ設定が `docs/` 配下にまとまっている

## 関連ファイル

- `backend/app/core/config.py`
- `backend/app/main.py`（CORS / middleware）
- `backend/app/routers/internal.py`
- `nginx/`（本番設定）

## 設計メモ

- Rate limiting は `slowapi` などのライブラリか nginx 側で行うか要検討
- 現状 JWT 認証のみなので CSRF 対策は必須でないが、Cookie 化する将来案も含めて整理

## 進捗ログ

- 2026-05-18: タスク移行（個別ファイル化）
- 2026-05-18: コールバック認証を jobs.py に追加（require_internal_api_key）
- 2026-05-18: CORS allow_headers を明示的リストに変更（Authorization, Content-Type, X-Internal-Api-Key）
- 2026-05-18: nginx で Rate limiting を実装（アップロード系 10req/min、API 60req/min）
- 2026-05-18: docs/security.md を作成（INTERNAL_API_KEY ローテーション手順・CSRF判定の根拠を記載）
- 2026-05-18: 動作確認完了（コールバック認証・CORS ヘッダが正しく機能することを確認）
- 2026-05-18: 完了
- 2026-05-19: RunPod Serverless への INTERNAL_API_KEY 環境変数設定を確認（ml/runpod_handler.py は既に実装済み、RunPod Console 側で Environment Variables に INTERNAL_API_KEY を追加設定が必要）
