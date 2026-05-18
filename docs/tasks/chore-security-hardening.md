# chore: セキュリティ強化

**Status: In Progress**

重要度: 中

## 背景・目的

公開サービスとして最低限固めるべきセキュリティ項目が未対応。Rate limiting / CORS / 内部 API キーのローテーション / CSRF を整備し、不正アクセス・悪用への耐性を上げる。

## 受け入れ条件

- [ ] アップロード系エンドポイントに Rate limiting が掛かっている（per-user / per-IP）
- [ ] CORS 設定がレビュー済みで、本番ドメインのみ許可になっている
- [ ] `INTERNAL_API_KEY` のローテーション手順が文書化され、無停止で切り替え可能
- [ ] CSRF 対策が必要なエンドポイント（特に Cookie ベース部分）を整理し、対応 or 不要判定が済んでいる
- [ ] セキュリティ設定が `docs/` 配下にまとまっている

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
