# chore: CI/CDパイプライン構築

**Status: In Progress**

重要度: 高

## 背景・目的

現状デプロイが完全手動で、人的ミスのリスクとリリース速度の遅さが課題。GitHub Actions を使い lint / test / build / deploy を自動化することで、変更ごとの品質ゲートを担保しつつデプロイ頻度を上げる。

## 受け入れ条件

- [ ] PR 作成時に backend / frontend / ml の lint が自動実行される
- [ ] PR 作成時に backend (pytest) / ml (pytest) / frontend (型チェック・build) のテストが自動実行される
- [ ] main へのマージで本番デプロイが自動実行される（またはタグ push トリガ）
- [ ] ML イメージ（`takumm/tabletennis-ml`）の Docker build & push が CI から実行できる
- [ ] シークレット（R2/RunPod/DB等）が GitHub Secrets で管理されている
- [ ] CI/CD の README またはセクションがドキュメント化されている

## 関連ファイル

- `.github/workflows/`（新規作成）
- `backend/pyproject.toml` / `frontend/package.json` / `ml/pyproject.toml`
- `Dockerfile`（各サービス）
- `docker-compose.yml` / `docker-compose.dev.yml`

## 設計メモ

- backend / ml は uv ベース → `astral-sh/setup-uv` を使う
- ML イメージは `--target runpod-worker` でビルド、linux/amd64 固定
- デプロイ先（VPS? Cloud Run?）の方針は要決定

## 進捗ログ

- 2026-05-18: タスク移行（個別ファイル化）
