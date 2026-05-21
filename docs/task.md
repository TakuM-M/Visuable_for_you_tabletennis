# Tasks

> 個別ファイルは [`tasks/`](./tasks/) 配下。重い／複数セッション必要なもののみ分割しています。
> 元の単一ファイル版は [`task.md.bak`](./task.md.bak) にバックアップ。

## インフラ / CI/CD
- [ ] [CI/CDパイプライン構築](./tasks/chore-cicd-pipeline.md) - 重要度: 高
- [ ] dockerについて，GPU割り当てによっては対応できていないPytorch version

## バックエンド
<!-- 未完了タスクなし -->


## セキュリティ
<!-- 未完了タスクなし -->

## ML / アルゴリズム
- [ ] [卓球台検出ロジックの堅牢化](./tasks/fix-ml-table-detection-robustness.md) - 重要度: 中

## UX
- []フロントでstatus変更が生じない場合に詰むという問題

## 完了
- [x] ナビゲーションタブの二重アンダーバー表示バグ修正 - 2026-05-20
- [x] [UX改善](./tasks/completion/feat-ux-improvements.md) - 2026-05-20
- [x] [セキュリティ強化](./tasks/completion/chore-security-hardening.md) - 2026-05-18
- [x] [動画の保持ポリシー・容量管理](./tasks/completion/feat-video-retention-policy.md) - 2026-05-18 ※アラート通知のみスコープ外
- [x] [ジョブ耐障害性・エラーハンドリング強化](./tasks/completion/feat-job-resilience.md) - 2026-05-18
- [x] ストレージのオブジェクトストレージ移行（Cloudflare R2） - 2026-05-18 (PR #6)
- [x] メール新規登録後の再登録不可問題 - 2026-05-18
