# デプロイ
## Phase 1: ローカル確認
- [x] docker compose up --build で起動
- [x] ユーザー登録・ログイン動作
- [x] 動画アップロード・処理・再生が動作
- [x] ブラウザコンソールにエラーがない

## Phase 2: 本番サーバ構築
- [-] サーバ（VPS/クラウド）調達
- [-] Docker・docker-compose インストール
- [ ] PostgreSQL コンテナ起動
- [ ] Nginx + Let's Encrypt SSL 設定
- [ ] Backend・Frontend コンテナ起動
- [ ] https://your-domain.com にアクセス可能

## Phase 3: RunPod 本格運用
- [ ] Network Volume 作成・モデルアップロード
- [ ] Endpoint に Volume マウント
- [ ] 環境変数設定（TABLE_MODEL_PATH など）
- [ ] .env.prod で USE_RUNPOD=true に設定
- [ ] 本番データで動作確認

## Phase 4: 本番運用準備
- [ ] SSL 更新自動化（certbot --renew）
- [ ] ログ管理・ローテーション設定
- [ ] 毎日バックアップ実行
- [ ] ヘルスチェック監視開始
- [ ] 利用規約・プライバシーポリシー設定

## Phase 5: 運用開始
- [ ] ユーザーアカウント作成
- [ ] 動画投稿開始
- [ ] エラーログ監視
