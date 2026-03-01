# 🏓 卓球動画解析 - Docker & RunPod クイックスタート

このプロジェクトをRunPodのリモートGPU環境で実行するための簡易ガイドです。

## 📋 概要

このプロジェクトは卓球動画から以下を自動で行います：

1. **卓球台の検出**
2. **プレイヤーの骨格検出・追跡**
3. **プレーシーンの分類**（プレー中/非プレー中）
4. **ハイライト動画の自動生成**

## 🚀 クイックスタート（3ステップ）

### 1️⃣ Dockerイメージをビルド（ローカル）

```bash
# プロジェクトディレクトリで
docker build -t tabletennis-analyzer:latest .
docker build --platform linux/amd64 -t tabletennis-analyzer:latest .
```

### 2️⃣ Docker Hubにプッシュ

```bash
# ログイン
docker login

# タグ付け（your-username を自分のIDに変更）
docker tag tabletennis-analyzer:latest your-username/tabletennis-analyzer:latest
docker tag tabletennis-analyzer:latest takumm/tabletennis-analyzer:latest

# プッシュ
docker push your-username/tabletennis-analyzer:latest
docker push takumm/tabletennis-analyzer:latest
```

### 3️⃣ RunPodで実行

1. [RunPod](https://www.runpod.io/)にログイン
2. **Templates** > **New Template**
   - Container Image: `your-username/tabletennis-analyzer:latest`
   - Container Image: `takumm
   /tabletennis-analyzer:latest`
   - Container Disk: 20GB以上
3. **Deploy Pod** with RTX 3060以上
4. Pod起動後、ターミナルで:

```bash
cd /workspace

# 動画をアップロードして推論実行
python scripts/run_inference_runpod.py \
  --input data/raw/your_video.MOV \
  --output output/result
```

## 📚 詳細ドキュメント

初めての方やトラブル時は、詳細ガイドを参照してください：

**[📖 RunPod Docker 完全ガイド（初心者向け）](docs/RUNPOD_DOCKER_GUIDE.md)**

以下の内容を網羅しています：
- ステップバイステップの手順
- 段階的なDockerビルド方法
- RunPodのテンプレート作成
- GPU選択ガイド
- トラブルシューティング
- コスト最適化Tips

## 📁 プロジェクト構成

```
.
├── Dockerfile                        # Dockerイメージ定義
├── docker-compose.yml                # ローカル開発用
├── .dockerignore                     # Dockerビルド除外ファイル
├── scripts/
│   ├── docker_build_step_by_step.sh # 段階的ビルドスクリプト
│   ├── run_inference_runpod.py      # RunPod用推論スクリプト
│   └── setup_runpod.sh              # 環境セットアップ
├── src/                              # ソースコード
├── configs/                          # 設定ファイル
├── models/                           # 学習済みモデル
└── docs/
    └── RUNPOD_DOCKER_GUIDE.md       # 詳細ガイド
```

## 🔧 ローカル開発（docker-compose）

```bash
# 開発環境起動
docker-compose up -d dev

# コンテナに入る
docker-compose exec dev bash

# Jupyter起動
docker-compose up jupyter
# ブラウザで http://localhost:8888
```

## 💰 コスト目安

| GPU | 処理速度 | コスト/時間 | 30分動画の処理コスト |
|-----|---------|-----------|-------------------|
| RTX 3060 | 1倍速 | $0.3-0.4 | $0.15-0.2 |
| RTX 4090 | 4倍速 | $0.8-1.0 | $0.1-0.15 |

## 🛠️ トラブルシューティング

### CUDA Out of Memory

```json
// configs/crip_app_config.json で FPS を下げる
{
  "video_processing": {
    "target_fps": 15.0  // 30.0 → 15.0
  }
}
```

### その他のエラー

[詳細ガイドのトラブルシューティング](docs/RUNPOD_DOCKER_GUIDE.md#トラブルシューティング)を参照

## 📖 関連ドキュメント

- **[RunPod Docker完全ガイド](docs/RUNPOD_DOCKER_GUIDE.md)** - 初心者向け詳細手順
- **[アーキテクチャ](docs/architecture/)** - システム設計
- **[コンポーネント](docs/components/)** - 各モジュール詳細

## 🤝 サポート

質問やバグ報告は以下へ：
- GitHub Issues
- プロジェクトドキュメント

## 📝 ライセンス

[プロジェクトライセンスを記載]
