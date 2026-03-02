# RunPod環境セットアップガイド（Docker版）

このドキュメントは、**Dockerイメージ**を使ってRunPodのリモートGPU環境で卓球動画解析スクリプトを実行するための、**初心者向け**セットアップ手順を説明します。

## 📚 このガイドの使い方

- **Docker初心者の方**: ステップ1から順番に進めてください
- **Docker経験者の方**: ステップ3から開始できます
- **トラブル時**: 最後のトラブルシューティングセクションを参照

---

## ステップ1: Dockerイメージをビルドする（ローカル）

### 1-1. 前提条件の確認

ローカルマシンに以下がインストールされていることを確認：

```bash
# Dockerのバージョン確認
docker --version
# 出力例: Docker version 24.0.0

# Git確認（プロジェクト取得用）
git --version
```

**インストールされていない場合:**
- **Docker Desktop**: https://www.docker.com/products/docker-desktop
- **Git**: https://git-scm.com/downloads

### 1-2. プロジェクトを取得

```bash
# ホームディレクトリまたは任意の作業ディレクトリに移動
cd ~/workspace

# プロジェクトをクローン（または既にある場合はスキップ）
git clone <your-repository-url>
cd Visuable_for_you_tabletennis

# プロジェクト構造を確認
ls -la
```

### 1-3. Dockerイメージをビルド

#### 📘 方法A: 段階的ビルド（初心者向け・推奨）

対話式スクリプトで各ステップを確認しながらビルド：

```bash
# 実行権限を付与
chmod +x scripts/docker_build_step_by_step.sh

# ビルド開始（各ステップで確認が入ります）
bash scripts/docker_build_step_by_step.sh
```

**ビルドの4段階:**

1. **ベースイメージ確認** - PyTorch + CUDAの確認
2. **システムパッケージ** - OpenCV、FFmpegなど（3-5分）
3. **Pythonパッケージ** - NumPy、YOLOなど（5-10分）
4. **アプリケーション** - プロジェクトファイルのコピー（1-2分）

#### 📘 方法B: 一括ビルド（経験者向け）

全て一気にビルド：

```bash
# 全ステージを一括ビルド
docker build -t tabletennis-analyzer:latest .

# ビルド時間: 約10-15分（インターネット速度による）
```

### 1-4. ビルド確認

```bash
# イメージが作成されたか確認
docker images | grep tabletennis-analyzer

# 出力例:
# tabletennis-analyzer  latest  abc123def456  2 minutes ago  8.5GB
```

### 1-5. ローカルでテスト（オプション）

```bash
# CPUモードでテスト起動
docker run -it --rm tabletennis-analyzer:latest bash

# コンテナ内で確認
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
exit
```

---

## ステップ2: Docker Hubにプッシュする

RunPodで使用するために、ビルドしたイメージをDocker Hubにアップロードします。

### 2-1. Docker Hubアカウント作成

1. https://hub.docker.com/ にアクセス
2. **Sign Up** から無料アカウント作成
3. ユーザー名を控えておく（例: `takumi123`）

### 2-2. Docker Hubにログイン

```bash
# ターミナルでログイン
docker login

# 入力を求められます
Username: your-dockerhub-username
Password: your-password

# 成功すると: Login Succeeded
```

### 2-3. イメージにタグ付け

```bash
# あなたのDocker Hub ユーザー名に置き換える
docker tag tabletennis-analyzer:latest your-dockerhub-username/tabletennis-analyzer:latest

# 例:
# docker tag tabletennis-analyzer:latest takumi123/tabletennis-analyzer:latest
```

### 2-4. Docker Hubにプッシュ

```bash
# アップロード開始
docker push your-dockerhub-username/tabletennis-analyzer:latest

# プッシュの進捗が表示されます
# 完了まで10-30分程度（回線速度による）
```

**注意:**
- イメージサイズが大きい（約8GB）ため、時間がかかります
- WiFi環境推奨（モバイル回線だと通信量に注意）
- 完了まで放置してOK

### 2-5. プッシュ確認

```bash
# Docker Hubで確認
# https://hub.docker.com/r/your-dockerhub-username/tabletennis-analyzer
```

---

## ステップ3: RunPodでテンプレートを作成

### 3-1. RunPodアカウント作成とログイン

1. https://www.runpod.io/ にアクセス
2. アカウント作成（GitHub連携が簡単）
3. ダッシュボードにログイン

### 3-2. テンプレート作成

#### 手順:

1. 左メニュー: **Templates** をクリック
2. 右上: **+ New Template** をクリック
3. 以下の項目を入力:

```
┌─────────────────────────────────────────────────┐
│ Template Configuration                          │
├─────────────────────────────────────────────────┤
│ Template Name:                                  │
│   TableTennis Analyzer                          │
│                                                 │
│ Container Image:                                │
│   your-dockerhub-username/tabletennis-analyzer:latest │
│   例: takumi123/tabletennis-analyzer:latest     │
│                                                 │
│ Container Disk:                                 │
│   20 GB （推奨: 30GB）                           │
│                                                 │
│ Docker Command: （空欄でOK）                     │
│   /bin/bash                                     │
│                                                 │
│ Expose HTTP Ports: （Jupyter使う場合）          │
│   8888                                          │
│                                                 │
│ Environment Variables: （不要）                  │
│   （空欄でOK）                                   │
└─────────────────────────────────────────────────┘
```

4. **Save Template** をクリック

### 3-3. GPU選択ガイド

推論に適したGPUと性能目安:

| GPU | VRAM | 処理速度目安 | コスト（目安/時間） | おすすめ度 |
|-----|------|-------------|-------------------|-----------|
| **RTX 3060** | 12GB | 1倍速 | $0.3-0.4 | ⭐⭐⭐ コスパ良 |
| RTX 3070 | 8GB | 1.5倍速 | $0.35-0.45 | ⭐⭐ |
| **RTX 3080** | 10GB | 2-3倍速 | $0.4-0.5 | ⭐⭐⭐ |
| RTX 4080 | 16GB | 3-4倍速 | $0.6-0.7 | ⭐⭐ |
| **RTX 4090** | 24GB | 4-5倍速 | $0.8-1.0 | ⭐⭐⭐ 高速 |

**選び方:**
- **初めての方**: RTX 3060（安くて安定）
- **速度重視**: RTX 4090
- **最小要件**: VRAM 8GB以上

---

## ステップ4: Podを起動して推論実行

### 4-1. Pod起動

1. ダッシュボード: **Pods** > **+ Deploy** をクリック
2. **Template** タブ: 先ほど作成した "TableTennis Analyzer" を選択
3. **GPU Type**: 推奨GPUを選択（例: RTX 3060）
4. **Deploy On-Demand** または **Deploy Spot** をクリック
   - On-Demand: 安定、少し高い
   - Spot: 安い、中断リスクあり（初心者はOn-Demand推奨）

5. 起動を待つ（1-3分）

### 4-2. Podに接続

起動したPodの **Connect** > **Start Web Terminal** をクリック

または SSH接続:
```bash
# ローカルターミナルで（RunPodが提供するSSHコマンドをコピペ）
ssh root@<pod-id>.runpod.io -p <port>
```

### 4-3. 環境確認

```bash
# 作業ディレクトリに移動
cd /workspace

# GPU確認
nvidia-smi

# Python環境確認
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

出力例:
```
CUDA available: True
GPU: NVIDIA GeForce RTX 3060
```

### 4-4. データのアップロード

#### 方法A: RunPodのファイルアップロード機能

1. Web Terminal の **File Browser** を開く
2. `/workspace/data/raw/` ディレクトリを作成
3. 動画ファイルをドラッグ&ドロップ

#### 方法B: SCP経由（ローカルから）

```bash
# ローカルターミナルで
scp -P <port> /path/to/video.MOV root@<pod-id>.runpod.io:/workspace/data/raw/
```

#### 方法C: Google Drive/Dropbox URLから

```bash
# Pod内で
cd /workspace/data/raw
wget "https://your-file-url.com/video.MOV"
```

### 4-5. モデルファイルの配置

**事前にモデルを訓練済みの場合:**

```bash
# ローカルからアップロード
scp -P <port> models/table_detection/best.pt root@<pod-id>.runpod.io:/workspace/models/table_detection/
scp -P <port> models/play_classifier/lstm_model.pth root@<pod-id>.runpod.io:/workspace/models/play_classifier/
```

**YOLOポーズモデル（自動ダウンロード）:**
- 初回実行時にUltralyticsが自動でダウンロードします

### 4-6. 推論実行

```bash
cd /workspace

# 推論スクリプト実行
python scripts/run_inference_runpod.py \
  --input data/raw/sample_video.MOV \
  --output output/sample_video \
  --config configs/crip_app_config.json

# 処理時間: 動画の長さとGPUによる（5-30分程度）
```

**実行中の画面:**
```
==================================================
環境チェック
==================================================
Python version: 3.10.x
PyTorch version: 2.1.0
CUDA available: True
GPU: NVIDIA GeForce RTX 3060
==================================================

Processing: 100%|███████████████| 5000/5000 [10:23<00:00, 48.2frames/s]
```

### 4-7. 結果のダウンロード

#### 方法A: File Browser（簡単）

1. Web Terminalの **File Browser** を開く
2. `output/` ディレクトリを開く
3. ファイルを選択してダウンロード

#### 方法B: SCP（複数ファイル）

```bash
# ローカルターミナルで
scp -r -P <port> root@<pod-id>.runpod.io:/workspace/output/sample_video ./local_output/
```

**生成されるファイル:**
```
output/sample_video/
├── sample_video_pose.csv              # 骨格データ（CSV）
├── sample_video_pose.mp4              # 骨格可視化動画
├── sample_video_scenes.json           # シーン情報（JSON）
├── sample_video_play_scenes.mp4       # ハイライト動画
└── sample_video_prediction_graph.png  # 予測グラフ
```

### 4-8. Pod停止（重要！）

**必ず停止してコスト削減:**

```bash
# 作業完了後、RunPodダッシュボードで:
# Pods > 該当Pod > Stop Pod
```

または自動停止設定:
```
Pods > Edit Pod > Auto-stop: 1 hour idle
```

---

## ステップ5: 応用編

### バッチ処理（複数動画）

```bash
# 複数動画を一括処理
cd /workspace

for video in data/raw/*.MOV; do
    basename=$(basename "$video" .MOV)
    python scripts/run_inference_runpod.py \
        --input "$video" \
        --output "output/${basename}"
done
```

### Jupyter Notebookで実行

```bash
# Jupyter起動
cd /workspace
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# ブラウザでアクセス（RunPodのポートフォワーディング経由）
# notebooks/05_crip.ipynb を開いて実行
```

### 設定ファイルのカスタマイズ

```bash
# 設定を編集
vi configs/crip_app_config.json

# 例: FPSを下げてメモリ節約
{
  "video_processing": {
    "target_fps": 15.0  // 30.0 → 15.0
  }
}
```

---

## トラブルシューティング

### 1. CUDA Out of Memory エラー

**症状:**
```
RuntimeError: CUDA out of memory. Tried to allocate XXX MiB
```

**解決策:**

**方法A: FPSを下げる**
```json
// configs/crip_app_config.json
{
  "video_processing": {
    "target_fps": 15.0  // デフォルト: 30.0
  }
}
```

**方法B: より軽量なモデルを使用**
```json
{
  "models": {
    "pose_estimation": "models/pretrained/yolo11m-pose.pt"  // l → m
  }
}
```

**方法C: より多くのVRAMを持つGPUに変更**
- RTX 4090 (24GB) など

### 2. OpenCV エラー（libGL.so.1 not found）

**症状:**
```
ImportError: libGL.so.1: cannot open shared object file
```

**解決策:**
```bash
# システムパッケージ再インストール
apt-get update
apt-get install -y libgl1-mesa-glx
```

### 3. モデルファイルが見つからない

**症状:**
```
FileNotFoundError: models/table_detection/best.pt
```

**解決策:**
```bash
# ディレクトリ構造を確認
ls -la models/

# 存在しない場合はアップロード
# または設定ファイルのパスを修正
```

### 4. Docker Hubへのプッシュが遅い

**対策:**
- 有線LAN接続を使う
- 深夜など回線が空いている時間帯に実行
- Docker Hubの別リージョンを試す

### 5. RunPodでPod起動が失敗

**原因:**
- 選択したGPUが利用不可
- Container Diskが小さすぎる

**解決策:**
- 別のGPUを選択
- Container Diskを30GB以上に設定

### 6. 推論が途中で止まる

**原因:**
- Spot Instanceが中断された
- メモリ不足

**解決策:**
- On-Demand Podを使用
- より高性能なGPUに変更

---

## コスト最適化Tips

### 1. Spot Instanceを活用

```
On-Demand: $0.4/h
Spot: $0.2/h （最大50%安い）
```

リスク: 需要が高いと中断される可能性

### 2. 処理完了後すぐ停止

```bash
# 処理完了後、手動停止またはスクリプトで自動停止
python scripts/run_inference_runpod.py ... && shutdown -h now
```

### 3. データの前処理

```bash
# ローカルで動画を圧縮してからアップロード
ffmpeg -i input.MOV -vf scale=1280:720 -c:v libx264 -crf 23 output.mp4
```

### 4. バッチ処理でまとめて実行

複数動画を一度の起動で処理する

---

## 参考資料

- **RunPod公式ドキュメント**: https://docs.runpod.io/
- **Docker公式ドキュメント**: https://docs.docker.com/
- **PyTorch CUDA互換性**: https://pytorch.org/get-started/locally/
- **Ultralytics YOLO**: https://docs.ultralytics.com/

---

## よくある質問（FAQ）

### Q1. Dockerイメージのビルドに失敗する

**A:** ディスク容量を確認してください（最低20GB必要）

```bash
df -h
```

### Q2. RunPodのコストはどのくらい？

**A:**
- RTX 3060: $0.3-0.4/時間
- 1時間の動画処理: 約30分 = $0.15-0.2
- 月に10時間使用: $3-4程度

### Q3. ローカルマシンにGPUがなくてもビルドできる？

**A:** はい、CPUだけでビルド可能です。RunPod上でGPUを使用します。

### Q4. Docker Hubを使わずに済む方法は？

**A:** RunPodはDocker Hubからのpullが基本なので推奨されませんが、sshで直接ファイルを転送する方法もあります（非効率）。

### Q5. 既存のDockerイメージを更新したい

**A:**
```bash
# ローカルで再ビルド
docker build -t tabletennis-analyzer:latest .

# 新しいタグでプッシュ
docker tag tabletennis-analyzer:latest your-username/tabletennis-analyzer:v2
docker push your-username/tabletennis-analyzer:v2

# RunPodのテンプレートを更新
```

---

## 次のステップ

1. ✅ Dockerイメージビルド完了
2. ✅ Docker Hubにプッシュ完了
3. ✅ RunPodで推論実行完了

**さらに学ぶ:**
- Dockerfileのカスタマイズ
- モデルの再トレーニング
- APIサーバー化（FastAPI）
- Web UIの追加

---

**サポートが必要な場合:**
- GitHub Issues: <your-repo-url>/issues
- プロジェクトドキュメント: `docs/` ディレクトリ
