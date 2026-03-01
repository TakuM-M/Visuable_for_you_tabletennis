#!/bin/bash
# Dockerイメージを段階的にビルドするスクリプト（初心者向け）

set -e

# カラー出力用
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# プロジェクトルート
cd "$(dirname "$0")/.."

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Dockerイメージビルド（段階的）${NC}"
echo -e "${BLUE}========================================${NC}"

# ========================================
# Step 1: ベースイメージの確認
# ========================================
echo -e "\n${GREEN}【Step 1】ベースイメージの確認${NC}"
echo "使用するベースイメージ: pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime"
echo "このイメージには以下が含まれています："
echo "  - Python 3.10"
echo "  - PyTorch 2.1.0"
echo "  - CUDA 12.1"
echo "  - cuDNN 8"
echo ""
read -p "続行しますか？ (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${RED}中断しました${NC}"
    exit 1
fi

# ========================================
# Step 2: システム依存パッケージ層のビルド
# ========================================
echo -e "\n${GREEN}【Step 2】システム依存パッケージのインストール${NC}"
echo "以下のパッケージをインストールします："
echo "  - OpenCV依存ライブラリ (libgl1-mesa-glx, libglib2.0-0など)"
echo "  - FFmpeg (動画処理用)"
echo "  - Git, Wget, Curl (ユーティリティ)"
echo ""
read -p "この段階をビルドしますか？ (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}ビルド中...${NC}"
    docker build --target system-deps -t tabletennis-analyzer:system-deps .
    echo -e "${GREEN}✓ Step 2 完了${NC}"
else
    echo -e "${YELLOW}スキップしました${NC}"
fi

# ========================================
# Step 3: Python依存パッケージ層のビルド
# ========================================
echo -e "\n${GREEN}【Step 3】Python依存パッケージのインストール${NC}"
echo "以下のパッケージをインストールします："
echo "  - OpenCV (4.10.0.84)"
echo "  - Ultralytics (YOLO)"
echo "  - NumPy, Pandas, Matplotlib など"
echo ""
echo "注意: この段階は時間がかかります（5-10分程度）"
read -p "この段階をビルドしますか？ (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}ビルド中...${NC}"
    docker build --target python-deps -t tabletennis-analyzer:python-deps .
    echo -e "${GREEN}✓ Step 3 完了${NC}"
else
    echo -e "${YELLOW}スキップしました${NC}"
fi

# ========================================
# Step 4: アプリケーション層のビルド
# ========================================
echo -e "\n${GREEN}【Step 4】アプリケーションのセットアップ${NC}"
echo "以下を実行します："
echo "  - requirements.txtからの追加パッケージインストール"
echo "  - プロジェクトファイルのコピー"
echo "  - 実行権限の設定"
echo ""
read -p "最終イメージをビルドしますか？ (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}ビルド中...${NC}"
    docker build --target app -t tabletennis-analyzer:latest .
    echo -e "${GREEN}✓ Step 4 完了${NC}"
else
    echo -e "${YELLOW}スキップしました${NC}"
fi

# ========================================
# ビルド完了
# ========================================
echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}ビルド完了！${NC}"
echo -e "${BLUE}========================================${NC}"

# イメージサイズの表示
echo -e "\n${GREEN}ビルドされたイメージ:${NC}"
docker images | grep tabletennis-analyzer || echo "イメージが見つかりません"

# 次のステップの案内
echo -e "\n${GREEN}次のステップ:${NC}"
echo "1. ローカルでテスト実行:"
echo "   docker run --gpus all -it tabletennis-analyzer:latest bash"
echo ""
echo "2. Docker Hubにプッシュ（RunPodで使用する場合）:"
echo "   docker tag tabletennis-analyzer:latest <あなたのDocker Hub ID>/tabletennis-analyzer:latest"
echo "   docker push <あなたのDocker Hub ID>/tabletennis-analyzer:latest"
echo ""
echo "3. RunPodでテンプレート作成:"
echo "   - RunPodダッシュボード > Templates > New Template"
echo "   - Container Image: <あなたのDocker Hub ID>/tabletennis-analyzer:latest"
echo "   - Container Disk: 20GB以上推奨"
echo ""

echo -e "${GREEN}詳細は docs/RUNPOD_DOCKER_GUIDE.md を参照してください${NC}"
