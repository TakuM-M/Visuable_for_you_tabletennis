# ========================================
# Stage 1: ベースイメージ（PyTorch + CUDA）
# ========================================
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime AS base

# 作業ディレクトリ設定
WORKDIR /workspace

# 環境変数設定
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ========================================
# Stage 2: システム依存パッケージのインストール
# ========================================
FROM base AS system-deps

# システムパッケージ更新とインストール
RUN apt-get update && apt-get install -y \
    # OpenCV依存
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # 動画処理
    ffmpeg \
    # その他ユーティリティ
    git \
    wget \
    curl \
    vim \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ========================================
# Stage 3: Python依存パッケージのインストール
# ========================================
FROM system-deps AS python-deps

# pipアップグレード
RUN pip install --no-cache-dir --upgrade pip

# OpenCVインストール
RUN pip install --no-cache-dir \
    opencv-python==4.10.0.84 \
    opencv-contrib-python==4.10.0.84

# YOLO関連
RUN pip install --no-cache-dir ultralytics

# 基本的な科学計算ライブラリ
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    scipy==1.13.0 \
    pandas>=2.0.0 \
    matplotlib==3.9.0 \
    pillow==10.3.0 \
    tqdm==4.66.4

# ========================================
# Stage 4: アプリケーション（最終イメージ）
# ========================================
FROM python-deps AS app

# requirements.txtをコピーしてインストール
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# プロジェクトファイルをコピー
COPY . /workspace/

# 実行権限付与
RUN chmod +x /workspace/scripts/*.sh || true
RUN chmod +x /workspace/scripts/*.py || true

# Jupyter用ポート公開
EXPOSE 8888

# デフォルトコマンド（RunPodではこれが上書きされる）
CMD ["/bin/bash"]
