# ========================================
# Stage 1: ベースイメージ（PyTorch + CUDA）
# ========================================
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime AS base

WORKDIR /workspace

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# uv を取得（backend と同様）
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# base image の Python をそのまま使う（仮想環境を作らない）
ENV UV_SYSTEM_PYTHON=1

# ========================================
# Stage 2: システム依存パッケージのインストール
# ========================================
FROM base AS system-deps

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

# 依存ファイルをコピー
COPY pyproject.toml uv.lock* ./

# 本番依存のみインストール（dev グループを除外）
# torch / torchvision は base image 提供のため uv がスキップする
RUN uv sync --no-cache --no-dev

# uv sync は /workspace/.venv/ に依存をインストールするため
# venv の bin を PATH の先頭に追加し、python コマンドが venv を指すようにする
ENV PATH="/workspace/.venv/bin:$PATH"

# ========================================
# Stage 4: ML開発環境
# ========================================
FROM python-deps AS app

COPY . /workspace/

RUN chmod +x /workspace/scripts/*.sh 2>/dev/null || true

EXPOSE 8888

CMD ["/bin/bash"]

# ========================================
# Stage 5: ML推論サービス（FastAPI）
# ========================================
FROM python-deps AS service

COPY . /workspace/

EXPOSE 8001

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]

# ========================================
# Stage 6: Mock サービス（ローカル開発用・軽量）
# ML 依存なし。fastapi/httpx のみ。
# ========================================
FROM python:3.10-slim AS mock

WORKDIR /workspace

RUN pip install --no-cache-dir fastapi uvicorn[standard] httpx

COPY mock_app.py .

EXPOSE 8001

CMD ["uvicorn", "mock_app:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]

# ========================================
# Stage 7: RunPod Serverless ワーカー
# ========================================
FROM python-deps AS runpod-worker

COPY . /workspace/

# モデルはRunPod Network Volumeをマウントして使用する
# 環境変数で各モデルパスを指定:
#   TABLE_MODEL_PATH, POSE_MODEL_PATH,
#   PLAY_CLASSIFIER_MODEL_PATH, PLAY_CLASSIFIER_CONFIG_PATH
#   ML_DEVICE (default: cuda)

CMD ["python", "-u", "runpod_handler.py"]
