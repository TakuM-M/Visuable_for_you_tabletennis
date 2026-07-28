# ========================================
# Stage 1: ベースイメージ（Python のみ）
#
# CUDA ランタイムは torch の cu124 wheel が依存する nvidia-*-cu12 パッケージが
# 同梱するため、pytorch/pytorch イメージ（CUDA 12.1 + torch 2.1.0）は使わない。
# pyproject.toml で cu124 を明示している以上 base image 側の torch/CUDA は
# 一度も読み込まれず、約 7GB の死荷重になるだけだった。
# コンテナに必要なのはホスト側の NVIDIA ドライバのみ（RunPod が提供する）。
#
# Debian は bookworm を明示ピンする。trixie では libglib2.0-0 が
# libglib2.0-0t64 に、bookworm 未満では libgl1 が libgl1-mesa-glx になるなど
# パッケージ名が変わり、タグ追従で apt が壊れるため。
# ========================================
FROM python:3.10-slim-bookworm AS base

WORKDIR /workspace

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# uv を取得（backend と同様）
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# イメージ同梱の Python を使う（uv 管理 Python を追加ダウンロードさせない）
ENV UV_PYTHON=/usr/local/bin/python3.10
ENV UV_PYTHON_DOWNLOADS=never

# ========================================
# Stage 2: システム依存パッケージのインストール
# ========================================
FROM base AS system-deps

# libgl1 / libglib2.0-0 / libsm6 / libxext6 / libxrender1: OpenCV の実行時依存
# libgomp1: OpenMP ランタイム
# ffmpeg: 動画コーデック
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ========================================
# Stage 3: Python依存パッケージのインストール
# ========================================
FROM system-deps AS python-deps

# 依存ファイルをコピー
COPY pyproject.toml uv.lock* ./

# 推論に必要な依存のみ（dev / training グループは含めない）。
# uv のダウンロードキャッシュは BuildKit キャッシュマウントに置く。
# イメージには残らず、再ビルド時の wheel 再取得（torch + CUDA で数 GB）を省ける。
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-default-groups

# uv sync は /workspace/.venv/ に依存をインストールするため
# venv の bin を PATH の先頭に追加し、python コマンドが venv を指すようにする
ENV PATH="/workspace/.venv/bin:$PATH"

# ========================================
# Stage 4: ML開発環境（学習・実験用）
# ========================================
FROM python-deps AS app

# 開発時にだけ欲しいユーティリティ。推論イメージ（runpod-worker）には入れない
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    vim \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 学習でのみ使う依存（tensorboard）と dev 依存を追加する
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync

COPY . /workspace/

RUN chmod +x /workspace/scripts/*.sh 2>/dev/null || true

EXPOSE 8888

CMD ["/bin/bash"]

# ========================================
# Stage 5: Mock サービス（ローカル開発用・軽量）
# ML 依存なし。fastapi/httpx のみ。
# ========================================
FROM python:3.10-slim-bookworm AS mock

WORKDIR /workspace

RUN pip install --no-cache-dir fastapi uvicorn[standard] httpx

COPY mock_app.py .

EXPOSE 8001

CMD ["uvicorn", "mock_app:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]

# ========================================
# Stage 6: RunPod Serverless ワーカー
# ========================================
FROM python-deps AS runpod-worker

COPY . /workspace/

# モデルはRunPod Network Volumeをマウントして使用する
# 環境変数で各モデルパスを指定:
#   TABLE_MODEL_PATH, POSE_MODEL_PATH,
#   PLAY_CLASSIFIER_MODEL_PATH, PLAY_CLASSIFIER_CONFIG_PATH
#   ML_DEVICE (default: cuda)

CMD ["python", "-u", "runpod_handler.py"]
