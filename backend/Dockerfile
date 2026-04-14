# ========================================
# Stage 1: 依存パッケージのインストール
# ========================================
FROM python:3.12-slim AS deps

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml uv.lock* ./

# 本番依存のみインストール（dev グループを除外）
RUN uv sync --no-cache --no-dev

# ========================================
# Stage 2: アプリケーション（開発用）
# ========================================
FROM deps AS development

WORKDIR /app

# 開発時は dev グループも追加インストール
RUN uv sync --no-cache

# アプリコードは docker-compose.yml のボリュームでマウント
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# ========================================
# Stage 3: アプリケーション（本番用）
# ========================================
FROM deps AS production

WORKDIR /app

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
