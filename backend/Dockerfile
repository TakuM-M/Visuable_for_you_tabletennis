# ========================================
# Stage 1: 依存パッケージのインストール
# ========================================
FROM python:3.12-slim AS deps

WORKDIR /app

# システムパッケージ（psycopg2のビルドに必要）
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ========================================
# Stage 2: アプリケーション（開発用）
# ========================================
FROM deps AS development

WORKDIR /app

# アプリコードは docker-compose.yml のボリュームでマウントするため COPY 不要
# uvicorn --reload でホットリロードを有効化
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# ========================================
# Stage 3: アプリケーション（本番用）
# ========================================
FROM deps AS production

WORKDIR /app

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
