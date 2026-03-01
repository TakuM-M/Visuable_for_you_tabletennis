# ========================================
# Stage 1: 依存パッケージのインストール
# ========================================
FROM node:20-slim AS deps

WORKDIR /app

COPY package.json package-lock.json* ./
RUN npm ci

# ========================================
# Stage 2: 開発用（Vite dev server）
# ========================================
FROM deps AS development

WORKDIR /app

# node_modules はコンテナ内のものを使用（docker-compose.yml でボリューム指定済み）
EXPOSE 5173

CMD ["npm", "run", "dev"]

# ========================================
# Stage 3: ビルド
# ========================================
FROM deps AS builder

WORKDIR /app

COPY . .
RUN npm run build

# ========================================
# Stage 4: 本番用（nginx で静的配信）
# ========================================
FROM nginx:1.27-alpine AS production

COPY --from=builder /app/dist /usr/share/nginx/html

# React Router の history モードに対応（全パスを index.html へフォールバック）
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
