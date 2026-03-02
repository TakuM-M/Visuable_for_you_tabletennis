import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",   // Dockerコンテナ外からアクセスできるようにする
    port: 5173,
    proxy: {
      "/api": {
        target: "http://backend:8000",  // docker-compose のサービス名
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),  // /api を除去して転送
      },
    },
  },
})
