import { defineConfig } from "orval";

export default defineConfig({
  api: {
    input: {
      // FastAPI が自動生成する OpenAPI スキーマの URL
      target: "http://localhost:8000/openapi.json",
    },
    output: {
      // 生成したクライアントコードの出力先
      target: "src/api/generated.ts",
      // axios ではなく fetch を使う
      client: "fetch",
      // baseUrl はすべてのリクエストに付与されるプレフィックス
      // Vite proxy で /api → backend:8000 に転送されるため /api を指定
      baseUrl: "/api",
      override: {
        fetch: {
          includeHttpStatusReturnType: false,
        },
      },
    },
  },
});
