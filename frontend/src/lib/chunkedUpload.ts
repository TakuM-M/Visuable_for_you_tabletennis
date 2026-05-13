import { authHeaders } from "./auth";

const CHUNK_SIZE = 50 * 1024 * 1024; // 50MB

export interface ChunkedUploadOptions {
  file: File;
  title: string;
  onProgress?: (percent: number) => void;
  signal?: AbortSignal;
}

export async function chunkedUpload({
  file,
  title,
  onProgress,
  signal,
}: ChunkedUploadOptions) {
  const totalChunks = Math.ceil(file.size / CHUNK_SIZE);
  const headers = authHeaders();

  // 1. Init
  const initRes = await fetch("/api/videos/upload/init", {
    method: "POST",
    headers: { "Content-Type": "application/json", ...headers },
    body: JSON.stringify({
      title,
      filename: file.name,
      total_chunks: totalChunks,
    }),
    signal,
  });
  if (!initRes.ok) {
    throw new Error(`Init failed: ${initRes.status}`);
  }
  const { upload_id } = (await initRes.json()) as { upload_id: string };

  // 2. Upload chunks
  for (let i = 0; i < totalChunks; i++) {
    const start = i * CHUNK_SIZE;
    const end = Math.min(start + CHUNK_SIZE, file.size);
    const chunk = file.slice(start, end);

    const formData = new FormData();
    formData.append("file", chunk, file.name);

    const chunkRes = await fetch(
      `/api/videos/upload/${upload_id}/chunk?index=${i}`,
      {
        method: "POST",
        headers: { ...headers },
        body: formData,
        signal,
      },
    );
    if (!chunkRes.ok) {
      throw new Error(`Chunk ${i} failed: ${chunkRes.status}`);
    }

    onProgress?.(Math.round(((i + 1) / totalChunks) * 100));
  }

  // 3. Complete
  const completeRes = await fetch(
    `/api/videos/upload/${upload_id}/complete`,
    {
      method: "POST",
      headers: { ...headers },
      signal,
    },
  );
  if (!completeRes.ok) {
    throw new Error(`Complete failed: ${completeRes.status}`);
  }

  return (await completeRes.json()) as { id: string };
}
