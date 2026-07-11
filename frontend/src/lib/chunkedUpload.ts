import { authHeaders } from "./auth";

const CHUNK_SIZE = 50 * 1024 * 1024; // 50MB
const MAX_ATTEMPTS = 4; // 初回 + リトライ3回
const RETRY_BASE_DELAY_MS = 1000; // リトライ間隔（指数バックオフの初期値）

export interface ChunkedUploadOptions {
  file: File;
  title: string;
  onProgress?: (percent: number) => void;
  signal?: AbortSignal;
}

// HTTP ステータス付きエラー。リトライ可否の判定に使う。
class UploadHttpError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

const abortError = () => new DOMException("Aborted", "AbortError");

const isAbortError = (e: unknown): boolean =>
  e instanceof DOMException && e.name === "AbortError";

// 一時的な失敗（ネットワークエラー・5xx・429）のみリトライ対象とする。
// モバイル回線の瞬断や画面ロックによる切断、レート制限の 503 をここで救う。
const isTransient = (e: unknown): boolean =>
  !isAbortError(e) &&
  (!(e instanceof UploadHttpError) || e.status >= 500 || e.status === 429);

// complete は再実行すると動画が二重登録されうるため、
// backend に到達していないことが確実な失敗（レート制限の 503 / 429）のみリトライする。
const isSafeToRetryComplete = (e: unknown): boolean =>
  e instanceof UploadHttpError && (e.status === 503 || e.status === 429);

// signal が abort されたら即座に中断する sleep
const sleep = (ms: number, signal?: AbortSignal) =>
  new Promise<void>((resolve, reject) => {
    if (signal?.aborted) {
      reject(abortError());
      return;
    }
    const onAbort = () => {
      clearTimeout(timer);
      reject(abortError());
    };
    const timer = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    signal?.addEventListener("abort", onAbort, { once: true });
  });

async function withRetry<T>(
  fn: () => Promise<T>,
  shouldRetry: (e: unknown) => boolean,
  signal?: AbortSignal,
): Promise<T> {
  for (let attempt = 1; ; attempt++) {
    try {
      return await fn();
    } catch (e) {
      if (attempt >= MAX_ATTEMPTS || !shouldRetry(e)) throw e;
      await sleep(RETRY_BASE_DELAY_MS * 2 ** (attempt - 1), signal);
    }
  }
}

// XMLHttpRequest でチャンクを送信する。fetch では取れない送信バイト数の進捗
// （upload.onprogress）を取得するため。低速なモバイル回線でも進捗表示が
// チャンク完了までの数十秒間 0% のまま止まらないようにする。
function sendChunk(
  url: string,
  label: string,
  chunk: Blob,
  filename: string,
  onChunkProgress: (ratio: number) => void,
  signal?: AbortSignal,
): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(abortError());
      return;
    }
    const xhr = new XMLHttpRequest();
    const onAbort = () => xhr.abort();

    xhr.open("POST", url);
    for (const [key, value] of Object.entries(
      authHeaders() as Record<string, string>,
    )) {
      xhr.setRequestHeader(key, value);
    }
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && e.total > 0) onChunkProgress(e.loaded / e.total);
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) resolve();
      else reject(new UploadHttpError(`${label} failed: ${xhr.status}`, xhr.status));
    };
    xhr.onerror = () => reject(new Error(`${label} failed: network error`));
    xhr.onabort = () => reject(abortError());
    xhr.onloadend = () => signal?.removeEventListener("abort", onAbort);
    signal?.addEventListener("abort", onAbort, { once: true });

    const formData = new FormData();
    formData.append("file", chunk, filename);
    xhr.send(formData);
  });
}

export async function chunkedUpload({
  file,
  title,
  onProgress,
  signal,
}: ChunkedUploadOptions) {
  const totalChunks = Math.ceil(file.size / CHUNK_SIZE);

  // 進捗は「送信済みバイト / 全体バイト」で報告する。
  // リトライでチャンクを送り直しても表示が巻き戻らないよう単調増加に丸める。
  let reportedPercent = 0;
  const reportProgress = (uploadedBytes: number) => {
    const percent = Math.min(100, Math.round((uploadedBytes / file.size) * 100));
    if (percent > reportedPercent) {
      reportedPercent = percent;
      onProgress?.(percent);
    }
  };

  // 1. Init
  const initRes = await withRetry(
    () =>
      fetch("/api/videos/upload/init", {
        method: "POST",
        headers: { "Content-Type": "application/json", ...authHeaders() },
        body: JSON.stringify({
          title,
          filename: file.name,
          total_chunks: totalChunks,
        }),
        signal,
      }).then((res) => {
        if (!res.ok) throw new UploadHttpError(`Init failed: ${res.status}`, res.status);
        return res;
      }),
    isTransient,
    signal,
  );
  const { upload_id } = (await initRes.json()) as { upload_id: string };

  // 2. Upload chunks
  for (let i = 0; i < totalChunks; i++) {
    const start = i * CHUNK_SIZE;
    const end = Math.min(start + CHUNK_SIZE, file.size);
    const chunk = file.slice(start, end);

    await withRetry(
      () =>
        sendChunk(
          `/api/videos/upload/${upload_id}/chunk?index=${i}`,
          `Chunk ${i}`,
          chunk,
          file.name,
          (ratio) => reportProgress(start + ratio * (end - start)),
          signal,
        ),
      isTransient,
      signal,
    );
    reportProgress(end);
  }

  // 3. Complete
  const completeRes = await withRetry(
    () =>
      fetch(`/api/videos/upload/${upload_id}/complete`, {
        method: "POST",
        headers: { ...authHeaders() },
        signal,
      }).then((res) => {
        if (!res.ok) {
          throw new UploadHttpError(`Complete failed: ${res.status}`, res.status);
        }
        return res;
      }),
    isSafeToRetryComplete,
    signal,
  );

  return (await completeRes.json()) as { id: string };
}
