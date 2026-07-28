import { authHeaders } from "./auth";

const CHUNK_SIZE = 50 * 1024 * 1024; // 50MB
const MAX_ATTEMPTS = 4; // 初回 + リトライ3回
const RETRY_BASE_DELAY_MS = 1000; // リトライ間隔（指数バックオフの初期値）

// アップロードサイズの上限。backend の settings.max_upload_bytes と揃えること。
// ここで先に弾くのは、上限超過が確定しているファイルのために何 GB も
// 送らせないため。最終的な担保は backend 側（413）が行う。
export const MAX_UPLOAD_BYTES = 5 * 1024 * 1024 * 1024; // 5GB

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

// そのまま画面に出してよいエラー。原因がユーザー側にあり、やり直しても
// 同じ結果になるもの（サイズ超過・長さ超過）だけをこれで包む。
export class UploadRejectedError extends Error {}

// 解析を受け付ける動画長の上限。backend の settings.max_video_duration_seconds と揃える
export const MAX_VIDEO_DURATION_SECONDS = 3600; // 60分

export function validateUploadFile(file: File): string | null {
  if (file.size > MAX_UPLOAD_BYTES) {
    const gb = MAX_UPLOAD_BYTES / 1024 ** 3;
    return `ファイルサイズが上限 ${gb}GB を超えています`;
  }
  if (file.size === 0) {
    return "ファイルが空です";
  }
  return null;
}

export function validateVideoDuration(duration: number | null): string | null {
  // 読み取れなかった場合は backend 側（結合後の ffprobe）の判定に委ねる
  if (duration === null) return null;
  if (duration > MAX_VIDEO_DURATION_SECONDS) {
    return (
      `動画の長さが上限 ${MAX_VIDEO_DURATION_SECONDS / 60}分 を超えています` +
      `（${(duration / 60).toFixed(1)}分）`
    );
  }
  return null;
}

// ブラウザにメタデータだけ読ませて再生時間を得る。長さの上限超過を
// アップロード前に知らせるためのもので、数 GB 送りきってから 413 で
// 落とされる事態を避ける。読めなければ null（判定は backend に委ねる）。
export function readVideoDuration(file: File): Promise<number | null> {
  return new Promise((resolve) => {
    const url = URL.createObjectURL(file);
    const video = document.createElement("video");
    const finish = (value: number | null) => {
      URL.revokeObjectURL(url);
      resolve(value);
    };
    video.preload = "metadata";
    video.onloadedmetadata = () =>
      finish(Number.isFinite(video.duration) ? video.duration : null);
    video.onerror = () => finish(null);
    video.src = url;
  });
}

const abortError = () => new DOMException("Aborted", "AbortError");

const isAbortError = (e: unknown): boolean =>
  e instanceof DOMException && e.name === "AbortError";

// レスポンスボディの detail（FastAPI の HTTPException）を取り出す
const parseDetail = (body: string): string | null => {
  try {
    const parsed = JSON.parse(body);
    return typeof parsed?.detail === "string" ? parsed.detail : null;
  } catch {
    return null;
  }
};

// 413 はサイズ・長さの上限超過。理由をそのまま画面に出し、リトライはしない。
const httpError = (status: number, label: string, body: string): Error =>
  status === 413
    ? new UploadRejectedError(
        parseDetail(body) ?? "アップロード可能なサイズを超えています",
      )
    : new UploadHttpError(`${label} failed: ${status}`, status);

// 一時的な失敗（ネットワークエラー・5xx・429）のみリトライ対象とする。
// モバイル回線の瞬断や画面ロックによる切断、レート制限の 503 をここで救う。
// 上限超過（UploadRejectedError）は何度送っても同じなので除外する。
const isTransient = (e: unknown): boolean =>
  !isAbortError(e) &&
  !(e instanceof UploadRejectedError) &&
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
      else reject(httpError(xhr.status, label, xhr.responseText));
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
  const invalid = validateUploadFile(file);
  if (invalid) throw new UploadRejectedError(invalid);

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
    async () => {
      const res = await fetch("/api/videos/upload/init", {
        method: "POST",
        headers: { "Content-Type": "application/json", ...authHeaders() },
        body: JSON.stringify({
          title,
          filename: file.name,
          total_chunks: totalChunks,
          // 上限判定を 1 バイト送る前に済ませてもらう
          total_bytes: file.size,
        }),
        signal,
      });
      if (!res.ok) throw httpError(res.status, "Init", await res.text());
      return res;
    },
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
    async () => {
      const res = await fetch(`/api/videos/upload/${upload_id}/complete`, {
        method: "POST",
        headers: { ...authHeaders() },
        signal,
      });
      if (!res.ok) throw httpError(res.status, "Complete", await res.text());
      return res;
    },
    isSafeToRetryComplete,
    signal,
  );

  return (await completeRes.json()) as { id: string };
}
