import { useRef, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { useNavigate } from "react-router-dom";
import { z } from "zod";
import {
  UploadRejectedError,
  chunkedUpload,
  readVideoDuration,
  validateUploadFile,
  validateVideoDuration,
} from "../lib/chunkedUpload";
import AppShell from "../components/layout/AppShell";
import Button from "../components/ui/Button";
import Stripes from "../components/ui/Stripes";
import { IconChevL, IconClose, IconUpload } from "../components/ui/Icons";

const schema = z.object({
  title: z.string().min(1, "タイトルを入力してください"),
  file: z
    .custom<File>((v) => v instanceof File, "動画ファイルを選択してください")
    // 上限超過は送る前に弾く。数 GB 送りきってから 413 で落ちるのが一番つらい
    .superRefine((f: File, ctx) => {
      const message = validateUploadFile(f);
      if (message) ctx.addIssue({ code: "custom", message });
    }),
});

type FormValues = z.infer<typeof schema>;

const ACCEPT = "video/mp4,video/quicktime,video/x-matroska";

export default function VideoUploadPage() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [progress, setProgress] = useState<number | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [duration, setDuration] = useState<number | null>(null);
  const [durationError, setDurationError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const {
    register,
    handleSubmit,
    setValue,
    resetField,
    watch,
    formState: { errors },
  } = useForm<FormValues>({ resolver: zodResolver(schema) });

  // file は input に ref を渡さず setValue のみで管理する。RHF の setValue は
  // 対象 ref が file input だと入力をクリアするため、ref を結び付けると選択値が消える。
  register("file");
  const titleValue = watch("title");

  const mutation = useMutation({
    mutationFn: (values: FormValues) => {
      const controller = new AbortController();
      abortRef.current = controller;
      setProgress(0);
      return chunkedUpload({
        file: values.file,
        title: values.title,
        onProgress: setProgress,
        signal: controller.signal,
      });
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ["videos"] });
      navigate(`/videos/${data.id}`);
    },
    onError: () => setProgress(null),
  });

  const handleCancel = () => {
    abortRef.current?.abort();
    mutation.reset();
    setProgress(null);
  };

  const onFileChange = async (files: FileList | null) => {
    if (!files?.length) return;
    const file = files[0];
    setSelectedFile(file);
    setDuration(null);
    setDurationError(null);
    // input.files の FileList は live オブジェクトで後から空になり得るため、File を切り出して保持する
    setValue("file", file, { shouldValidate: true });
    if (!titleValue) {
      const stem = file.name.replace(/\.[^.]+$/, "");
      setValue("title", stem);
    }
    // 長さの判定は非同期（メタデータ読み込み待ち）なので zod のスキーマには
    // 載せられない。選択直後に別途チェックして送信前に知らせる
    const seconds = await readVideoDuration(file);
    setDuration(seconds);
    setDurationError(validateVideoDuration(seconds));
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    onFileChange(e.dataTransfer.files);
  };

  const onClear = () => {
    setSelectedFile(null);
    setDuration(null);
    setDurationError(null);
    resetField("file");
  };

  const isUploading = mutation.isPending;

  return (
    <AppShell>
      <div className="scroll-thin h-full overflow-auto">
        <div className="mx-auto max-w-[760px] px-8 pt-8 pb-16">
          <div className="mb-6">
            <div className="mb-1.5 font-mono text-[11px] uppercase tracking-[0.1em] text-fg-4">
              New upload
            </div>
            <h1 className="m-0 text-[22px] font-semibold tracking-[-0.015em]">
              動画をアップロード
            </h1>
            <p className="mt-1.5 text-[13px] leading-[1.6] text-fg-3">
              アップロード後、自動でプレーシーンを抽出します。
            </p>
          </div>

          <form
            onSubmit={handleSubmit((d) => mutation.mutate(d))}
            className="flex flex-col gap-5"
          >
            {/* Title */}
            <Field
              label="タイトル"
              hint="例: 練習試合_2025-05-12 vs 田中"
              error={errors.title?.message}
            >
              <input
                {...register("title")}
                type="text"
                placeholder="動画タイトルを入力"
                className="w-full rounded-lg border border-border bg-surface px-3 py-2 text-[13.5px] outline-none focus:border-accent"
                disabled={isUploading}
              />
            </Field>

            {/* File */}
            <Field
              label="動画ファイル"
              hint="MP4 / MOV / MKV"
              error={(errors.file?.message as string | undefined) ?? durationError ?? undefined}
            >
              {selectedFile ? (
                <div className="flex items-center gap-3 rounded-[10px] border border-border bg-surface p-3">
                  <div className="h-10 w-16 flex-none overflow-hidden rounded-md">
                    <Stripes />
                  </div>
                  <div className="min-w-0 flex-1">
                    <div className="truncate text-[13px] font-medium">
                      {selectedFile.name}
                    </div>
                    <div className="mt-0.5 font-mono text-[11px] text-fg-4">
                      {formatBytes(selectedFile.size)}
                      {duration !== null && ` · ${formatDuration(duration)}`}
                    </div>
                  </div>
                  {!isUploading && (
                    <button
                      type="button"
                      onClick={onClear}
                      className="bg-transparent p-1.5 text-fg-3 hover:text-fg"
                      aria-label="ファイルをクリア"
                    >
                      <IconClose size={14} />
                    </button>
                  )}
                </div>
              ) : (
                <label
                  onDragOver={(e) => {
                    e.preventDefault();
                    setIsDragging(true);
                  }}
                  onDragLeave={() => setIsDragging(false)}
                  onDrop={onDrop}
                  className={`
                    block cursor-pointer rounded-[10px] border-[1.5px] border-dashed px-5 py-8 text-center transition-colors
                    ${isDragging
                      ? "border-accent bg-accent-soft"
                      : "border-border-strong bg-subtle hover:bg-subtle-2"}
                  `}
                >
                  <input
                    type="file"
                    accept={ACCEPT}
                    className="hidden"
                    onChange={(e) => onFileChange(e.target.files)}
                  />
                  <div className="mx-auto grid h-9 w-9 place-items-center rounded-lg border border-border bg-surface text-fg-2">
                    <IconUpload size={16} />
                  </div>
                  <div className="mt-3 text-[13.5px] font-medium">
                    ファイルをドロップ <span className="font-normal text-fg-3">または</span>{" "}
                    <span className="text-accent">選択</span>
                  </div>
                  <div className="mt-1.5 font-mono text-[10.5px] text-fg-4">
                    MP4 / MOV / MKV · 最大 5GB / 60分
                  </div>
                </label>
              )}
            </Field>

            {/* Progress */}
            {progress !== null && (
              <div className="rounded-[10px] border border-accent-ink/30 bg-accent-soft p-3.5">
                <div className="mb-2 flex justify-between">
                  <span className="text-[12.5px] font-medium text-accent-ink">
                    アップロード中...
                  </span>
                  <span className="font-mono text-[11.5px] text-accent-ink">
                    {progress}%
                  </span>
                </div>
                <div className="h-1 overflow-hidden rounded-full bg-white/60">
                  <div
                    className="h-full rounded-full bg-accent transition-[width] duration-300"
                    style={{ width: `${progress}%` }}
                  />
                </div>
              </div>
            )}

            {mutation.isError && (
              <p className="text-[12.5px] text-err">
                {mutation.error instanceof UploadRejectedError
                  ? mutation.error.message
                  : "アップロードに失敗しました。もう一度お試しください。"}
              </p>
            )}

            {/* Actions */}
            <div className="mt-1 flex items-center justify-between">
              <Button
                type="button"
                kind="ghost"
                size="sm"
                onClick={() => navigate("/videos")}
                disabled={isUploading}
              >
                <IconChevL size={13} /> 一覧に戻る
              </Button>
              <div className="flex gap-2">
                {isUploading && (
                  <Button type="button" kind="secondary" size="sm" onClick={handleCancel}>
                    キャンセル
                  </Button>
                )}
                <Button
                  type="submit"
                  kind="primary"
                  size="sm"
                  disabled={isUploading || durationError !== null}
                >
                  {isUploading ? "アップロード中..." : "アップロードを開始"}
                </Button>
              </div>
            </div>
          </form>
        </div>
      </div>
    </AppShell>
  );
}

function Field({
  label,
  hint,
  error,
  children,
}: {
  label: string;
  hint?: string;
  error?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-baseline justify-between">
        <label className="text-[12.5px] font-medium">{label}</label>
        {hint && (
          <span className="font-mono text-[10.5px] text-fg-4">{hint}</span>
        )}
      </div>
      {children}
      {error && <p className="text-[11.5px] text-err">{error}</p>}
    </div>
  );
}

function formatDuration(seconds: number) {
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return `${m}分${String(s).padStart(2, "0")}秒`;
}

function formatBytes(b: number) {
  if (b < 1024) return `${b} B`;
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`;
  if (b < 1024 * 1024 * 1024) return `${(b / 1024 / 1024).toFixed(1)} MB`;
  return `${(b / 1024 / 1024 / 1024).toFixed(2)} GB`;
}
