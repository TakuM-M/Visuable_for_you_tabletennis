import { useRef, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { useNavigate } from "react-router-dom";
import { z } from "zod";
import { chunkedUpload } from "../lib/chunkedUpload";

const schema = z.object({
  title: z.string().min(1, "タイトルを入力してください"),
  file: z.instanceof(FileList).refine(
    (files) => files.length > 0,
    "動画ファイルを選択してください"
  ),
});

type FormValues = z.infer<typeof schema>;

export default function VideoUploadPage() {
  const navigate = useNavigate();
  const [progress, setProgress] = useState<number | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const { register, handleSubmit, formState: { errors } } = useForm<FormValues>({
    resolver: zodResolver(schema),
  });

  const mutation = useMutation({
    mutationFn: (values: FormValues) => {
      const controller = new AbortController();
      abortRef.current = controller;
      setProgress(0);
      return chunkedUpload({
        file: values.file[0],
        title: values.title,
        onProgress: setProgress,
        signal: controller.signal,
      });
    },
    onSuccess: () => {
      navigate("/");
    },
    onError: () => {
      setProgress(null);
    },
  });

  const handleCancel = () => {
    abortRef.current?.abort();
    mutation.reset();
    setProgress(null);
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-gray-50">
      <div className="w-full max-w-sm rounded-lg bg-white p-8 shadow">
        <h1 className="mb-6 text-2xl font-bold text-gray-800">動画アップロード</h1>

        <form onSubmit={handleSubmit((data) => mutation.mutate(data))} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700">
              タイトル
            </label>
            <input
              {...register("title")}
              type="text"
              className="mt-1 w-full rounded border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            {errors.title && (
              <p className="mt-1 text-sm text-red-500">{errors.title.message}</p>
            )}
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700">
              動画ファイル
            </label>
            <input
              {...register("file")}
              type="file"
              accept="video/*"
              className="mt-1 w-full text-sm"
            />
            {errors.file && (
              <p className="mt-1 text-sm text-red-500">{errors.file.message}</p>
            )}
          </div>

          {progress !== null && (
            <div>
              <div className="mb-1 flex justify-between text-sm text-gray-600">
                <span>アップロード中...</span>
                <span>{progress}%</span>
              </div>
              <div className="h-2 w-full rounded-full bg-gray-200">
                <div
                  className="h-2 rounded-full bg-blue-600 transition-all"
                  style={{ width: `${progress}%` }}
                />
              </div>
            </div>
          )}

          {mutation.isError && (
            <p className="text-sm text-red-500">
              アップロードに失敗しました。もう一度お試しください。
            </p>
          )}

          <button
            type="submit"
            disabled={mutation.isPending}
            className="w-full rounded bg-blue-600 px-4 py-2 text-white hover:bg-blue-700 disabled:opacity-50"
          >
            {mutation.isPending ? "アップロード中..." : "アップロード"}
          </button>

          {mutation.isPending && (
            <button
              type="button"
              onClick={handleCancel}
              className="w-full rounded border border-gray-300 px-4 py-2 text-sm text-gray-600 hover:bg-gray-50"
            >
              キャンセル
            </button>
          )}
        </form>

        <button
          onClick={() => navigate("/")}
          className="mt-4 w-full text-center text-sm text-gray-500 hover:underline"
        >
          ← 一覧に戻る
        </button>
      </div>
    </div>
  );
}
