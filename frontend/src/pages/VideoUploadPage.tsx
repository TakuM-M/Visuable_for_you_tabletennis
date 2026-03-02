import { useMutation } from "@tanstack/react-query";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { uploadVideoVideosPost } from "../api/generated";
import { authHeaders } from "../lib/auth";
import { useNavigate } from "react-router-dom";
import { z } from "zod";

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

  const { register, handleSubmit, formState: { errors } } = useForm<FormValues>({
    resolver: zodResolver(schema),
  });

  const mutation = useMutation({
    mutationFn: (values: FormValues) =>
      uploadVideoVideosPost(
        { title: values.title, file: values.file[0] },
        { headers: authHeaders() }
      ),
    onSuccess: (res) => {
      if (res.status === 201) {
        navigate("/");
      }
    },
  });

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

          <button
            type="submit"
            disabled={mutation.isPending}
            className="w-full rounded bg-blue-600 px-4 py-2 text-white hover:bg-blue-700 disabled:opacity-50"
          >
            {mutation.isPending ? "アップロード中..." : "アップロード"}
          </button>
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
