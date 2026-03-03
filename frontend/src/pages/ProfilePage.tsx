import { zodResolver } from "@hookform/resolvers/zod";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { useForm } from "react-hook-form";
import { useNavigate } from "react-router-dom";
import { z } from "zod";
import { getMeUsersMeGet, updateUsersMePatch } from "../api/generated";
import { authHeaders } from "../lib/auth";

// バリデーションルール
const schema = z.object({
  display_name: z.string().min(1, "表示名を入力してください"),
  password: z.string().min(8, "パスワードは8文字以上で入力してください").or(z.literal("")),
});

type FormValues = z.infer<typeof schema>;

export default function ProfilePage() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // 現在のユーザー情報を取得
  const { data: meRes, isLoading } = useQuery({
    queryKey: ["me"],
    queryFn: () => getMeUsersMeGet({ headers: authHeaders() }),
  });

  const {
    register,
    handleSubmit,
    reset,
    formState: { errors },
  } = useForm<FormValues>({
    resolver: zodResolver(schema),
    defaultValues: { display_name: "", password: "" },
  });

  // ユーザー情報取得後にフォームの初期値をセット
  useEffect(() => {
    if (meRes?.data) {
      reset({ display_name: meRes.data.display_name, password: "" });
    }
  }, [meRes, reset]);

  const mutation = useMutation({
    mutationFn: (values: FormValues) =>
      updateUsersMePatch(
        { display_name: values.display_name, password: values.password || undefined },
        { headers: authHeaders() }
      ),
    onSuccess: (res) => {
      if (res.status === 200) {
        setSuccessMessage("プロフィールを更新しました");
        setErrorMessage(null);
        queryClient.invalidateQueries({ queryKey: ["me"] });
      } else {
        setErrorMessage("更新に失敗しました");
      }
    },
    onError: () => {
      setErrorMessage("エラーが発生しました。もう一度お試しください。");
    },
  });

  const onSubmit = (values: FormValues) => {
    setSuccessMessage(null);
    setErrorMessage(null);
    mutation.mutate(values);
  };

  if (isLoading) {
    return <div className="flex min-h-screen items-center justify-center">読み込み中...</div>;
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-gray-50">
      <div className="w-full max-w-sm rounded-lg bg-white p-8 shadow">
        <h1 className="mb-6 text-2xl font-bold text-gray-800">プロフィール編集</h1>

        <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700">表示名</label>
            <input
              {...register("display_name")}
              type="text"
              className="mt-1 w-full rounded border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            {errors.display_name && (
              <p className="mt-1 text-xs text-red-500">{errors.display_name.message}</p>
            )}
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700">
              新しいパスワード（変更しない場合は空欄）
            </label>
            <input
              {...register("password")}
              type="password"
              className="mt-1 w-full rounded border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            {errors.password && (
              <p className="mt-1 text-xs text-red-500">{errors.password.message}</p>
            )}
          </div>

          {successMessage && (
            <p className="text-xs text-green-600">{successMessage}</p>
          )}
          {errorMessage && (
            <p className="text-xs text-red-500">{errorMessage}</p>
          )}

          <button
            type="submit"
            disabled={mutation.isPending}
            className="w-full rounded bg-blue-600 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50"
          >
            {mutation.isPending ? "更新中..." : "更新する"}
          </button>
        </form>

        <button
          onClick={() => navigate("/")}
          className="mt-4 w-full rounded border border-gray-300 py-2 text-sm text-gray-600 hover:bg-gray-50"
        >
          戻る
        </button>
      </div>
    </div>
  );
}
