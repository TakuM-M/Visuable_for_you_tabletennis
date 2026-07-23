import { zodResolver } from "@hookform/resolvers/zod";
import { useMutation } from "@tanstack/react-query";
import { useState } from "react";
import { useForm } from "react-hook-form";
import { Navigate, useNavigate, useLocation } from "react-router-dom";
import { z } from "zod";
import { loginAuthLoginPost } from "../api/generated";
import { isAuthenticated, setToken } from "../lib/auth";

// バリデーションルール（Zod v4 の書き方）
const schema = z.object({
  email: z.email("正しいメールアドレスを入力してください"),
  password: z.string().min(1, "パスワードを入力してください"),
});

type FormValues = z.infer<typeof schema>;

export default function LoginPage() {
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const navigate = useNavigate();
  const location = useLocation();
  const [successMessage, setSuccessMessage] = useState<string | null>(location.state?.message ?? null);

  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<FormValues>({ resolver: zodResolver(schema) });

  const mutation = useMutation({
    mutationFn: (values: FormValues) =>
      loginAuthLoginPost({ username: values.email, password: values.password }),
        onSuccess: (res:any) => {
          if (res.status === 200) {
            setToken(res.data.access_token);
            navigate("/videos");
          } else if (res.status === 403) {
            setErrorMessage("メール認証が完了していません。届いたメールを確認してください。");
          } else {
            setErrorMessage("メールアドレスまたはパスワードが違います");
          }
        },
        onError: () => {
          setErrorMessage("エラーが発生しました。もう一度お試しください。");
        },
  });

  const onSubmit = (values: FormValues) => {
    setErrorMessage(null);
    setSuccessMessage(null);
    mutation.mutate(values);
  };

  if (isAuthenticated()) return <Navigate to="/videos" replace />;

  return (
    <div className="flex min-h-screen items-center justify-center bg-gray-50">
      <div className="w-full max-w-sm rounded-lg bg-white p-8 shadow">
        <h1 className="mb-6 text-2xl font-bold text-gray-800">ログイン</h1>

        <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700">
              メールアドレス
            </label>
            <input
              {...register("email")}
              type="email"
              className="mt-1 w-full rounded border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            {errors.email && (
              <p className="mt-1 text-xs text-red-500">{errors.email.message}</p>
            )}
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700">
              パスワード
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

          {errorMessage && (
            <p className="text-xs text-red-500">
              {errorMessage}
            </p>
          )}
          {successMessage && (
            <p className="text-xs text-green-600">{successMessage}</p>
          )}
          <button
            type="submit"
            disabled={mutation.isPending}
            className="w-full rounded bg-blue-600 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50"
          >
            {mutation.isPending ? "ログイン中..." : "ログイン"}
          </button>
        </form>

        <p className="mt-4 text-center text-sm text-gray-500">
          アカウントがない方は{" "}
          <a href="/register" className="text-blue-600 hover:underline">
            新規登録
          </a>
        </p>
      </div>
    </div>
  );
}
