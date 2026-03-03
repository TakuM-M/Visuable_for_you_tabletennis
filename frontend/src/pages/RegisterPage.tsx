import { zodResolver } from "@hookform/resolvers/zod";
import { useMutation } from "@tanstack/react-query";
import { useState } from "react";
import { useForm } from "react-hook-form";
import { useNavigate } from "react-router-dom";
import { z } from "zod";
import { registerUsersPost } from "../api/generated";

// バリデーションルール
const schema = z.object({
  display_name: z.string().min(1, "表示名を入力してください"),
  email: z.email("正しいメールアドレスを入力してください"),
  password: z.string().min(8, "パスワードは8文字以上で入力してください"),
});

type FormValues = z.infer<typeof schema>;

export default function RegisterPage() {
  const navigate = useNavigate();
  const [registerError, setRegisterError] = useState<string | null>(null);

  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<FormValues>({ resolver: zodResolver(schema) });

  const mutation = useMutation({
    mutationFn: (values: FormValues) => registerUsersPost(values),
    onSuccess: (res) => {
      if (res.status === 201) {
        // 登録成功 → ログインページへ
        navigate("/login", { state: { message: "確認メールを送信しました。メールを確認してからログインしてください。" } });
      } else {
        // 400: メールアドレスが既に使用されている
        setRegisterError("このメールアドレスは既に使用されています");
      }
    },
  });

  const onSubmit = (values: FormValues) => {
    setRegisterError(null);
    mutation.mutate(values);
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-gray-50">
      <div className="w-full max-w-sm rounded-lg bg-white p-8 shadow">
        <h1 className="mb-6 text-2xl font-bold text-gray-800">新規登録</h1>

        <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700">
              表示名
            </label>
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

          {(registerError || mutation.isError) && (
            <p className="text-xs text-red-500">
              {registerError ?? "エラーが発生しました。もう一度お試しください。"}
            </p>
          )}

          <button
            type="submit"
            disabled={mutation.isPending}
            className="w-full rounded bg-blue-600 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50"
          >
            {mutation.isPending ? "登録中..." : "登録する"}
          </button>
        </form>

        <p className="mt-4 text-center text-sm text-gray-500">
          既にアカウントをお持ちの方は{" "}
          <a href="/login" className="text-blue-600 hover:underline">
            ログイン
          </a>
        </p>
      </div>
    </div>
  );
}
