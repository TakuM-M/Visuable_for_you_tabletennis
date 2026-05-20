import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { useNavigate } from "react-router-dom";
import { z } from "zod";
import { getMeUsersMeGet, updateUsersMePatch } from "../api/generated";
import { authHeaders, removeToken } from "../lib/auth";
import AppShell from "../components/layout/AppShell";
import Button from "../components/ui/Button";
import { IconLogout } from "../components/ui/Icons";

const schema = z.object({
  display_name: z.string().min(1, "表示名を入力してください"),
  password: z
    .string()
    .min(8, "パスワードは8文字以上で入力してください")
    .or(z.literal("")),
});
type FormValues = z.infer<typeof schema>;

export default function ProfilePage() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [okMsg, setOkMsg] = useState<string | null>(null);
  const [errMsg, setErrMsg] = useState<string | null>(null);

  const { data: meRes, isLoading } = useQuery({
    queryKey: ["me"],
    queryFn: () => getMeUsersMeGet({ headers: authHeaders() }),
  });
  const me = meRes?.status === 200 ? meRes.data : null;

  const {
    register,
    handleSubmit,
    reset,
    formState: { errors, isDirty },
  } = useForm<FormValues>({
    resolver: zodResolver(schema),
    defaultValues: { display_name: "", password: "" },
  });

  useEffect(() => {
    if (me) reset({ display_name: me.display_name, password: "" });
  }, [me, reset]);

  const mutation = useMutation({
    mutationFn: (values: FormValues) =>
      updateUsersMePatch(
        {
          display_name: values.display_name,
          password: values.password || undefined,
        },
        { headers: authHeaders() },
      ),
    onSuccess: (res) => {
      if (res.status === 200) {
        setOkMsg("プロフィールを更新しました");
        setErrMsg(null);
        queryClient.invalidateQueries({ queryKey: ["me"] });
      } else {
        setErrMsg("更新に失敗しました");
      }
    },
    onError: () => setErrMsg("エラーが発生しました。もう一度お試しください。"),
  });

  const onSubmit = (values: FormValues) => {
    setOkMsg(null);
    setErrMsg(null);
    mutation.mutate(values);
  };

  const onLogout = () => {
    removeToken();
    navigate("/login");
  };

  if (isLoading) {
    return (
      <AppShell>
        <div className="grid h-full place-items-center text-fg-3">読み込み中...</div>
      </AppShell>
    );
  }

  const initials = me?.display_name?.slice(0, 2).toUpperCase() ?? "—";

  return (
    <AppShell>
      <div className="scroll-thin h-full overflow-auto">
        <div className="mx-auto max-w-[720px] px-8 pt-8 pb-16">
          <div className="mb-6">
            <div className="mb-1.5 font-mono text-[11px] uppercase tracking-[0.1em] text-fg-4">
              Account
            </div>
            <h1 className="m-0 text-[22px] font-semibold tracking-[-0.015em]">プロフィール</h1>
          </div>

          {/* Identity card */}
          <div className="mb-6 flex items-center gap-4 rounded-xl border border-border bg-surface p-5">
            <div className="grid h-14 w-14 flex-none place-items-center rounded-xl bg-accent-soft text-[20px] font-semibold tracking-[-0.02em] text-accent-ink">
              {initials}
            </div>
            <div className="min-w-0 flex-1">
              <div className="text-[15px] font-semibold">{me?.display_name}</div>
              <div className="mt-0.5 font-mono text-[11.5px] text-fg-3">{me?.email}</div>
            </div>
            <span className="rounded-full bg-subtle-2 px-2 py-0.5 font-mono text-[11px] text-fg-2">
              FREE
            </span>
          </div>

          <form onSubmit={handleSubmit(onSubmit)}>
            <SettingsSection title="表示設定">
              <Row label="表示名" hint="一覧などに表示されます">
                <input
                  {...register("display_name")}
                  type="text"
                  className="w-full rounded-lg border border-border bg-surface px-3 py-2 text-[13.5px] outline-none focus:border-accent"
                />
                {errors.display_name && (
                  <p className="mt-1.5 text-[11.5px] text-err">{errors.display_name.message}</p>
                )}
              </Row>
              <Row label="メールアドレス" hint="現在ログイン中">
                <input
                  readOnly
                  value={me?.email ?? ""}
                  className="w-full rounded-lg border border-border bg-subtle px-3 py-2 text-[13.5px] text-fg-3 outline-none"
                />
              </Row>
            </SettingsSection>

            <SettingsSection title="セキュリティ">
              <Row label="新しいパスワード" hint="変更しない場合は空欄">
                <input
                  {...register("password")}
                  type="password"
                  placeholder="••••••••"
                  className="w-full rounded-lg border border-border bg-surface px-3 py-2 text-[13.5px] outline-none focus:border-accent"
                />
                {errors.password && (
                  <p className="mt-1.5 text-[11.5px] text-err">{errors.password.message}</p>
                )}
              </Row>
            </SettingsSection>

            {okMsg && <p className="text-[12.5px] text-ok-ink">{okMsg}</p>}
            {errMsg && <p className="text-[12.5px] text-err">{errMsg}</p>}

            <div className="mt-6 flex items-center justify-between border-t border-dashed border-border pt-5">
              <Button type="button" kind="danger" size="sm" onClick={onLogout}>
                <IconLogout size={13} />
                ログアウト
              </Button>
              <div className="flex gap-2">
                <Button
                  type="button"
                  kind="secondary"
                  size="sm"
                  onClick={() => me && reset({ display_name: me.display_name, password: "" })}
                  disabled={!isDirty || mutation.isPending}
                >
                  キャンセル
                </Button>
                <Button
                  type="submit"
                  kind="primary"
                  size="sm"
                  disabled={!isDirty || mutation.isPending}
                >
                  {mutation.isPending ? "更新中..." : "変更を保存"}
                </Button>
              </div>
            </div>
          </form>
        </div>
      </div>
    </AppShell>
  );
}

function SettingsSection({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  const items = Array.isArray(children) ? children : [children];
  return (
    <div className="mb-6">
      <div className="mb-3 font-mono text-[10.5px] uppercase tracking-[0.1em] text-fg-4">
        {title}
      </div>
      <div className="overflow-hidden rounded-xl border border-border bg-surface">
        {items.map((c, i) => (
          <div key={i}>
            {i > 0 && <div className="h-px bg-border" />}
            {c}
          </div>
        ))}
      </div>
    </div>
  );
}

function Row({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="grid grid-cols-1 items-start gap-2 px-5 py-4 md:grid-cols-[200px_1fr] md:gap-6">
      <div>
        <div className="text-[13px] font-medium">{label}</div>
        {hint && <div className="mt-0.5 text-[11.5px] leading-[1.5] text-fg-3">{hint}</div>}
      </div>
      <div>{children}</div>
    </div>
  );
}
