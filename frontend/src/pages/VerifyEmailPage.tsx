import { useEffect, useState } from "react";
import { useSearchParams, useNavigate } from "react-router-dom";

export default function VerifyEmailPage() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const [status, setStatus] = useState<"loading" | "success" | "error">("loading");

  useEffect(() => {
    const token = searchParams.get("token");
    if (!token) {
      setStatus("error");
      return;
    }
    fetch(`/api/auth/verify-email?token=${token}`)
      .then((res) => {
        if (res.ok) setStatus("success");
        else setStatus("error");
      })
      .catch(() => setStatus("error"));
  }, []);

  return (
    <div className="flex min-h-screen items-center justify-center">
      {status === "loading" && <p>認証中...</p>}
      {status === "success" && (
        <div className="text-center">
          <p className="text-green-600 font-bold">メール認証が完了しました！</p>
          <button onClick={() => navigate("/login")} className="mt-4 text-blue-500">
            ログインへ
          </button>
        </div>
      )}
      {status === "error" && <p className="text-red-500">認証に失敗しました。リンクが無効か期限切れです。</p>}
    </div>
  );
}
