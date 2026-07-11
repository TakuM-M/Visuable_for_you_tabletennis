import {
  QueryClient,
  QueryClientProvider,
  QueryCache,
  MutationCache,
} from "@tanstack/react-query";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { isAuthenticated, removeToken } from "./lib/auth";
import LoginPage from "./pages/LoginPage";
import RegisterPage from "./pages/RegisterPage";
import LandingPage from "./pages/LandingPage";
import VideoListPage from "./pages/VideoListPage";
import VideoDetailPage from "./pages/VideoDetailPage";
import VideoUploadPage from "./pages/VideoUploadPage";
import VerifyEmailPage from "./pages/VerifyEmailPage";
import ProfilePage from "./pages/ProfilePage";


// API レスポンスが 401（認証切れ）なら自動でログアウトしてログイン画面へ送る。
// Orval 生成の fetch クライアントは 401 でも例外を投げず { status: 401 } を resolve するため、
// onError ではなく onSuccess 側でステータスを判定する。
const handleApiResult = (data: unknown) => {
  if (
    data &&
    typeof data === "object" &&
    (data as { status?: number }).status === 401
  ) {
    removeToken();
    // QueryClient は Router の外で生成されるため navigate が使えない。location で遷移する。
    if (window.location.pathname !== "/login") {
      window.location.replace("/login");
    }
  }
};

const queryClient = new QueryClient({
  queryCache: new QueryCache({ onSuccess: handleApiResult }),
  mutationCache: new MutationCache({ onSuccess: handleApiResult }),
});

// 未認証（トークンが無い／期限切れ）の場合は /login にリダイレクトするラッパー
function PrivateRoute({ children }: { children: React.ReactNode }) {
  if (!isAuthenticated()) {
    removeToken();
    return (
      <Navigate
        to="/login"
        replace
        state={{ message: "セッションの有効期限が切れました。再度ログインしてください。" }}
      />
    );
  }
  return <>{children}</>;
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route path="/login" element={<LoginPage />} />
          <Route path="/register" element={<RegisterPage />} />
          <Route path="/" element={<LandingPage />} />
          <Route path="/verify-email" element={<VerifyEmailPage />} />
          <Route path="/videos" element={<PrivateRoute><VideoListPage /></PrivateRoute>} />
          <Route path="/videos/new" element={<PrivateRoute><VideoUploadPage /></PrivateRoute>} />
          <Route path="/videos/:id" element={<PrivateRoute><VideoDetailPage /></PrivateRoute>} />
          <Route path="/profile" element={<PrivateRoute><ProfilePage /></PrivateRoute>} />
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}

export default App;
