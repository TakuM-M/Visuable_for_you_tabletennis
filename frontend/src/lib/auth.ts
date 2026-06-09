const TOKEN_KEY = "access_token";

export const getToken = (): string | null => {
  return localStorage.getItem(TOKEN_KEY);
};

export const setToken = (token: string): void => {
  localStorage.setItem(TOKEN_KEY, token);
};

export const removeToken = (): void => {
  localStorage.removeItem(TOKEN_KEY);
};

// JWT の有効期限（exp）をデコードし、トークンが存在しかつ未失効なら true を返す。
// localStorage にトークンが残っていても、期限切れなら未認証として扱う。
export const isAuthenticated = (): boolean => {
  const token = getToken();
  if (!token) return false;
  try {
    const payload = JSON.parse(atob(token.split(".")[1]));
    // exp は秒単位の UNIX 時刻。ミリ秒に変換して現在時刻と比較する。
    return typeof payload.exp === "number" && payload.exp * 1000 > Date.now();
  } catch {
    // 形式が壊れたトークンなどは未認証扱い
    return false;
  }
};

// 認証が必要な API リクエストに付与する Authorization ヘッダーを返す
export const authHeaders = (): HeadersInit => {
  const token = getToken();
  if (!token) return {};
  return { Authorization: `Bearer ${token}` };
};
