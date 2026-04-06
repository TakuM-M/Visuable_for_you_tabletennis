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

// 認証が必要な API リクエストに付与する Authorization ヘッダーを返す
export const authHeaders = (): HeadersInit => {
  const token = getToken();
  if (!token) return {};
  return { Authorization: `Bearer ${token}` };
};
