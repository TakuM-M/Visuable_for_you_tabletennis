from pydantic import BaseModel


class LoginRequest(BaseModel):
    """ログインリクエスト"""

    email: str
    password: str


class TokenResponse(BaseModel):
    """JWTトークンレスポンス"""

    access_token: str
    token_type: str = "bearer"
