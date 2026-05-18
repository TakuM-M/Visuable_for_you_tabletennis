from fastapi import Depends, Header, HTTPException
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import decode_token
from app.db.session import get_db
from app.models.user import User
from app.repositories import user as user_repo

# Authorization: Bearer <token> ヘッダーからトークンを自動取得する FastAPI の既製品
# tokenUrl はSwagger UIの「Authorize」ボタンが使うログインエンドポイント
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    """JWTトークンを検証してログイン中のユーザーを返す"""
    try:
        user_id = decode_token(token)
    except JWTError:
        raise HTTPException(
            status_code=401,
            detail="トークンが無効または期限切れです",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = user_repo.get_by_id(db, user_id)
    if user is None:
        raise HTTPException(status_code=401, detail="ユーザーが見つかりません")

    return user


def require_internal_api_key(
    x_internal_api_key: str | None = Header(default=None),
) -> None:
    """X-Internal-Api-Key ヘッダで /admin/* と内部エンドポイントを保護する。

    internal_api_key が未設定の環境では常に 401 を返し、誤公開を防ぐ。
    """
    if not settings.internal_api_key or x_internal_api_key != settings.internal_api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")
