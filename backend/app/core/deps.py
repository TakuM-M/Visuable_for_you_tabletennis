import uuid

from fastapi import Depends, Header, HTTPException
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import decode_token
from app.db.session import get_db
from app.models.user import User
from app.models.video import Video
from app.repositories.user import user_repository as user_repo
from app.repositories import video as video_repo

# Authorization: Bearer <token> ヘッダーからトークンを自動取得する FastAPI の既製品
# tokenUrl はSwagger UIの「Authorize」ボタンが使うログインエンドポイント
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    """JWTトークンを検証してログイン中のユーザーを返す"""
    try:
        user_id = uuid.UUID(decode_token(token))
    except (JWTError, ValueError):
        raise HTTPException(
            status_code=401,
            detail="トークンが無効または期限切れです",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = user_repo.get_by_id(db, user_id)
    if user is None:
        raise HTTPException(status_code=401, detail="ユーザーが見つかりません")

    return user


def get_owned_video(
    video_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Video:
    """video_id の動画を取得し、ログインユーザーの所有物か検証して返す。

    存在しなければ 404、他人の動画なら 403 を返す。video_id をパスに持つ
    動画関連エンドポイント（詳細・削除・出力・jobs/clips 一覧）の所有者チェックを
    一元化するための共通依存。
    """
    video = video_repo.get_by_id(db, video_id)
    if video is None:
        raise HTTPException(status_code=404, detail="動画が見つかりません")
    if video.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="この動画への権限がありません")
    return video


def require_internal_api_key(
    x_internal_api_key: str | None = Header(default=None),
) -> None:
    """X-Internal-Api-Key ヘッダで /admin/* と内部エンドポイントを保護する。

    internal_api_key が未設定の環境では常に 401 を返し、誤公開を防ぐ。
    """
    if not settings.internal_api_key or x_internal_api_key != settings.internal_api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")
