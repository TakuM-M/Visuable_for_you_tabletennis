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
from app.repositories.protocols import UserRepository, VideoRepository
from app.repositories.user import user_repository
from app.repositories.video import video_repository

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


# リポジトリを FastAPI の依存として供給する。サービス層のようにキーワード専用引数の
# 既定値で渡すと、FastAPI が signature を読んで「HTTP から受け取る値」と解釈し、
# ルート登録時に FastAPIError になる（* の有無は関係ない）。Depends で包むことで
# 依存として扱われ、テストからは app.dependency_overrides で差し替えられる。
def get_user_repository() -> UserRepository:
    return user_repository


def get_video_repository() -> VideoRepository:
    return video_repository


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
    user_repo: UserRepository = Depends(get_user_repository),
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
    video_repo: VideoRepository = Depends(get_video_repository),
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
