from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from app.core.security import create_access_token, verify_password
from app.db.session import get_db
from app.repositories import user as user_repo
from app.schemas.auth import TokenResponse

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=TokenResponse)
def login(
    # OAuth2PasswordRequestForm はフォームデータで username と password を受け取る
    # Swagger UI の Authorize ボタンはこの形式で送信する
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
) -> TokenResponse:
    """ログイン（JWTトークンを発行する）"""
    # OAuth2 の仕様では email を username フィールドで受け取る
    user = user_repo.get_by_email(db, form_data.username)

    if user is None or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(
            status_code=401,
            detail="メールアドレスまたはパスワードが違います",
        )

    token = create_access_token(str(user.id))
    return TokenResponse(access_token=token)
