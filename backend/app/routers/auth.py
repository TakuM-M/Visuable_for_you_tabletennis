from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from app.core.security import create_access_token, verify_password
from app.db.session import get_db
from app.repositories.user import user_repository as user_repo
from app.schemas.auth import TokenResponse
from app.services import auth_service

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post(
    "/login",
    response_model=TokenResponse,
    responses={
        401: {"description": "メールアドレスまたはパスワードが違う"},
        403: {"description": "メールアドレスが未認証"},
    },
)
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
) -> TokenResponse:
    """ログイン（JWTトークンを発行する）"""
    user = user_repo.get_by_email(db, form_data.username)

    if user is None or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(
            status_code=401,
            detail="メールアドレスまたはパスワードが違います",
        )

    if not user.email_verified:
        raise HTTPException(
            status_code=403,
            detail="メールアドレスが認証されていません。メールを確認してください。",
        )

    token = create_access_token(str(user.id))
    return TokenResponse(access_token=token)


@router.get("/verify-email")
def verify_email(token: str, db: Session = Depends(get_db)):
    try:
        auth_service.verify_email_token(token, db)
        return {"message": "メール認証が完了しました"}
    except ValueError:
        raise HTTPException(status_code=400, detail="無効なトークンです")
