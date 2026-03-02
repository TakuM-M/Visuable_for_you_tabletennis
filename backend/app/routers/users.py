from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.deps import get_current_user
from app.core.security import hash_password
from app.db.session import get_db
from app.models.user import User
from app.repositories import user as user_repo
from app.schemas.user import UserCreate, UserResponse

router = APIRouter(prefix="/users", tags=["users"])


@router.post("", response_model=UserResponse, status_code=201)
def register(body: UserCreate, db: Session = Depends(get_db)) -> UserResponse:
    """ユーザー登録"""
    if user_repo.get_by_email(db, body.email):
        raise HTTPException(
            status_code=400,
            detail="このメールアドレスは既に使用されています",
        )

    user = user_repo.create(
        db=db,
        email=body.email,
        password_hash=hash_password(body.password),
        display_name=body.display_name,
    )
    return user


@router.get("/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_user)) -> UserResponse:
    """ログイン中のユーザー情報を取得"""
    return current_user
