from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.deps import get_current_user
from app.core.security import hash_password
from app.db.session import get_db
from app.models.user import User
from app.repositories import user as user_repo
from app.schemas.user import UserCreate, UserUpdate, UserResponse
from app.services import auth_service

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
    
    auth_service.send_verification_email(user) 
    return user


@router.get("/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_user)) -> UserResponse:
    """ログイン中のユーザー情報を取得"""
    return current_user

@router.patch("/me", response_model=UserResponse)
def update(body: UserUpdate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)) -> UserResponse:
    """ユーザー情報を更新"""
    if body.password:
        password_hash = hash_password(body.password)
    else:
        password_hash = current_user.password_hash

    user_repo.update(
        db=db,
        user_id=current_user.id,
        display_name=body.display_name or current_user.display_name,
        password_hash=password_hash,
    )
    
    updated_user = user_repo.get_by_id(db, current_user.id)
    return updated_user