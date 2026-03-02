import hashlib

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.repositories import user as user_repo
from app.schemas.user import UserCreate, UserResponse

router = APIRouter(prefix="/users", tags=["users"])


def _hash_password(password: str) -> str:
    # TODO: 認証実装時に bcrypt 等の適切なハッシュに置き換える
    return hashlib.sha256(password.encode()).hexdigest()


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
        password_hash=_hash_password(body.password),
        display_name=body.display_name,
    )
    return user
