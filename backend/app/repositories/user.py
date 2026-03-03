import uuid

from sqlalchemy.orm import Session

from app.models.user import User


def create(
    db: Session,
    email: str,
    password_hash: str,
    display_name: str,
) -> User:
    user = User(
        email=email,
        password_hash=password_hash,
        display_name=display_name,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def get_by_id(db: Session, user_id: uuid.UUID) -> User | None:
    return db.query(User).filter(User.id == user_id).first()


def get_by_email(db: Session, email: str) -> User | None:
    return db.query(User).filter(User.email == email).first()

def verify_email(db: Session, user_id: uuid.UUID) -> None:
    user = get_by_id(db, user_id)
    if user:
        user.email_verified = True
        db.commit()
