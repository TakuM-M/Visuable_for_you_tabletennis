"""リポジトリ Protocol を満たす Fake の土台。"""

import uuid

from sqlalchemy.orm import Session

from app.models.user import User


class FakeUserRepository:
    """UserRepository Protocol の形だけを満たす土台"""

    def create(
        self, db: Session, email: str, password_hash: str, display_name: str
    ) -> User:
        raise NotImplementedError

    def get_by_id(self, db: Session, user_id: uuid.UUID) -> User | None:
        raise NotImplementedError

    def get_by_email(self, db: Session, email: str) -> User | None:
        raise NotImplementedError

    def verify_email(self, db: Session, user_id: uuid.UUID) -> None:
        raise NotImplementedError

    def update(
        self, db: Session, user_id: uuid.UUID, display_name: str, password_hash: str
    ) -> User | None:
        raise NotImplementedError
