import uuid
from typing import Protocol

from sqlalchemy.orm import Session

from app.models.user import User


class UserRepository(Protocol):
    def create(self, db: Session, email: str, password_hash: str, display_name: str) -> User:
        ...
        
    def get_by_id(self, db: Session, user_id: uuid.UUID) -> User | None:
        ...
        
    def get_by_email(self, db: Session, email: str) -> User | None:
        ...
    
    def verify_email(self, db: Session, user_id: uuid.UUID) -> None:
        ...
    
    def update(self, db: Session, user_id: uuid.UUID, display_name: str, password_hash: str) -> User | None:
        ...