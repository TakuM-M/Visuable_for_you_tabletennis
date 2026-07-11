"""user repository の実 DB テスト。

conftest.py の `db` fixture を引数で受け取ると、テスト用 PostgreSQL に
接続済みの SQLAlchemy Session が渡される。各テスト終了時に全テーブルが
truncate されるので、テスト同士は独立。
"""
import uuid

import pytest
from sqlalchemy.exc import IntegrityError

from app.repositories import user as user_repo


def test_create_returns_user_with_generated_fields(db):
    """create() は id / created_at / updated_at / email_verified=False を自動設定"""
    user = user_repo.create(
        db=db,
        email="alice@example.com",
        password_hash="hashed_pw",
        display_name="Alice",
    )
    assert isinstance(user.id, uuid.UUID)  # default=uuid.uuid4 で自動採番
    assert user.email == "alice@example.com"
    assert user.password_hash == "hashed_pw"
    assert user.display_name == "Alice"
    assert user.email_verified is False  # User モデルのデフォルト
    assert user.created_at is not None  # server_default=now() で DB が埋める
    assert user.updated_at is not None


def test_get_by_id_returns_created_user(db):
    """create したユーザーを id で取り戻せる"""
    created = user_repo.create(
        db=db, email="bob@example.com", password_hash="x", display_name="Bob"
    )
    fetched = user_repo.get_by_id(db, created.id)
    assert fetched is not None
    assert fetched.id == created.id
    assert fetched.email == "bob@example.com"


def test_get_by_id_returns_none_when_not_found(db):
    """存在しない id では None が返る（例外ではない）"""
    result = user_repo.get_by_id(db, uuid.uuid4())
    assert result is None


def test_get_by_email_finds_existing_user(db):
    """email で検索できる"""
    user_repo.create(
        db, email="carol@example.com", password_hash="x", display_name="Carol"
    )
    fetched = user_repo.get_by_email(db, "carol@example.com")
    assert fetched is not None
    assert fetched.display_name == "Carol"


def test_email_unique_constraint_is_enforced(db):
    """users.email の UNIQUE 制約により、同じメールで 2 人は作れない"""
    user_repo.create(
        db, email="dup@example.com", password_hash="x", display_name="One"
    )
    with pytest.raises(IntegrityError):
        user_repo.create(
            db, email="dup@example.com", password_hash="y", display_name="Two"
        )
    db.rollback()  # IntegrityError 後は session がアボート状態なので明示的に戻す


def test_verify_email_sets_flag_true(db):
    """verify_email を呼ぶと email_verified が True になる"""
    user = user_repo.create(
        db, email="eve@example.com", password_hash="x", display_name="Eve"
    )
    assert user.email_verified is False
    user_repo.verify_email(db, user.id)
    db.refresh(user)  # DB の最新状態を Python オブジェクトに反映
    assert user.email_verified is True


def test_update_changes_display_name_and_password(db):
    """update は display_name と password_hash の両方を書き換える"""
    user = user_repo.create(
        db, email="frank@example.com", password_hash="old_hash", display_name="Frank"
    )
    user_repo.update(
        db, user.id, display_name="Franklin", password_hash="new_hash"
    )
    db.refresh(user)
    assert user.display_name == "Franklin"
    assert user.password_hash == "new_hash"
