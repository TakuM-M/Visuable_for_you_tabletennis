"""users ルーターのテスト。

登録（新規 / 既存認証済み / 既存未認証）と、認証必須エンドポイント
(/users/me) の認証要件を TestClient で検証する。
"""

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.deps import get_current_user
from app.db.session import get_db
from app.routers import users


def _make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(users.router)
    app.dependency_overrides[get_db] = lambda: None
    return app


def _make_user(**kw) -> SimpleNamespace:
    defaults = dict(
        id=uuid.uuid4(),
        email="user@example.com",
        password_hash="hashed",
        display_name="ユーザー",
        email_verified=True,
        created_at=datetime.now(timezone.utc),
    )
    defaults.update(kw)
    return SimpleNamespace(**defaults)


def test_register_creates_new_user() -> None:
    client = TestClient(_make_app())
    user = _make_user(email="new@example.com", email_verified=False)
    with (
        patch("app.routers.users.user_repo.get_by_email", return_value=None),
        patch("app.routers.users.hash_password", return_value="hashed"),
        patch("app.routers.users.user_repo.create", return_value=user),
        patch("app.routers.users.auth_service.send_verification_email") as send,
    ):
        resp = client.post(
            "/users",
            json={
                "email": "new@example.com",
                "password": "pw123456",
                "display_name": "新規",
            },
        )

    assert resp.status_code == 201
    assert resp.json()["email"] == "new@example.com"
    send.assert_called_once()


def test_register_existing_verified_email_returns_400() -> None:
    client = TestClient(_make_app())
    existing = _make_user(email_verified=True)
    with (
        patch("app.routers.users.user_repo.get_by_email", return_value=existing),
        patch("app.routers.users.hash_password", return_value="h"),
    ):
        resp = client.post(
            "/users",
            json={"email": existing.email, "password": "pw123456", "display_name": "x"},
        )

    assert resp.status_code == 400


def test_register_existing_unverified_updates_and_resends() -> None:
    client = TestClient(_make_app())
    existing = _make_user(email_verified=False)
    with (
        patch("app.routers.users.user_repo.get_by_email", return_value=existing),
        patch("app.routers.users.hash_password", return_value="h"),
        patch("app.routers.users.user_repo.update") as update,
        patch("app.routers.users.auth_service.send_verification_email") as send,
    ):
        resp = client.post(
            "/users",
            json={
                "email": existing.email,
                "password": "pw123456",
                "display_name": "再登録",
            },
        )

    assert resp.status_code == 201
    update.assert_called_once()
    send.assert_called_once()


def test_get_me_requires_auth() -> None:
    client = TestClient(_make_app())
    resp = client.get("/users/me")
    assert resp.status_code == 401


def test_get_me_returns_current_user() -> None:
    app = _make_app()
    user = _make_user()
    app.dependency_overrides[get_current_user] = lambda: user
    client = TestClient(app)

    resp = client.get("/users/me")
    assert resp.status_code == 200
    assert resp.json()["email"] == user.email


def test_update_me_requires_auth() -> None:
    client = TestClient(_make_app())
    resp = client.patch("/users/me", json={"display_name": "x"})
    assert resp.status_code == 401


def test_update_me_updates_user() -> None:
    app = _make_app()
    user = _make_user()
    updated = _make_user(id=user.id, display_name="新しい名前")
    app.dependency_overrides[get_current_user] = lambda: user
    client = TestClient(app)

    with (
        patch("app.routers.users.hash_password", return_value="h"),
        patch("app.routers.users.user_repo.update") as update,
        patch("app.routers.users.user_repo.get_by_id", return_value=updated),
    ):
        resp = client.patch("/users/me", json={"display_name": "新しい名前"})

    assert resp.status_code == 200
    assert resp.json()["display_name"] == "新しい名前"
    update.assert_called_once()
