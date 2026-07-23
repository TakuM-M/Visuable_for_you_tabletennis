"""auth ルーターのテスト。

DB / セキュリティ関数 / auth_service を mock し、TestClient でログインと
メール認証エンドポイントのステータスコード分岐を検証する。
"""

import uuid
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.db.session import get_db
from app.routers import auth


def _make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(auth.router)
    app.dependency_overrides[get_db] = lambda: None
    return app


def _make_user(**kw) -> SimpleNamespace:
    defaults = dict(
        id=uuid.uuid4(),
        email="user@example.com",
        password_hash="hashed",
        email_verified=True,
    )
    defaults.update(kw)
    return SimpleNamespace(**defaults)


def test_login_success_returns_token() -> None:
    client = TestClient(_make_app())
    user = _make_user()
    with (
        patch("app.routers.auth.user_repo.get_by_email", return_value=user),
        patch("app.routers.auth.verify_password", return_value=True),
        patch("app.routers.auth.create_access_token", return_value="tok123"),
    ):
        resp = client.post(
            "/auth/login", data={"username": user.email, "password": "pw"}
        )

    assert resp.status_code == 200
    assert resp.json()["access_token"] == "tok123"


def test_login_wrong_password_returns_401() -> None:
    client = TestClient(_make_app())
    user = _make_user()
    with (
        patch("app.routers.auth.user_repo.get_by_email", return_value=user),
        patch("app.routers.auth.verify_password", return_value=False),
    ):
        resp = client.post(
            "/auth/login", data={"username": user.email, "password": "wrong"}
        )

    assert resp.status_code == 401


def test_login_unknown_user_returns_401() -> None:
    client = TestClient(_make_app())
    with patch("app.routers.auth.user_repo.get_by_email", return_value=None):
        resp = client.post(
            "/auth/login", data={"username": "nobody@example.com", "password": "pw"}
        )

    assert resp.status_code == 401


def test_login_unverified_email_returns_403() -> None:
    client = TestClient(_make_app())
    user = _make_user(email_verified=False)
    with (
        patch("app.routers.auth.user_repo.get_by_email", return_value=user),
        patch("app.routers.auth.verify_password", return_value=True),
    ):
        resp = client.post(
            "/auth/login", data={"username": user.email, "password": "pw"}
        )

    assert resp.status_code == 403


def test_verify_email_success() -> None:
    client = TestClient(_make_app())
    with patch("app.routers.auth.auth_service.verify_email_token") as verify:
        resp = client.get("/auth/verify-email", params={"token": "abc"})

    assert resp.status_code == 200
    verify.assert_called_once()


def test_verify_email_invalid_token_returns_400() -> None:
    client = TestClient(_make_app())
    with patch(
        "app.routers.auth.auth_service.verify_email_token",
        side_effect=ValueError("不正なトークン"),
    ):
        resp = client.get("/auth/verify-email", params={"token": "bad"})

    assert resp.status_code == 400
