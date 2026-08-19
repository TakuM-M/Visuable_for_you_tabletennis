"""auth_service の mock テスト。

外部 I/O（Resend メール送信・JWT 生成/検証）は patch で、user リポジトリは
Fake の注入で差し替えて、トークン生成 → 送信 / トークン検証 → email_verified
更新 の振る舞いを検証する。
"""

import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from jose import JWTError
from sqlalchemy.orm import Session

from app.services import auth_service
from tests.fakes import FakeUserRepository


def _make_user() -> SimpleNamespace:
    return SimpleNamespace(id=uuid.uuid4(), email="user@example.com")


class _RecordingUserRepository(FakeUserRepository):
    """verify_email の呼び出しだけを記録する。他のメソッドは呼ばれたら落ちる。"""

    def __init__(self) -> None:
        self.verify_email_calls: list[tuple[Session, uuid.UUID]] = []

    def verify_email(self, db: Session, user_id: uuid.UUID) -> None:
        self.verify_email_calls.append((db, user_id))


def test_send_verification_email_sends_via_resend() -> None:
    """トークンを生成し、Resend にユーザー宛メールを送る"""
    user = _make_user()
    with (
        patch(
            "app.services.auth_service.create_verification_token", return_value="tok123"
        ),
        patch("app.services.auth_service.resend.Emails.send") as send_mock,
    ):
        auth_service.send_verification_email(user)

    send_mock.assert_called_once()
    payload = send_mock.call_args.args[0]
    assert payload["to"] == user.email
    assert "tok123" in payload["html"]


def test_send_verification_email_swallows_send_errors() -> None:
    """Resend 送信が例外を投げても呼び出し側へは伝播しない（silent）"""
    user = _make_user()
    with (
        patch(
            "app.services.auth_service.create_verification_token", return_value="tok"
        ),
        patch(
            "app.services.auth_service.resend.Emails.send",
            side_effect=RuntimeError("resend down"),
        ),
    ):
        # 例外が外に出ないこと
        auth_service.send_verification_email(user)


def test_verify_email_token_marks_email_verified() -> None:
    """正常トークンなら verify_email がデコードした user_id で呼ばれる"""
    db = MagicMock()
    user_id = uuid.uuid4()
    fake_repo = _RecordingUserRepository()
    with patch(
        "app.services.auth_service.decode_verification_token",
        return_value=str(user_id),
    ):
        auth_service.verify_email_token("tok", db, user_repo=fake_repo)

    assert fake_repo.verify_email_calls == [(db, user_id)]


def test_verify_email_token_raises_value_error_on_invalid_token() -> None:
    """JWTError は ValueError に変換される（ルーターが 400 に変換する前提）"""
    db = MagicMock()
    fake_repo = _RecordingUserRepository()
    with patch(
        "app.services.auth_service.decode_verification_token",
        side_effect=JWTError("bad"),
    ):
        with pytest.raises(ValueError):
            auth_service.verify_email_token("bad-token", db, user_repo=fake_repo)

    assert fake_repo.verify_email_calls == []
