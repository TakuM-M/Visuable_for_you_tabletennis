"""メール送信サービスのテスト。

email_service は外部 API (Resend) をネットワーク越しに叩くので、
`resend.Emails.send` を mock で差し替える。これにより:
  - 実際にメールを送らずに済む（副作用なし）
  - ネットワークに依存しない（毎回同じ結果＝flaky でない）
  - 「成功時 True / 失敗時 False」「正しいペイロードで呼んだか」を検証できる
"""
from unittest.mock import patch

from app.services import email_service


def test_send_completion_email_returns_true_on_success():
    """resend.Emails.send が正常終了すれば True を返す"""
    with patch("app.services.email_service.resend.Emails.send") as send_mock:
        result = email_service.send_clip_completion_email(
            to_email="user@example.com",
            video_title="テスト動画",
            clip_count=5,
            video_url="https://example.com/video/123",
        )
    assert result is True
    send_mock.assert_called_once()  # 送信は1回だけ行われた


def test_send_completion_email_returns_false_on_failure():
    """resend.Emails.send が例外を投げたら except 経路で False を返す"""
    with patch(
        "app.services.email_service.resend.Emails.send",
        side_effect=Exception("Resend API error"),
    ):
        result = email_service.send_clip_completion_email(
            to_email="user@example.com",
            video_title="テスト動画",
            clip_count=5,
            video_url="https://example.com/video/123",
        )
    assert result is False


def test_send_completion_email_builds_correct_payload():
    """送信ペイロードの宛先・件名が引数から正しく組み立てられる"""
    with patch("app.services.email_service.resend.Emails.send") as send_mock:
        email_service.send_clip_completion_email(
            to_email="user@example.com",
            video_title="練習試合",
            clip_count=3,
            video_url="https://example.com/v/1",
        )
    payload = send_mock.call_args.args[0]  # send({...}) に渡した辞書
    assert payload["to"] == "user@example.com"
    assert "練習試合" in payload["subject"]


def test_send_failure_email_returns_true_on_success():
    """失敗通知メールも同様に、成功すれば True を返す"""
    with patch("app.services.email_service.resend.Emails.send") as send_mock:
        result = email_service.send_clip_failure_email(
            to_email="user@example.com",
            video_title="テスト動画",
            video_url="https://example.com/v/1",
            error_message="ML タイムアウト",
        )
    assert result is True
    send_mock.assert_called_once()
