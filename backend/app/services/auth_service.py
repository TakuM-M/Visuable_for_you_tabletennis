import os
import uuid

import resend
from jose import JWTError
from sqlalchemy.orm import Session

from app.core.security import create_verification_token, decode_verification_token
from app.repositories import user as user_repo

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")
FROM_EMAIL = os.getenv("RESEND_FROM_EMAIL", "onboarding@resend.dev")
resend.api_key = os.getenv("RESEND_API_KEY", "")


def send_verification_email(user) -> None:
    token = create_verification_token(str(user.id))
    url = f"{FRONTEND_URL}/verify-email?token={token}"
    try:
        resend.Emails.send(
            {
                "from": FROM_EMAIL,
                "to": user.email,
                "subject": "【ClipMaster】メールアドレスの確認",
                "html": f"<h2>メールアドレスの確認</h2><a href='{url}'>確認する</a>",
            }
        )
    except Exception as e:
        print(f"メール送信失敗: {e}")


def verify_email_token(token: str, db: Session) -> None:
    try:
        user_id = decode_verification_token(token)
    except JWTError:
        raise ValueError("不正なトークン")
    user_repo.verify_email(db, uuid.UUID(user_id))
