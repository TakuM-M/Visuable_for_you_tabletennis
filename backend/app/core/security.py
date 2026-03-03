from datetime import datetime, timedelta, timezone

from jose import jwt
from passlib.context import CryptContext

from app.core.config import settings

# --- パスワードハッシュ ---

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    """平文パスワードを bcrypt でハッシュ化する"""
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    """入力パスワードとDBのハッシュを照合する"""
    return pwd_context.verify(plain, hashed)


# --- JWT トークン ---

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24時間


def create_access_token(user_id: str) -> str:
    """user_id を埋め込んだ JWT トークンを生成する"""
    payload = {
        "sub": user_id,
        "exp": datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, settings.secret_key, algorithm=ALGORITHM)

def decode_token(token: str) -> str:
    """JWT トークンを検証して user_id を返す。不正なトークンは JWTError を送出する"""
    payload = jwt.decode(token, settings.secret_key, algorithms=[ALGORITHM])
    return payload["sub"]

def create_verification_token(user_id: str) -> str:
    """メール確認用の JWT トークンを生成する"""
    payload = {
        "sub": user_id,
        "type": "email_verification",  # ログイン用トークンと区別するためのフィールド
        "exp": datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, settings.secret_key, algorithm=ALGORITHM)

def decode_verification_token(token: str) -> str:
    """メール確認用の JWT トークンを検証して user_id を返す。不正なトークンは JWTError を送出する"""
    payload = jwt.decode(token, settings.secret_key, algorithms=[ALGORITHM])
    if payload.get("type") != "email_verification":
        raise jwt.JWTError("不正なトークン")
    return payload["sub"]