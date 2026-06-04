import pytest

from jose import jwt
from app.core import security

def test_hash_password_uses_random_salt():
    """同じパスワードでも毎回違うハッシュになる（bcrypt のソルト）"""
    password = "samepassword"
    assert security.hash_password(password) != security.hash_password(password)
    
def test_verify_password():
    """verify_password は正しいパスワードで True、間違ったパスワードで False を返す"""
    hashed = security.hash_password("correctpassword")
    assert security.verify_password("correctpassword", hashed) is True
    assert security.verify_password("incorrectpassword", hashed) is False
    
def test_access_token_roundtrip():
    """create_access_token で作ったトークンを decode_token で読むと同じ user_id"""
    user_id = "user-123"
    token = security.create_access_token(user_id)
    assert security.decode_token(token) == user_id

def test_decode_token_rejects_garbage():
    """デタラメな文字列は JWTError になる"""
    with pytest.raises(jwt.JWTError):
        security.decode_token("not-a-real-token")

def test_decode_token_rejects_wrong_secret():
    """別の鍵で署名されたトークンは検証に失敗する（改ざん検知）"""
    forged = jwt.encode({"sub": "x"}, "wrong-secret", algorithm=security.ALGORITHM)
    with pytest.raises(jwt.JWTError):
        security.decode_token(forged)
        
def test_verification_token_roundtrip():
    """確認用トークンは確認用 decode で user_id に戻る"""
    user_id = "user-abc"
    token = security.create_verification_token(user_id)
    assert security.decode_verification_token(token) == user_id

def test_decode_verification_token_rejects_access_token():
    """type の無いログイン用トークンは、確認用 decode で弾かれる"""
    access = security.create_access_token("user-1")  # type フィールドが無い
    with pytest.raises(jwt.JWTError):
        security.decode_verification_token(access)
        
def test_decode_token_rejects_expired_token():
    """期限切れトークンは JWTError になる"""
    from datetime import datetime, timedelta, timezone
    from app.core.config import settings
    payload = {
        "sub": "user-1",
        "exp": datetime.now(timezone.utc) - timedelta(minutes=1),  # 1分前に失効
    }
    expired = jwt.encode(payload, settings.secret_key, algorithm=security.ALGORITHM)
    with pytest.raises(jwt.JWTError):
        security.decode_token(expired)