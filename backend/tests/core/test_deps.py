"""core/deps.py の get_owned_video（動画の所有者チェック依存）のテスト。

この依存は video_id をパスに持つ動画関連エンドポイント
（詳細・削除・出力・jobs/clips 一覧）の認可を一元化している。
ここで「本人=返す / 他人=403 / 不在=404」の3系統を担保し、
各ルーターテスト側では happy path と配線（403/404 への波及）を確認する。
"""
import uuid
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from app.core.deps import get_owned_video


def _user(**kw) -> SimpleNamespace:
    defaults = dict(id=uuid.uuid4())
    defaults.update(kw)
    return SimpleNamespace(**defaults)


def _video(**kw) -> SimpleNamespace:
    defaults = dict(id=uuid.uuid4(), user_id=uuid.uuid4())
    defaults.update(kw)
    return SimpleNamespace(**defaults)


def test_returns_video_for_owner() -> None:
    """ログインユーザーが所有者なら、その動画をそのまま返す。"""
    user = _user()
    video = _video(user_id=user.id)
    with patch("app.core.deps.video_repo.get_by_id", return_value=video):
        result = get_owned_video(video.id, current_user=user, db=None)
    assert result is video


def test_not_found_raises_404() -> None:
    """動画が存在しなければ 404。"""
    user = _user()
    with patch("app.core.deps.video_repo.get_by_id", return_value=None):
        with pytest.raises(HTTPException) as exc:
            get_owned_video(uuid.uuid4(), current_user=user, db=None)
    assert exc.value.status_code == 404


def test_other_users_video_raises_403() -> None:
    """他人の動画（UUID を知っていても）は 403 で弾く（IDOR 対策）。"""
    user = _user()
    video = _video(user_id=uuid.uuid4())  # 別ユーザーの所有
    with patch("app.core.deps.video_repo.get_by_id", return_value=video):
        with pytest.raises(HTTPException) as exc:
            get_owned_video(video.id, current_user=user, db=None)
    assert exc.value.status_code == 403
