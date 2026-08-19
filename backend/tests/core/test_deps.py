"""core/deps.py の get_owned_video（動画の所有者チェック依存）のテスト。

この依存は video_id をパスに持つ動画関連エンドポイント
（詳細・削除・出力・jobs/clips 一覧）の認可を一元化している。
ここで「本人=返す / 他人=403 / 不在=404」の3系統を担保し、
各ルーターテスト側では happy path と配線（403/404 への波及）を確認する。

get_owned_video は FastAPI の依存なので、本番では video_repo が
Depends(get_video_repository) 経由で解決される。ここでは関数を直接呼ぶため、
Fake を引数で明示的に渡す。
"""

import uuid
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.core.deps import get_owned_video
from tests.fakes import FakeVideoRepository


def _user(**kw) -> SimpleNamespace:
    defaults = dict(id=uuid.uuid4())
    defaults.update(kw)
    return SimpleNamespace(**defaults)


def _video(**kw) -> SimpleNamespace:
    defaults = dict(id=uuid.uuid4(), user_id=uuid.uuid4())
    defaults.update(kw)
    return SimpleNamespace(**defaults)


class _VideoRepositoryStub(FakeVideoRepository):
    """get_by_id だけ答える。他のメソッドは呼ばれたら落ちる。"""

    def __init__(self, video: SimpleNamespace | None = None) -> None:
        self.video = video

    def get_by_id(self, db: Session, video_id: uuid.UUID):
        return self.video


def test_returns_video_for_owner() -> None:
    """ログインユーザーが所有者なら、その動画をそのまま返す。"""
    user = _user()
    video = _video(user_id=user.id)
    result = get_owned_video(
        video.id,
        current_user=user,
        db=None,
        video_repo=_VideoRepositoryStub(video),
    )
    assert result is video


def test_not_found_raises_404() -> None:
    """動画が存在しなければ 404。"""
    user = _user()
    with pytest.raises(HTTPException) as exc:
        get_owned_video(
            uuid.uuid4(),
            current_user=user,
            db=None,
            video_repo=_VideoRepositoryStub(None),
        )
    assert exc.value.status_code == 404


def test_other_users_video_raises_403() -> None:
    """他人の動画（UUID を知っていても）は 403 で弾く（IDOR 対策）。"""
    user = _user()
    video = _video(user_id=uuid.uuid4())  # 別ユーザーの所有
    with pytest.raises(HTTPException) as exc:
        get_owned_video(
            video.id,
            current_user=user,
            db=None,
            video_repo=_VideoRepositoryStub(video),
        )
    assert exc.value.status_code == 403
