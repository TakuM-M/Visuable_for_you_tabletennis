"""動画本数クォータの enforcement テスト"""

import uuid
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.orm import Session

from app.services import video_service
from tests.fakes import FakeVideoRepository


class _CountingVideoRepository(FakeVideoRepository):
    """count_by_user_id だけ答える。他のメソッドは呼ばれたら落ちる。"""

    def __init__(self, count: int) -> None:
        self.count = count

    def count_by_user_id(self, db: Session, user_id: uuid.UUID) -> int:
        return self.count


def test_quota_not_exceeded_passes() -> None:
    """上限未満なら QuotaExceededError は raise されない"""
    db = MagicMock()
    user_id = uuid.uuid4()
    with patch("app.services.video_service.settings.user_video_quota", 10):
        video_service._ensure_under_quota(
            db, user_id, video_repo=_CountingVideoRepository(5)
        )


def test_quota_just_under_limit_passes() -> None:
    """上限直下（quota-1）はまだ通る"""
    db = MagicMock()
    user_id = uuid.uuid4()
    with patch("app.services.video_service.settings.user_video_quota", 10):
        video_service._ensure_under_quota(
            db, user_id, video_repo=_CountingVideoRepository(9)
        )


def test_quota_at_limit_raises() -> None:
    """上限ちょうどでアップロード拒否される（次に1本増えると quota+1 になるため）"""
    db = MagicMock()
    user_id = uuid.uuid4()
    with patch("app.services.video_service.settings.user_video_quota", 10):
        with pytest.raises(video_service.QuotaExceededError):
            video_service._ensure_under_quota(
                db, user_id, video_repo=_CountingVideoRepository(10)
            )


def test_quota_exceeded_raises() -> None:
    """上限超過時に QuotaExceededError が raise される"""
    db = MagicMock()
    user_id = uuid.uuid4()
    with patch("app.services.video_service.settings.user_video_quota", 10):
        with pytest.raises(video_service.QuotaExceededError) as exc:
            video_service._ensure_under_quota(
                db, user_id, video_repo=_CountingVideoRepository(15)
            )
        assert "10" in str(exc.value)
