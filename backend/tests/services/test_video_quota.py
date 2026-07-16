"""動画本数クォータの enforcement テスト"""

import uuid
from unittest.mock import MagicMock, patch

import pytest

from app.services import video_service


def test_quota_not_exceeded_passes() -> None:
    """上限未満なら QuotaExceededError は raise されない"""
    db = MagicMock()
    user_id = uuid.uuid4()
    with (
        patch("app.services.video_service.video_repo.count_by_user_id", return_value=5),
        patch("app.services.video_service.settings.user_video_quota", 10),
    ):
        video_service._ensure_under_quota(db, user_id)


def test_quota_just_under_limit_passes() -> None:
    """上限直下（quota-1）はまだ通る"""
    db = MagicMock()
    user_id = uuid.uuid4()
    with (
        patch("app.services.video_service.video_repo.count_by_user_id", return_value=9),
        patch("app.services.video_service.settings.user_video_quota", 10),
    ):
        video_service._ensure_under_quota(db, user_id)


def test_quota_at_limit_raises() -> None:
    """上限ちょうどでアップロード拒否される（次に1本増えると quota+1 になるため）"""
    db = MagicMock()
    user_id = uuid.uuid4()
    with (
        patch(
            "app.services.video_service.video_repo.count_by_user_id", return_value=10
        ),
        patch("app.services.video_service.settings.user_video_quota", 10),
    ):
        with pytest.raises(video_service.QuotaExceededError):
            video_service._ensure_under_quota(db, user_id)


def test_quota_exceeded_raises() -> None:
    """上限超過時に QuotaExceededError が raise される"""
    db = MagicMock()
    user_id = uuid.uuid4()
    with (
        patch(
            "app.services.video_service.video_repo.count_by_user_id", return_value=15
        ),
        patch("app.services.video_service.settings.user_video_quota", 10),
    ):
        with pytest.raises(video_service.QuotaExceededError) as exc:
            video_service._ensure_under_quota(db, user_id)
        assert "10" in str(exc.value)
