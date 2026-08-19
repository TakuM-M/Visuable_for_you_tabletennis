"""保持期限切れ動画の自動削除バッチのテスト"""

import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sqlalchemy.orm import Session

from app.services import job_reaper
from tests.fakes import FakeVideoRepository


def _make_video(video_id: uuid.UUID) -> SimpleNamespace:
    return SimpleNamespace(id=video_id)


class _ExpiredVideoRepository(FakeVideoRepository):
    """期限切れ動画を答え、渡された threshold を記録する。"""

    def __init__(self, videos: list) -> None:
        self.videos = videos
        self.thresholds: list[datetime] = []

    def get_expired(self, db: Session, threshold: datetime) -> list:
        self.thresholds.append(threshold)
        return self.videos


def test_cleanup_calls_delete_for_each_expired_video() -> None:
    """get_expired が返した動画それぞれに delete_video が呼ばれる"""
    ids = [uuid.uuid4() for _ in range(3)]
    expired = [_make_video(i) for i in ids]

    with (
        patch("app.services.job_reaper.SessionLocal") as session_local,
        patch(
            "app.services.job_reaper.video_service.delete_video", return_value=True
        ) as delete_mock,
        patch("app.services.job_reaper.settings.video_retention_days", 7.0),
    ):
        session_local.return_value.__enter__.return_value = MagicMock()
        job_reaper.cleanup_expired_videos(video_repo=_ExpiredVideoRepository(expired))

    assert delete_mock.call_count == 3
    called_ids = {call.args[1] for call in delete_mock.call_args_list}
    assert called_ids == set(ids)


def test_cleanup_uses_retention_threshold() -> None:
    """get_expired に渡される threshold が now - retention_days になっている"""
    repo = _ExpiredVideoRepository([])
    with (
        patch("app.services.job_reaper.SessionLocal") as session_local,
        patch("app.services.job_reaper.settings.video_retention_days", 7.0),
    ):
        session_local.return_value.__enter__.return_value = MagicMock()
        before = datetime.now(timezone.utc)
        job_reaper.cleanup_expired_videos(video_repo=repo)
        after = datetime.now(timezone.utc)

    threshold = repo.thresholds[0]
    assert before - timedelta(days=7) <= threshold <= after - timedelta(days=7)


def test_cleanup_continues_when_one_delete_fails() -> None:
    """1件の削除失敗で他の削除が止まらない"""
    ids = [uuid.uuid4() for _ in range(3)]
    expired = [_make_video(i) for i in ids]

    def delete_side_effect(db, video_id):
        if video_id == ids[1]:
            raise RuntimeError("R2 障害")
        return True

    with (
        patch("app.services.job_reaper.SessionLocal") as session_local,
        patch(
            "app.services.job_reaper.video_service.delete_video",
            side_effect=delete_side_effect,
        ) as delete_mock,
        patch("app.services.job_reaper.settings.video_retention_days", 7.0),
    ):
        session_local.return_value.__enter__.return_value = MagicMock()
        job_reaper.cleanup_expired_videos(video_repo=_ExpiredVideoRepository(expired))

    # 3件全部試みた
    assert delete_mock.call_count == 3
