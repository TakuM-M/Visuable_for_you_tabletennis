"""notification_log リポジトリの実 DB テスト。

user / job への FK を持つ。status のデフォルトと update_status の部分更新が要点。
"""

import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy.exc import IntegrityError

from app.models.notification_log import NotificationStatus
from app.repositories.job import job_repository as job_repo
from app.repositories.notification_log import notification_log_repository as nlog_repo


def test_create_sets_defaults(db, user, job):
    log = nlog_repo.create(
        db=db,
        user_id=user.id,
        job_id=job.id,
        email="to@example.com",
    )

    assert isinstance(log.id, int)
    assert log.user_id == user.id
    assert log.job_id == job.id
    assert log.email == "to@example.com"
    assert log.status == NotificationStatus.pending
    assert log.sent_at is None
    assert log.created_at is not None


def test_create_with_unknown_user_id_violates_fk(db, job):
    with pytest.raises(IntegrityError):
        nlog_repo.create(
            db=db, user_id=uuid.uuid4(), job_id=job.id, email="x@example.com"
        )
    db.rollback()


def test_create_with_unknown_job_id_violates_fk(db, user):
    with pytest.raises(IntegrityError):
        nlog_repo.create(
            db=db, user_id=user.id, job_id=uuid.uuid4(), email="x@example.com"
        )
    db.rollback()


def test_update_status_changes_status_and_sets_sent_at(db, notification_log):
    sent_at = datetime.now(timezone.utc)
    updated = nlog_repo.update_status(
        db=db,
        log_id=notification_log.id,
        status=NotificationStatus.sent,
        sent_at=sent_at,
    )

    assert updated is not None
    assert updated.status == NotificationStatus.sent
    assert updated.sent_at is not None


def test_update_status_without_sent_at_keeps_it_none(db, notification_log):
    """sent_at を省略した場合は更新されない（部分更新）"""
    updated = nlog_repo.update_status(
        db=db,
        log_id=notification_log.id,
        status=NotificationStatus.failed,
    )

    assert updated.status == NotificationStatus.failed
    assert updated.sent_at is None


def test_update_status_returns_none_for_unknown_id(db):
    assert (
        nlog_repo.update_status(db=db, log_id=999999, status=NotificationStatus.sent)
        is None
    )


def test_get_by_job_id_returns_only_that_jobs_logs(db, user, video, job):
    nlog_repo.create(db=db, user_id=user.id, job_id=job.id, email="a@example.com")
    other_job = job_repo.create(db=db, video_id=video.id)
    nlog_repo.create(db=db, user_id=user.id, job_id=other_job.id, email="b@example.com")

    result = nlog_repo.get_by_job_id(db, job.id)
    assert len(result) == 1
    assert result[0].email == "a@example.com"


def test_delete_by_job_id_removes_all_and_returns_count(db, user, job):
    nlog_repo.create(db=db, user_id=user.id, job_id=job.id, email="a@example.com")
    nlog_repo.create(db=db, user_id=user.id, job_id=job.id, email="b@example.com")

    count = nlog_repo.delete_by_job_id(db, job.id)
    assert count == 2
    assert nlog_repo.get_by_job_id(db, job.id) == []


def test_delete_by_job_id_returns_zero_when_none(db, job):
    assert nlog_repo.delete_by_job_id(db, job.id) == 0
