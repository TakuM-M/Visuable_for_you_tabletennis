"""job_service の mock テスト。

DB リポジトリ・R2・メール・FFmpeg・httpx をすべて差し替え、
リトライ判定 / 失敗通知 / 手動再実行 / 完了処理の分岐を検証する。
"""

import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from app.models.job import JobStatus
from app.models.notification_log import NotificationStatus
from app.models.video import VideoStatus
from app.services import job_service


def _make_job(**kw) -> SimpleNamespace:
    defaults = dict(
        id=uuid.uuid4(),
        video_id=uuid.uuid4(),
        retry_count=0,
        status=JobStatus.failed,
    )
    defaults.update(kw)
    return SimpleNamespace(**defaults)


def _make_video(**kw) -> SimpleNamespace:
    defaults = dict(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        title="動画",
        storage_path="videos/x.mp4",
        output_path=None,
    )
    defaults.update(kw)
    return SimpleNamespace(**defaults)


def _make_user(**kw) -> SimpleNamespace:
    defaults = dict(id=uuid.uuid4(), email="user@example.com")
    defaults.update(kw)
    return SimpleNamespace(**defaults)


# --- _compute_next_retry_at -------------------------------------------------


def test_compute_next_retry_at_uses_backoff_for_index() -> None:
    with patch(
        "app.services.job_service.settings.job_retry_backoff_seconds", [60, 600]
    ):
        before = datetime.now(timezone.utc)
        result = job_service._compute_next_retry_at(0)
        after = datetime.now(timezone.utc)
    assert before + timedelta(seconds=60) <= result <= after + timedelta(seconds=60)


def test_compute_next_retry_at_clamps_to_last_value() -> None:
    """retry_count がリスト長を超えたら末尾値を使う"""
    with patch(
        "app.services.job_service.settings.job_retry_backoff_seconds", [60, 600]
    ):
        before = datetime.now(timezone.utc)
        result = job_service._compute_next_retry_at(5)
        after = datetime.now(timezone.utc)
    assert before + timedelta(seconds=600) <= result <= after + timedelta(seconds=600)


def test_compute_next_retry_at_defaults_to_60s_when_backoff_empty() -> None:
    with patch("app.services.job_service.settings.job_retry_backoff_seconds", []):
        before = datetime.now(timezone.utc)
        result = job_service._compute_next_retry_at(0)
        after = datetime.now(timezone.utc)
    assert before + timedelta(seconds=60) <= result <= after + timedelta(seconds=60)


# --- handle_ml_failure ------------------------------------------------------


def test_handle_ml_failure_returns_when_job_missing() -> None:
    db = MagicMock()
    with (
        patch("app.services.job_service.job_repo.get_by_id", return_value=None),
        patch("app.services.job_service.job_repo.mark_failed") as mark_failed,
    ):
        job_service.handle_ml_failure(db, uuid.uuid4(), "err")
    mark_failed.assert_not_called()


def test_handle_ml_failure_schedules_retry_when_under_limit() -> None:
    db = MagicMock()
    job = _make_job(retry_count=0)
    with (
        patch("app.services.job_service.job_repo.get_by_id", return_value=job),
        patch("app.services.job_service.settings.job_max_retries", 2),
        patch("app.services.job_service.job_repo.mark_failed") as mark_failed,
        patch("app.services.job_service.video_repo.update_status") as video_update,
        patch("app.services.job_service._send_failure_notification") as notify,
    ):
        job_service.handle_ml_failure(db, job.id, "boom")

    assert mark_failed.call_count == 1
    # mark_failed(db, job_id, error_message, next_retry_at) の 4 番目が next_retry_at
    assert mark_failed.call_args.args[3] is not None
    video_update.assert_not_called()
    notify.assert_not_called()


def test_handle_ml_failure_final_failure_when_exhausted() -> None:
    db = MagicMock()
    job = _make_job(retry_count=2)
    with (
        patch("app.services.job_service.job_repo.get_by_id", return_value=job),
        patch("app.services.job_service.settings.job_max_retries", 2),
        patch("app.services.job_service.job_repo.mark_failed") as mark_failed,
        patch("app.services.job_service.video_repo.update_status") as video_update,
        patch("app.services.job_service._send_failure_notification") as notify,
    ):
        job_service.handle_ml_failure(db, job.id, "boom")

    # 最終失敗では next_retry_at は None（自動リトライしない）
    assert mark_failed.call_args.args[3] is None
    video_update.assert_called_once_with(db, job.video_id, VideoStatus.failed)
    notify.assert_called_once()


# --- _send_failure_notification --------------------------------------------


def test_send_failure_notification_skips_when_video_missing() -> None:
    db = MagicMock()
    with (
        patch("app.services.job_service.video_repo.get_by_id", return_value=None),
        patch("app.services.job_service.notification_log_repo.create") as create,
    ):
        job_service._send_failure_notification(db, uuid.uuid4(), uuid.uuid4(), "err")
    create.assert_not_called()


def test_send_failure_notification_skips_when_user_missing() -> None:
    db = MagicMock()
    video = _make_video()
    with (
        patch("app.services.job_service.video_repo.get_by_id", return_value=video),
        patch("app.services.job_service.user_repo.get_by_id", return_value=None),
        patch("app.services.job_service.notification_log_repo.create") as create,
    ):
        job_service._send_failure_notification(db, uuid.uuid4(), video.id, "err")
    create.assert_not_called()


def test_send_failure_notification_records_sent_on_success() -> None:
    db = MagicMock()
    video = _make_video()
    user = _make_user(id=video.user_id)
    log = SimpleNamespace(id=7)
    with (
        patch("app.services.job_service.video_repo.get_by_id", return_value=video),
        patch("app.services.job_service.user_repo.get_by_id", return_value=user),
        patch(
            "app.services.job_service.notification_log_repo.create", return_value=log
        ),
        patch(
            "app.services.job_service.send_clip_failure_email", return_value=True
        ) as send_mail,
        patch(
            "app.services.job_service.notification_log_repo.update_status"
        ) as update_status,
    ):
        job_service._send_failure_notification(db, uuid.uuid4(), video.id, "err")

    send_mail.assert_called_once()
    assert update_status.call_args.kwargs["status"] == NotificationStatus.sent
    assert update_status.call_args.kwargs["sent_at"] is not None


def test_send_failure_notification_records_failed_when_email_fails() -> None:
    db = MagicMock()
    video = _make_video()
    user = _make_user(id=video.user_id)
    log = SimpleNamespace(id=7)
    with (
        patch("app.services.job_service.video_repo.get_by_id", return_value=video),
        patch("app.services.job_service.user_repo.get_by_id", return_value=user),
        patch(
            "app.services.job_service.notification_log_repo.create", return_value=log
        ),
        patch("app.services.job_service.send_clip_failure_email", return_value=False),
        patch(
            "app.services.job_service.notification_log_repo.update_status"
        ) as update_status,
    ):
        job_service._send_failure_notification(db, uuid.uuid4(), video.id, "err")

    assert update_status.call_args.kwargs["status"] == NotificationStatus.failed
    assert update_status.call_args.kwargs["sent_at"] is None


# --- retry_job --------------------------------------------------------------


def test_retry_job_404_when_job_missing() -> None:
    db, bt, cu = MagicMock(), MagicMock(), _make_user()
    with patch("app.services.job_service.job_repo.get_by_id", return_value=None):
        with pytest.raises(HTTPException) as exc:
            job_service.retry_job(
                db=db, job_id=uuid.uuid4(), current_user=cu, background_tasks=bt
            )
    assert exc.value.status_code == 404


def test_retry_job_404_when_video_missing() -> None:
    db, bt, cu = MagicMock(), MagicMock(), _make_user()
    job = _make_job()
    with (
        patch("app.services.job_service.job_repo.get_by_id", return_value=job),
        patch("app.services.job_service.video_repo.get_by_id", return_value=None),
    ):
        with pytest.raises(HTTPException) as exc:
            job_service.retry_job(
                db=db, job_id=job.id, current_user=cu, background_tasks=bt
            )
    assert exc.value.status_code == 404


def test_retry_job_403_when_not_owner() -> None:
    db, bt = MagicMock(), MagicMock()
    job = _make_job(status=JobStatus.failed)
    video = _make_video()
    cu = _make_user()  # video.user_id とは別人
    with (
        patch("app.services.job_service.job_repo.get_by_id", return_value=job),
        patch("app.services.job_service.video_repo.get_by_id", return_value=video),
    ):
        with pytest.raises(HTTPException) as exc:
            job_service.retry_job(
                db=db, job_id=job.id, current_user=cu, background_tasks=bt
            )
    assert exc.value.status_code == 403


def test_retry_job_409_when_not_failed() -> None:
    db, bt = MagicMock(), MagicMock()
    job = _make_job(status=JobStatus.processing)
    video = _make_video()
    cu = _make_user(id=video.user_id)
    with (
        patch("app.services.job_service.job_repo.get_by_id", return_value=job),
        patch("app.services.job_service.video_repo.get_by_id", return_value=video),
    ):
        with pytest.raises(HTTPException) as exc:
            job_service.retry_job(
                db=db, job_id=job.id, current_user=cu, background_tasks=bt
            )
    assert exc.value.status_code == 409


def test_retry_job_resets_and_rekicks_on_success() -> None:
    db, bt = MagicMock(), MagicMock()
    job = _make_job(status=JobStatus.failed)
    video = _make_video()
    cu = _make_user(id=video.user_id)
    with (
        patch("app.services.job_service.job_repo.get_by_id", return_value=job),
        patch("app.services.job_service.video_repo.get_by_id", return_value=video),
        patch("app.services.job_service.job_repo.reset_for_manual_retry") as reset,
        patch("app.services.job_service.video_repo.update_status") as vupd,
    ):
        job_service.retry_job(
            db=db, job_id=job.id, current_user=cu, background_tasks=bt
        )

    reset.assert_called_once_with(db, job.id)
    vupd.assert_called_once_with(db, video.id, VideoStatus.queued)
    bt.add_task.assert_called_once()


# --- process_complete_job ---------------------------------------------------


def test_process_complete_job_delegates_to_handle_on_error() -> None:
    job_id = uuid.uuid4()
    with (
        patch("app.services.job_service.SessionLocal") as sl,
        patch(
            "app.services.job_service.complete_job", side_effect=RuntimeError("boom")
        ),
        patch("app.services.job_service.handle_ml_failure") as handle,
    ):
        sl.return_value.__enter__.return_value = MagicMock()
        job_service.process_complete_job(job_id, [])

    handle.assert_called_once()
    assert handle.call_args.args[1] == job_id


def test_process_complete_job_no_failure_on_success() -> None:
    with (
        patch("app.services.job_service.SessionLocal") as sl,
        patch("app.services.job_service.complete_job"),
        patch("app.services.job_service.handle_ml_failure") as handle,
    ):
        sl.return_value.__enter__.return_value = MagicMock()
        job_service.process_complete_job(uuid.uuid4(), [])

    handle.assert_not_called()


# --- complete_job -----------------------------------------------------------


def test_complete_job_returns_when_job_missing() -> None:
    db = MagicMock()
    with (
        patch("app.services.job_service.job_repo.get_by_id", return_value=None),
        patch("app.services.job_service.video_repo.get_by_id") as vget,
    ):
        job_service.complete_job(db, uuid.uuid4(), [])
    vget.assert_not_called()


def test_complete_job_returns_when_video_missing() -> None:
    db = MagicMock()
    job = _make_job()
    with (
        patch("app.services.job_service.job_repo.get_by_id", return_value=job),
        patch("app.services.job_service.video_repo.get_by_id", return_value=None),
        patch("app.services.job_service.job_repo.update_status") as upd,
    ):
        job_service.complete_job(db, job.id, [])
    upd.assert_not_called()


def test_complete_job_with_empty_clips_sets_ready() -> None:
    """clips が空でも job=completed・video=ready になり、編集通知メールを送る。

    出力動画は書き出し操作時に作るので、ここでは clip も作らず FFmpeg も呼ばない。
    """
    db = MagicMock()
    job = _make_job()
    video = _make_video()
    user = _make_user(id=video.user_id)
    with (
        patch("app.services.job_service.job_repo.get_by_id", return_value=job),
        patch("app.services.job_service.video_repo.get_by_id", return_value=video),
        patch("app.services.job_service.clip_repo.create") as clip_create,
        patch("app.services.job_service.job_repo.update_status") as jupd,
        patch(
            "app.services.job_service.video_repo.update_status", return_value=video
        ) as vupd,
        patch("app.services.job_service.user_repo.get_by_id", return_value=user),
        patch(
            "app.services.job_service.notification_log_repo.create",
            return_value=SimpleNamespace(id=1),
        ),
        patch(
            "app.services.job_service.send_analysis_complete_email", return_value=True
        ) as send_mail,
        patch("app.services.job_service.notification_log_repo.update_status"),
    ):
        job_service.complete_job(db, job.id, [])

    clip_create.assert_not_called()
    assert jupd.call_args.kwargs["status"] == JobStatus.completed
    vupd.assert_called_once_with(db, job.video_id, VideoStatus.ready)
    assert send_mail.call_args.kwargs["clip_count"] == 0


def test_complete_job_with_clips_saves_and_sets_ready() -> None:
    """clips があれば検出順に sort_order を付けて保存し、video=ready にする。

    出力動画はここでは生成しないため FFmpeg(clip_video)は呼ばれない。
    """
    db = MagicMock()
    job = _make_job()
    video = _make_video()
    user = _make_user(id=video.user_id)
    clips = [
        {"start_time": 0.0, "end_time": 5.0},
        {"start_time": 6.0, "end_time": 9.0},
    ]
    with (
        patch("app.services.job_service.job_repo.get_by_id", return_value=job),
        patch("app.services.job_service.video_repo.get_by_id", return_value=video),
        patch("app.services.job_service.clip_repo.create") as clip_create,
        patch("app.services.job_service.job_repo.update_status") as jupd,
        patch(
            "app.services.job_service.video_repo.update_status", return_value=video
        ) as vupd,
        patch("app.services.job_service.user_repo.get_by_id", return_value=user),
        patch(
            "app.services.job_service.notification_log_repo.create",
            return_value=SimpleNamespace(id=1),
        ),
        patch(
            "app.services.job_service.send_analysis_complete_email", return_value=True
        ) as send_mail,
        patch("app.services.job_service.notification_log_repo.update_status"),
    ):
        job_service.complete_job(db, job.id, clips)

    assert clip_create.call_count == 2
    assert clip_create.call_args_list[0].kwargs["sort_order"] == 0
    assert clip_create.call_args_list[1].kwargs["sort_order"] == 1
    assert clip_create.call_args_list[0].kwargs["job_id"] == job.id
    assert jupd.call_args.kwargs["status"] == JobStatus.completed
    vupd.assert_called_once_with(db, job.video_id, VideoStatus.ready)
    assert send_mail.call_args.kwargs["clip_count"] == 2
