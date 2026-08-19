"""job_service の mock テスト。

DB リポジトリは Fake の注入で、R2・メール・FFmpeg・httpx は patch で差し替え、
リトライ判定 / 失敗通知 / 手動再実行 / 完了処理の分岐を検証する。
"""

import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.models.job import JobStatus
from app.models.notification_log import NotificationStatus
from app.models.video import VideoStatus
from app.services import job_service
from tests.fakes import (
    FakeClipRepository,
    FakeJobRepository,
    FakeNotificationLogRepository,
    FakeUserRepository,
    FakeVideoRepository,
)


def _make_job(**kw) -> SimpleNamespace:
    defaults = dict(
        id=uuid.uuid4(),
        video_id=uuid.uuid4(),
        retry_count=0,
        status=JobStatus.failed,
        runpod_job_id=None,
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


# --- リポジトリ Fake --------------------------------------------------------
#   job_service が実際に呼ぶメソッドだけを実装し、呼び出しを記録する。
#   未実装のメソッドは土台（tests/fakes.py）が NotImplementedError を送出するので、
#   想定外の呼び出しはその場でテストが落ちる。


class _JobRepositoryStub(FakeJobRepository):
    def __init__(self, job: SimpleNamespace | None = None) -> None:
        self.job = job
        self.mark_failed_calls: list[dict] = []
        self.update_status_calls: list[dict] = []
        self.reset_calls: list[uuid.UUID] = []

    def get_by_id(self, db: Session, job_id: uuid.UUID):
        return self.job

    def mark_failed(self, db: Session, job_id, error_message, next_retry_at):
        self.mark_failed_calls.append(
            {
                "job_id": job_id,
                "error_message": error_message,
                "next_retry_at": next_retry_at,
            }
        )
        return None

    def update_status(
        self,
        db: Session,
        job_id,
        status,
        started_at=None,
        completed_at=None,
        error_message=None,
    ):
        self.update_status_calls.append(
            {"job_id": job_id, "status": status, "completed_at": completed_at}
        )
        return None

    def reset_for_manual_retry(self, db: Session, job_id):
        self.reset_calls.append(job_id)
        return None


class _VideoRepositoryStub(FakeVideoRepository):
    def __init__(self, video: SimpleNamespace | None = None) -> None:
        self.video = video
        self.update_status_calls: list[tuple] = []

    def get_by_id(self, db: Session, video_id: uuid.UUID):
        return self.video

    def update_status(self, db: Session, video_id, status):
        self.update_status_calls.append((db, video_id, status))
        return self.video


class _UserRepositoryStub(FakeUserRepository):
    def __init__(self, user: SimpleNamespace | None = None) -> None:
        self.user = user

    def get_by_id(self, db: Session, user_id: uuid.UUID):
        return self.user


class _ClipRepositoryStub(FakeClipRepository):
    def __init__(self) -> None:
        self.created: list[dict] = []

    def create(
        self,
        db: Session,
        video_id,
        job_id,
        start_time,
        end_time,
        storage_path,
        sort_order=0,
    ):
        self.created.append(
            {
                "video_id": video_id,
                "job_id": job_id,
                "start_time": start_time,
                "end_time": end_time,
                "storage_path": storage_path,
                "sort_order": sort_order,
            }
        )
        return SimpleNamespace(id=uuid.uuid4())


class _NotificationLogRepositoryStub(FakeNotificationLogRepository):
    def __init__(self, log: SimpleNamespace | None = None) -> None:
        self.log = log if log is not None else SimpleNamespace(id=1)
        self.created: list[dict] = []
        self.status_calls: list[dict] = []

    def create(self, db: Session, user_id, job_id, email):
        self.created.append({"user_id": user_id, "job_id": job_id, "email": email})
        return self.log

    def update_status(self, db: Session, log_id, status, sent_at=None):
        self.status_calls.append(
            {"log_id": log_id, "status": status, "sent_at": sent_at}
        )
        return None


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
    job_repo = _JobRepositoryStub(None)
    job_service.handle_ml_failure(db, uuid.uuid4(), "err", job_repo=job_repo)
    assert job_repo.mark_failed_calls == []


def test_handle_ml_failure_schedules_retry_when_under_limit() -> None:
    db = MagicMock()
    job = _make_job(retry_count=0)
    job_repo = _JobRepositoryStub(job)
    video_repo = _VideoRepositoryStub()
    with (
        patch("app.services.job_service.settings.job_max_retries", 2),
        patch("app.services.job_service._send_failure_notification") as notify,
    ):
        job_service.handle_ml_failure(
            db, job.id, "boom", job_repo=job_repo, video_repo=video_repo
        )

    assert len(job_repo.mark_failed_calls) == 1
    assert job_repo.mark_failed_calls[0]["next_retry_at"] is not None
    assert video_repo.update_status_calls == []
    notify.assert_not_called()


def test_handle_ml_failure_final_failure_when_exhausted() -> None:
    db = MagicMock()
    job = _make_job(retry_count=2)
    job_repo = _JobRepositoryStub(job)
    video_repo = _VideoRepositoryStub()
    with (
        patch("app.services.job_service.settings.job_max_retries", 2),
        patch("app.services.job_service._send_failure_notification") as notify,
    ):
        job_service.handle_ml_failure(
            db, job.id, "boom", job_repo=job_repo, video_repo=video_repo
        )

    # 最終失敗では next_retry_at は None（自動リトライしない）
    assert job_repo.mark_failed_calls[0]["next_retry_at"] is None
    assert video_repo.update_status_calls == [(db, job.video_id, VideoStatus.failed)]
    notify.assert_called_once()


def test_handle_ml_failure_cancels_runpod_job() -> None:
    """失敗を記録する前に GPU を止める。

    ここを飛ばすと RunPod のワーカーが走り続けて課金が止まらず、さらに
    リトライで GPU が並列に増えてしまう。
    """
    db = MagicMock()
    job = _make_job(retry_count=0, runpod_job_id="rp-1")
    job_repo = _JobRepositoryStub(job)
    with (
        patch("app.services.job_service.settings.job_max_retries", 2),
        patch("app.services.job_service.runpod_service.cancel_job") as cancel,
    ):
        job_service.handle_ml_failure(db, job.id, "boom", job_repo=job_repo)

    cancel.assert_called_once_with("rp-1")


def test_handle_ml_failure_skips_cancel_without_runpod_id() -> None:
    """ml-mock 経路（runpod_job_id が無い）では停止 API を叩かない"""
    db = MagicMock()
    job = _make_job(retry_count=0, runpod_job_id=None)
    job_repo = _JobRepositoryStub(job)
    with (
        patch("app.services.job_service.settings.job_max_retries", 2),
        patch("app.services.job_service.runpod_service.cancel_job") as cancel,
    ):
        job_service.handle_ml_failure(db, job.id, "boom", job_repo=job_repo)

    cancel.assert_not_called()


def test_handle_ml_failure_proceeds_when_cancel_fails() -> None:
    """停止に失敗しても失敗処理は最後まで進む（DB が未更新のまま残らない）"""
    db = MagicMock()
    job = _make_job(retry_count=0, runpod_job_id="rp-1")
    job_repo = _JobRepositoryStub(job)
    with (
        patch("app.services.job_service.settings.job_max_retries", 2),
        patch(
            "app.services.job_service.runpod_service.cancel_job", return_value=False
        ),
    ):
        job_service.handle_ml_failure(db, job.id, "boom", job_repo=job_repo)

    assert len(job_repo.mark_failed_calls) == 1


# --- _send_failure_notification --------------------------------------------


def test_send_failure_notification_skips_when_video_missing() -> None:
    db = MagicMock()
    nlog_repo = _NotificationLogRepositoryStub()
    job_service._send_failure_notification(
        db,
        uuid.uuid4(),
        uuid.uuid4(),
        "err",
        video_repo=_VideoRepositoryStub(None),
        # 動画が無い時点で打ち切るので、ユーザー照会には進まない（進めば落ちる）
        user_repo=FakeUserRepository(),
        notification_log_repo=nlog_repo,
    )
    assert nlog_repo.created == []


def test_send_failure_notification_skips_when_user_missing() -> None:
    db = MagicMock()
    video = _make_video()
    nlog_repo = _NotificationLogRepositoryStub()
    job_service._send_failure_notification(
        db,
        uuid.uuid4(),
        video.id,
        "err",
        video_repo=_VideoRepositoryStub(video),
        user_repo=_UserRepositoryStub(None),
        notification_log_repo=nlog_repo,
    )
    assert nlog_repo.created == []


def test_send_failure_notification_records_sent_on_success() -> None:
    db = MagicMock()
    video = _make_video()
    user = _make_user(id=video.user_id)
    nlog_repo = _NotificationLogRepositoryStub(SimpleNamespace(id=7))
    with patch(
        "app.services.job_service.send_clip_failure_email", return_value=True
    ) as send_mail:
        job_service._send_failure_notification(
            db,
            uuid.uuid4(),
            video.id,
            "err",
            video_repo=_VideoRepositoryStub(video),
            user_repo=_UserRepositoryStub(user),
            notification_log_repo=nlog_repo,
        )

    send_mail.assert_called_once()
    assert nlog_repo.status_calls[0]["status"] == NotificationStatus.sent
    assert nlog_repo.status_calls[0]["sent_at"] is not None


def test_send_failure_notification_records_failed_when_email_fails() -> None:
    db = MagicMock()
    video = _make_video()
    user = _make_user(id=video.user_id)
    nlog_repo = _NotificationLogRepositoryStub(SimpleNamespace(id=7))
    with patch("app.services.job_service.send_clip_failure_email", return_value=False):
        job_service._send_failure_notification(
            db,
            uuid.uuid4(),
            video.id,
            "err",
            video_repo=_VideoRepositoryStub(video),
            user_repo=_UserRepositoryStub(user),
            notification_log_repo=nlog_repo,
        )

    assert nlog_repo.status_calls[0]["status"] == NotificationStatus.failed
    assert nlog_repo.status_calls[0]["sent_at"] is None


# --- retry_job --------------------------------------------------------------


def test_retry_job_404_when_job_missing() -> None:
    db, bt, cu = MagicMock(), MagicMock(), _make_user()
    with pytest.raises(HTTPException) as exc:
        job_service.retry_job(
            db=db,
            job_id=uuid.uuid4(),
            current_user=cu,
            background_tasks=bt,
            job_repo=_JobRepositoryStub(None),
        )
    assert exc.value.status_code == 404


def test_retry_job_404_when_video_missing() -> None:
    db, bt, cu = MagicMock(), MagicMock(), _make_user()
    job = _make_job()
    with pytest.raises(HTTPException) as exc:
        job_service.retry_job(
            db=db,
            job_id=job.id,
            current_user=cu,
            background_tasks=bt,
            job_repo=_JobRepositoryStub(job),
            video_repo=_VideoRepositoryStub(None),
        )
    assert exc.value.status_code == 404


def test_retry_job_403_when_not_owner() -> None:
    db, bt = MagicMock(), MagicMock()
    job = _make_job(status=JobStatus.failed)
    video = _make_video()
    cu = _make_user()  # video.user_id とは別人
    with pytest.raises(HTTPException) as exc:
        job_service.retry_job(
            db=db,
            job_id=job.id,
            current_user=cu,
            background_tasks=bt,
            job_repo=_JobRepositoryStub(job),
            video_repo=_VideoRepositoryStub(video),
        )
    assert exc.value.status_code == 403


def test_retry_job_409_when_not_failed() -> None:
    db, bt = MagicMock(), MagicMock()
    job = _make_job(status=JobStatus.processing)
    video = _make_video()
    cu = _make_user(id=video.user_id)
    with pytest.raises(HTTPException) as exc:
        job_service.retry_job(
            db=db,
            job_id=job.id,
            current_user=cu,
            background_tasks=bt,
            job_repo=_JobRepositoryStub(job),
            video_repo=_VideoRepositoryStub(video),
        )
    assert exc.value.status_code == 409


def test_retry_job_resets_and_rekicks_on_success() -> None:
    db, bt = MagicMock(), MagicMock()
    job = _make_job(status=JobStatus.failed)
    video = _make_video()
    cu = _make_user(id=video.user_id)
    job_repo = _JobRepositoryStub(job)
    video_repo = _VideoRepositoryStub(video)
    job_service.retry_job(
        db=db,
        job_id=job.id,
        current_user=cu,
        background_tasks=bt,
        job_repo=job_repo,
        video_repo=video_repo,
    )

    assert job_repo.reset_calls == [job.id]
    assert video_repo.update_status_calls == [(db, video.id, VideoStatus.queued)]
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


# --- process_fail_job -------------------------------------------------------


def test_process_fail_job_delegates_to_handle_ml_failure() -> None:
    """ML からの失敗通知はそのまま handle_ml_failure に渡り、リトライ判定に乗る。"""
    job_id = uuid.uuid4()
    with (
        patch("app.services.job_service.SessionLocal") as sl,
        patch("app.services.job_service.handle_ml_failure") as handle,
    ):
        sl.return_value.__enter__.return_value = MagicMock()
        job_service.process_fail_job(job_id, "HTTPStatusError: 403 Forbidden")

    handle.assert_called_once()
    assert handle.call_args.args[1] == job_id
    # 原因を追えるよう ML 側のエラー文字列を error_message に残す
    assert "HTTPStatusError: 403 Forbidden" in handle.call_args.args[2]


# --- complete_job -----------------------------------------------------------


def test_complete_job_returns_when_job_missing() -> None:
    db = MagicMock()
    job_service.complete_job(
        db,
        uuid.uuid4(),
        [],
        job_repo=_JobRepositoryStub(None),
        # ジョブが無い時点で打ち切るので、動画の照会には進まない（進めば落ちる）
        video_repo=FakeVideoRepository(),
    )


def test_complete_job_returns_when_video_missing() -> None:
    db = MagicMock()
    job = _make_job()
    job_repo = _JobRepositoryStub(job)
    job_service.complete_job(
        db,
        job.id,
        [],
        job_repo=job_repo,
        video_repo=_VideoRepositoryStub(None),
    )
    assert job_repo.update_status_calls == []


def test_complete_job_with_empty_clips_sets_ready() -> None:
    """clips が空でも job=completed・video=ready になり、編集通知メールを送る。

    出力動画は書き出し操作時に作るので、ここでは clip も作らず FFmpeg も呼ばない。
    """
    db = MagicMock()
    job = _make_job()
    video = _make_video()
    user = _make_user(id=video.user_id)
    job_repo = _JobRepositoryStub(job)
    video_repo = _VideoRepositoryStub(video)
    clip_repo = _ClipRepositoryStub()
    with patch(
        "app.services.job_service.send_analysis_complete_email", return_value=True
    ) as send_mail:
        job_service.complete_job(
            db,
            job.id,
            [],
            job_repo=job_repo,
            video_repo=video_repo,
            clip_repo=clip_repo,
            user_repo=_UserRepositoryStub(user),
            notification_log_repo=_NotificationLogRepositoryStub(),
        )

    assert clip_repo.created == []
    assert job_repo.update_status_calls[0]["status"] == JobStatus.completed
    assert video_repo.update_status_calls == [(db, job.video_id, VideoStatus.ready)]
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
    job_repo = _JobRepositoryStub(job)
    video_repo = _VideoRepositoryStub(video)
    clip_repo = _ClipRepositoryStub()
    with patch(
        "app.services.job_service.send_analysis_complete_email", return_value=True
    ) as send_mail:
        job_service.complete_job(
            db,
            job.id,
            clips,
            job_repo=job_repo,
            video_repo=video_repo,
            clip_repo=clip_repo,
            user_repo=_UserRepositoryStub(user),
            notification_log_repo=_NotificationLogRepositoryStub(),
        )

    assert len(clip_repo.created) == 2
    assert clip_repo.created[0]["sort_order"] == 0
    assert clip_repo.created[1]["sort_order"] == 1
    assert clip_repo.created[0]["job_id"] == job.id
    assert job_repo.update_status_calls[0]["status"] == JobStatus.completed
    assert video_repo.update_status_calls == [(db, job.video_id, VideoStatus.ready)]
    assert send_mail.call_args.kwargs["clip_count"] == 2
