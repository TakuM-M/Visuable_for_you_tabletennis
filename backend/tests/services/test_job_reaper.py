"""job_reaper の mock テスト（cleanup_expired_videos 以外）。

cleanup_expired_videos は test_retention_cleanup.py でカバー済み。ここでは
タイムアウト回収・自動リトライ・統計ログ・tmp 掃除を検証する。
"""

import os
import time
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.services import job_reaper


# --- reap_timeouts ----------------------------------------------------------


def test_reap_timeouts_handles_each_timed_out_job() -> None:
    jobs = [
        SimpleNamespace(id=uuid.uuid4(), started_at=datetime.now(timezone.utc))
        for _ in range(2)
    ]
    with (
        patch("app.services.job_reaper.SessionLocal") as sl,
        patch("app.services.job_reaper.job_repo.get_timed_out_jobs", return_value=jobs),
        patch("app.services.job_reaper.job_service.handle_ml_failure") as handle,
        patch("app.services.job_reaper.settings.job_timeout_hours", 24.0),
    ):
        sl.return_value.__enter__.return_value = MagicMock()
        job_reaper.reap_timeouts()

    assert handle.call_count == 2
    assert all(call.args[2] == "ML タイムアウト" for call in handle.call_args_list)


def test_reap_timeouts_noop_when_no_jobs() -> None:
    with (
        patch("app.services.job_reaper.SessionLocal") as sl,
        patch("app.services.job_reaper.job_repo.get_timed_out_jobs", return_value=[]),
        patch("app.services.job_reaper.job_service.handle_ml_failure") as handle,
        patch("app.services.job_reaper.settings.job_timeout_hours", 24.0),
    ):
        sl.return_value.__enter__.return_value = MagicMock()
        job_reaper.reap_timeouts()

    handle.assert_not_called()


# --- reconcile_runpod_jobs --------------------------------------------------
#   コールバックが届かないまま GPU 側が終了したケースを、job_timeout_hours を
#   待たずに検知する。RunPod の状態ごとの分岐が肝。


def _make_runpod_job(**kw) -> SimpleNamespace:
    defaults = dict(id=uuid.uuid4(), runpod_job_id="rp-1")
    defaults.update(kw)
    return SimpleNamespace(**defaults)


def _run_reconcile(status: str | None, jobs: list | None = None):
    """RunPod の状態を固定して reconcile_runpod_jobs を走らせ、handle mock を返す"""
    targets = [_make_runpod_job()] if jobs is None else jobs
    with (
        patch("app.services.video_service.USE_RUNPOD", True),
        patch("app.services.job_reaper.SessionLocal") as sl,
        patch(
            "app.services.job_reaper.job_repo.get_running_runpod_jobs",
            return_value=targets,
        ),
        patch(
            "app.services.job_reaper.runpod_service.get_job_status", return_value=status
        ),
        patch("app.services.job_reaper.job_service.handle_ml_failure") as handle,
    ):
        sl.return_value.__enter__.return_value = MagicMock()
        job_reaper.reconcile_runpod_jobs()
    return handle


def test_reconcile_skips_when_runpod_disabled() -> None:
    """dev（ml-mock 経路）では RunPod に問い合わせない"""
    with (
        patch("app.services.video_service.USE_RUNPOD", False),
        patch("app.services.job_reaper.job_repo.get_running_runpod_jobs") as get_jobs,
    ):
        job_reaper.reconcile_runpod_jobs()

    get_jobs.assert_not_called()


def test_reconcile_fails_job_when_runpod_dead() -> None:
    """FAILED / TIMED_OUT / CANCELLED は即座に失敗確定させる（(b) の本命）"""
    for status in ("FAILED", "TIMED_OUT", "CANCELLED"):
        handle = _run_reconcile(status)
        handle.assert_called_once()
        assert status in handle.call_args.args[2]


def test_reconcile_keeps_active_job_untouched() -> None:
    """まだ動いているジョブには手を出さない"""
    for status in ("IN_QUEUE", "IN_PROGRESS", "RUNNING"):
        handle = _run_reconcile(status)
        handle.assert_not_called()


def test_reconcile_keeps_job_when_status_unknown() -> None:
    """問い合わせ失敗（None）で稼働中のジョブを殺さない"""
    handle = _run_reconcile(None)
    handle.assert_not_called()


def test_reconcile_ignores_unrecognized_status() -> None:
    """未知の状態は判断を保留する（reap_timeouts が最終的に拾う）"""
    handle = _run_reconcile("SOMETHING_NEW")
    handle.assert_not_called()


def test_reconcile_fails_completed_job_without_callback() -> None:
    """COMPLETED でもコールバック未達なら失敗扱いにしてリトライさせる。

    エラーメッセージで「GPU は動いたがコールバックが届かなかった」と分かるようにする。
    """
    handle = _run_reconcile("COMPLETED")
    handle.assert_called_once()
    assert "コールバック未達" in handle.call_args.args[2]


def test_reconcile_handles_multiple_jobs() -> None:
    jobs = [_make_runpod_job(), _make_runpod_job()]
    handle = _run_reconcile("FAILED", jobs=jobs)
    assert handle.call_count == 2


# --- dispatch_retries -------------------------------------------------------


def test_dispatch_retries_prepares_and_rekicks() -> None:
    video = SimpleNamespace(storage_path="videos/a.mp4")
    job = SimpleNamespace(id=uuid.uuid4(), video_id=uuid.uuid4(), video=video)
    with (
        patch("app.services.job_reaper.SessionLocal") as sl,
        patch(
            "app.services.job_reaper.job_repo.get_jobs_ready_for_retry",
            return_value=[job],
        ),
        patch("app.services.job_reaper.job_repo.prepare_for_auto_retry") as prep,
        patch("app.services.job_reaper.call_ml_service") as call,
        patch("app.services.job_reaper.settings.job_max_retries", 2),
    ):
        sl.return_value.__enter__.return_value = MagicMock()
        job_reaper.dispatch_retries()

    assert prep.call_args.args[1] == job.id
    call.assert_called_once_with("videos/a.mp4", str(job.id), str(job.video_id))


def test_dispatch_retries_noop_when_no_jobs() -> None:
    with (
        patch("app.services.job_reaper.SessionLocal") as sl,
        patch(
            "app.services.job_reaper.job_repo.get_jobs_ready_for_retry", return_value=[]
        ),
        patch("app.services.job_reaper.job_repo.prepare_for_auto_retry"),
        patch("app.services.job_reaper.call_ml_service") as call,
        patch("app.services.job_reaper.settings.job_max_retries", 2),
    ):
        sl.return_value.__enter__.return_value = MagicMock()
        job_reaper.dispatch_retries()

    call.assert_not_called()


# --- log_storage_metrics ----------------------------------------------------


def test_log_storage_metrics_logs_on_success() -> None:
    metrics = SimpleNamespace(r2_total_bytes=1, r2_object_count=2, db_video_count=3)
    with (
        patch("app.services.job_reaper.SessionLocal") as sl,
        patch(
            "app.services.job_reaper.metrics_service.collect_storage_metrics",
            return_value=metrics,
        ) as collect,
    ):
        sl.return_value.__enter__.return_value = MagicMock()
        job_reaper.log_storage_metrics()

    collect.assert_called_once()


def test_log_storage_metrics_swallows_errors() -> None:
    with (
        patch("app.services.job_reaper.SessionLocal") as sl,
        patch(
            "app.services.job_reaper.metrics_service.collect_storage_metrics",
            side_effect=RuntimeError("R2 down"),
        ),
    ):
        sl.return_value.__enter__.return_value = MagicMock()
        # 例外が外に伝播しないこと
        job_reaper.log_storage_metrics()


# --- clean_tmp_dir ----------------------------------------------------------


def _make_old(path) -> None:
    old_time = time.time() - 100 * 3600
    os.utime(path, (old_time, old_time))


def test_clean_tmp_dir_removes_old_files_and_keeps_new(tmp_path) -> None:
    old_file = tmp_path / "old.txt"
    old_file.write_text("x")
    new_file = tmp_path / "new.txt"
    new_file.write_text("y")
    _make_old(old_file)

    with (
        patch("app.services.job_reaper.LOCAL_TMP_DIR", tmp_path),
        patch("app.services.job_reaper.settings.tmp_retention_hours", 24.0),
    ):
        job_reaper.clean_tmp_dir()

    assert not old_file.exists()
    assert new_file.exists()


def test_clean_tmp_dir_removes_old_directories(tmp_path) -> None:
    old_dir = tmp_path / "olddir"
    old_dir.mkdir()
    (old_dir / "f").write_text("x")
    _make_old(old_dir)

    with (
        patch("app.services.job_reaper.LOCAL_TMP_DIR", tmp_path),
        patch("app.services.job_reaper.settings.tmp_retention_hours", 24.0),
    ):
        job_reaper.clean_tmp_dir()

    assert not old_dir.exists()


def test_clean_tmp_dir_noop_when_dir_missing(tmp_path) -> None:
    missing = tmp_path / "does-not-exist"
    with patch("app.services.job_reaper.LOCAL_TMP_DIR", missing):
        # 例外が出ないこと
        job_reaper.clean_tmp_dir()
