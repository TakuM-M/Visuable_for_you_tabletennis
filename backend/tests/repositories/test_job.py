"""job repository の実 DB テスト。

このリポジトリはプロジェクトで最も複雑な状態遷移を持つ:
  - queued → processing → completed/failed
  - failed → (自動 or 手動リトライ) → queued
  - failed → 最終失敗 (next_retry_at=None)

特にリトライ機構 (mark_failed / prepare_for_auto_retry / reset_for_manual_retry /
get_jobs_ready_for_retry / get_timed_out_jobs) を網羅的にテストする。

`video` fixture は conftest.py で定義済み (db → user → video の連鎖)。
"""

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.exc import IntegrityError

from app.models.job import JobStatus
from app.repositories.job import job_repository as job_repo


# ----------------------------------------------------------------------
# create
# ----------------------------------------------------------------------


def test_create_sets_defaults(db, video):
    """create() 直後は queued / retry_count=0 / 各タイムスタンプ None"""
    job = job_repo.create(db, video_id=video.id)
    assert isinstance(job.id, uuid.UUID)
    assert job.video_id == video.id
    assert job.status == JobStatus.queued
    assert job.retry_count == 0
    assert job.started_at is None
    assert job.completed_at is None
    assert job.next_retry_at is None
    assert job.error_message is None
    assert job.created_at is not None  # server_default


def test_create_with_unknown_video_id_violates_fk(db):
    """存在しない video_id では FK 制約違反 (jobs.video_id → videos.id)"""
    with pytest.raises(IntegrityError):
        job_repo.create(db, video_id=uuid.uuid4())
    db.rollback()


# ----------------------------------------------------------------------
# get_by_id / get_by_video_id
# ----------------------------------------------------------------------


def test_get_by_id_returns_created_job(db, video):
    created = job_repo.create(db, video_id=video.id)
    fetched = job_repo.get_by_id(db, created.id)
    assert fetched is not None
    assert fetched.id == created.id


def test_get_by_video_id_returns_all_jobs_for_that_video(db, video):
    """同じ動画に対する複数ジョブ (例: リトライで再ディスパッチ) を全部返す"""
    j1 = job_repo.create(db, video_id=video.id)
    j2 = job_repo.create(db, video_id=video.id)
    jobs = job_repo.get_by_video_id(db, video.id)
    assert {j.id for j in jobs} == {j1.id, j2.id}


# ----------------------------------------------------------------------
# update_status (部分更新の挙動が肝)
# ----------------------------------------------------------------------


def test_update_status_changes_only_status_when_other_args_omitted(db, video):
    """status だけ渡せば status だけ書き換わる"""
    job = job_repo.create(db, video_id=video.id)
    updated = job_repo.update_status(db, job.id, JobStatus.processing)
    assert updated is not None
    assert updated.status == JobStatus.processing
    assert updated.started_at is None  # 渡してないので変わらない


def test_update_status_with_started_at_records_timestamp(db, video):
    job = job_repo.create(db, video_id=video.id)
    now = datetime.now(timezone.utc)
    updated = job_repo.update_status(db, job.id, JobStatus.processing, started_at=now)
    assert updated.started_at is not None


def test_update_status_partial_update_preserves_existing_fields(db, video):
    """以前セットした started_at は、次の update_status で省略しても残る。

    これは update_status の if started_at is not None: の挙動で、
    後続更新 (例: completed への遷移時) に started_at を再指定しなくていいようにしている。
    """
    job = job_repo.create(db, video_id=video.id)
    started = datetime.now(timezone.utc)
    job_repo.update_status(db, job.id, JobStatus.processing, started_at=started)
    # 次の更新では started_at を渡さない
    job_repo.update_status(db, job.id, JobStatus.completed)
    refreshed = job_repo.get_by_id(db, job.id)
    assert refreshed.status == JobStatus.completed
    assert refreshed.started_at is not None  # 保持されている


def test_update_status_returns_none_for_unknown_id(db):
    assert job_repo.update_status(db, uuid.uuid4(), JobStatus.processing) is None


# ----------------------------------------------------------------------
# mark_failed (リトライ枠付き失敗 vs 最終失敗)
# ----------------------------------------------------------------------


def test_mark_failed_with_next_retry_at_sets_retry_window(db, video):
    """next_retry_at を渡すと自動リトライ対象となる失敗状態に遷移"""
    job = job_repo.create(db, video_id=video.id)
    retry_at = datetime.now(timezone.utc) + timedelta(seconds=60)
    result = job_repo.mark_failed(db, job.id, "ML タイムアウト", retry_at)
    assert result.status == JobStatus.failed
    assert result.error_message == "ML タイムアウト"
    assert result.next_retry_at is not None


def test_mark_failed_with_none_next_retry_at_marks_terminal_failure(db, video):
    """next_retry_at=None は「これ以上リトライしない」最終失敗を表す"""
    job = job_repo.create(db, video_id=video.id)
    result = job_repo.mark_failed(db, job.id, "リトライ上限超過", None)
    assert result.status == JobStatus.failed
    assert result.next_retry_at is None


# ----------------------------------------------------------------------
# get_timed_out_jobs (タイムアウト検出)
# ----------------------------------------------------------------------


def test_get_timed_out_jobs_returns_processing_jobs_started_before_threshold(db, video):
    """processing 中で started_at が threshold より古いジョブを拾う"""
    job = job_repo.create(db, video_id=video.id)
    # processing にしつつ started_at を 25 時間前に
    job.status = JobStatus.processing
    job.started_at = datetime.now(timezone.utc) - timedelta(hours=25)
    db.commit()

    threshold = datetime.now(timezone.utc) - timedelta(hours=24)
    result = job_repo.get_timed_out_jobs(db, threshold)
    assert len(result) == 1
    assert result[0].id == job.id


def test_get_timed_out_jobs_excludes_jobs_with_null_started_at(db, video):
    """started_at が None のジョブ (= 一度も処理が始まっていない queued) は除外される"""
    job_repo.create(db, video_id=video.id)  # default で queued / started_at=None
    threshold = datetime.now(timezone.utc) - timedelta(hours=24)
    assert job_repo.get_timed_out_jobs(db, threshold) == []


def test_get_timed_out_jobs_excludes_completed_jobs(db, video):
    """completed や failed は除外される (status フィルタ)"""
    job = job_repo.create(db, video_id=video.id)
    job.status = JobStatus.completed
    job.started_at = datetime.now(timezone.utc) - timedelta(hours=25)
    db.commit()

    threshold = datetime.now(timezone.utc) - timedelta(hours=24)
    assert job_repo.get_timed_out_jobs(db, threshold) == []


def test_get_queued_started_null_jobs_returns_only_queued_with_null_started_at(
    db, video
):
    """status : queued started_at : null のジョブを取得する"""
    job1 = job_repo.create(db, video_id=video.id)  # queued / started_at=None
    job2 = job_repo.create(db, video_id=video.id)  # queued / started_at!=None
    job3 = job_repo.create(db, video_id=video.id)  # processing / started_at=None
    job2.started_at = datetime.now(timezone.utc)
    job3.status = JobStatus.processing
    db.commit()

    result = job_repo.get_queued_started_null_jobs(db)
    assert len(result) == 1
    assert result[0].id == job1.id


# ----------------------------------------------------------------------
# get_jobs_ready_for_retry (自動リトライ対象の探索)
# ----------------------------------------------------------------------


def test_get_jobs_ready_for_retry_returns_failed_jobs_past_retry_time(db, video):
    """failed かつ next_retry_at が過去のジョブを拾う"""
    job = job_repo.create(db, video_id=video.id)
    job.status = JobStatus.failed
    job.next_retry_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    db.commit()

    result = job_repo.get_jobs_ready_for_retry(
        db, now=datetime.now(timezone.utc), max_retries=3
    )
    assert len(result) == 1
    assert result[0].id == job.id


def test_get_jobs_ready_for_retry_excludes_jobs_at_max_retries(db, video):
    """retry_count >= max_retries のジョブは拾わない (上限到達)"""
    job = job_repo.create(db, video_id=video.id)
    job.status = JobStatus.failed
    job.retry_count = 3  # max_retries と同じ → 除外される
    job.next_retry_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    db.commit()

    result = job_repo.get_jobs_ready_for_retry(
        db, now=datetime.now(timezone.utc), max_retries=3
    )
    assert result == []


def test_get_jobs_ready_for_retry_excludes_future_retry_time(db, video):
    """next_retry_at がまだ未来のジョブは拾わない (バックオフ待機中)"""
    job = job_repo.create(db, video_id=video.id)
    job.status = JobStatus.failed
    job.next_retry_at = datetime.now(timezone.utc) + timedelta(hours=1)
    db.commit()

    result = job_repo.get_jobs_ready_for_retry(
        db, now=datetime.now(timezone.utc), max_retries=3
    )
    assert result == []


# ----------------------------------------------------------------------
# prepare_for_auto_retry / reset_for_manual_retry
# ----------------------------------------------------------------------


def test_prepare_for_auto_retry_increments_count_and_resets_state(db, video):
    """自動リトライ準備: retry_count++, status=queued, タイムスタンプ・エラー全クリア"""
    # 失敗状態をセットアップ
    job = job_repo.create(db, video_id=video.id)
    job.status = JobStatus.failed
    job.started_at = datetime.now(timezone.utc) - timedelta(minutes=5)
    job.completed_at = datetime.now(timezone.utc)
    job.error_message = "ML failed"
    job.next_retry_at = datetime.now(timezone.utc)
    db.commit()
    assert job.retry_count == 0  # スタート時点

    result = job_repo.prepare_for_auto_retry(db, job.id)
    assert result.status == JobStatus.queued
    assert result.retry_count == 1  # インクリメント
    assert result.started_at is None
    assert result.completed_at is None
    assert result.next_retry_at is None
    assert result.error_message is None


def test_reset_for_manual_retry_zeros_count_regardless_of_previous(db, video):
    """手動リトライ: retry_count を 0 に戻し、リトライ枠を完全リセット"""
    job = job_repo.create(db, video_id=video.id)
    job.status = JobStatus.failed
    job.retry_count = 2  # 既に何度かリトライ済みでも…
    job.error_message = "boom"
    job.next_retry_at = datetime.now(timezone.utc)
    job.started_at = datetime.now(timezone.utc)
    db.commit()

    result = job_repo.reset_for_manual_retry(db, job.id)
    assert result.status == JobStatus.queued
    assert result.retry_count == 0  # ←ここが auto_retry との違い
    assert result.next_retry_at is None
    assert result.started_at is None
    assert result.error_message is None


# ----------------------------------------------------------------------
# delete_by_video_id
# ----------------------------------------------------------------------


def test_delete_by_video_id_removes_all_jobs_and_returns_count(db, video):
    j1 = job_repo.create(db, video_id=video.id)
    j2 = job_repo.create(db, video_id=video.id)

    count = job_repo.delete_by_video_id(db, video.id)
    assert count == 2
    assert job_repo.get_by_id(db, j1.id) is None
    assert job_repo.get_by_id(db, j2.id) is None


def test_delete_by_video_id_returns_zero_when_no_jobs(db, video):
    assert job_repo.delete_by_video_id(db, video.id) == 0


# ----------------------------------------------------------------------
# get_latest_by_video_id（ユーザー編集 clip に流用する最新ジョブの取得）
# ----------------------------------------------------------------------


def test_get_latest_by_video_id_returns_most_recent(db, video):
    """created_at が最も新しいジョブを返す"""
    old = job_repo.create(db, video_id=video.id)
    old.created_at = datetime.now(timezone.utc) - timedelta(hours=1)
    db.commit()
    latest = job_repo.create(db, video_id=video.id)

    result = job_repo.get_latest_by_video_id(db, video.id)
    assert result is not None
    assert result.id == latest.id


def test_get_latest_by_video_id_returns_none_when_no_jobs(db, video):
    assert job_repo.get_latest_by_video_id(db, video.id) is None
