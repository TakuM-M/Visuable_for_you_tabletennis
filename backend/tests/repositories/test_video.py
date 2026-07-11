"""video repository の実 DB テスト。

user に比べて学ぶことが多い:
  - 外部キー (videos.user_id → users.id) があるので、テスト前提として user が必要
  - 複数ユーザーが入り混じった状態での絞り込みクエリの正しさを見る
  - created_at による時系列フィルタ (get_expired) を扱う
  - FK 違反が DB レベルで弾かれることを確認する

`user` fixture は conftest.py で定義済み。`db` も同様。
"""
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.exc import IntegrityError

from app.models.job import JobStatus
from app.models.video import VideoStatus
from app.repositories import job as job_repo
from app.repositories import user as user_repo
from app.repositories import video as video_repo


# ----------------------------------------------------------------------
# create
# ----------------------------------------------------------------------

def test_create_sets_defaults(db, user):
    """create() は status=uploaded, output_path=None, duration=None を既定値にする"""
    video = video_repo.create(
        db=db,
        user_id=user.id,
        title="練習試合",
        storage_path="videos/abc.mp4",
    )
    assert isinstance(video.id, uuid.UUID)
    assert video.user_id == user.id
    assert video.title == "練習試合"
    assert video.storage_path == "videos/abc.mp4"
    assert video.status == VideoStatus.uploaded  # モデルの default
    assert video.output_path is None             # nullable=True
    assert video.duration is None                # 引数省略時
    assert video.created_at is not None          # server_default=now()


def test_create_with_duration_records_it(db, user):
    """duration を明示的に渡せば保存される"""
    video = video_repo.create(
        db=db,
        user_id=user.id,
        title="試合 A",
        storage_path="videos/a.mp4",
        duration=123.5,
    )
    assert video.duration == 123.5


def test_create_with_unknown_user_id_violates_fk(db):
    """存在しない user_id で video を作ろうとすると FK 制約違反になる"""
    with pytest.raises(IntegrityError):
        video_repo.create(
            db=db,
            user_id=uuid.uuid4(),  # 誰もいない
            title="orphan",
            storage_path="videos/x.mp4",
        )
    db.rollback()


# ----------------------------------------------------------------------
# get_by_id / get_by_user_id / count_by_user_id
# ----------------------------------------------------------------------

def test_get_by_id_returns_created_video(db, user):
    created = video_repo.create(
        db, user_id=user.id, title="t", storage_path="p"
    )
    fetched = video_repo.get_by_id(db, created.id)
    assert fetched is not None
    assert fetched.id == created.id


def test_get_by_id_returns_none_when_not_found(db):
    assert video_repo.get_by_id(db, uuid.uuid4()) is None


def test_get_by_user_id_returns_only_that_users_videos(db):
    """複数ユーザーがいる状況で、指定 user の動画だけが返る"""
    alice = user_repo.create(db, email="a@x.com", password_hash="x", display_name="A")
    bob = user_repo.create(db, email="b@x.com", password_hash="x", display_name="B")

    video_repo.create(db, user_id=alice.id, title="alice-1", storage_path="p1")
    video_repo.create(db, user_id=alice.id, title="alice-2", storage_path="p2")
    video_repo.create(db, user_id=bob.id, title="bob-1", storage_path="p3")

    alice_videos = video_repo.get_by_user_id(db, alice.id)
    assert len(alice_videos) == 2
    assert {v.title for v in alice_videos} == {"alice-1", "alice-2"}

    bob_videos = video_repo.get_by_user_id(db, bob.id)
    assert len(bob_videos) == 1
    assert bob_videos[0].title == "bob-1"


def test_count_by_user_id_counts_only_owners_videos(db, user):
    """count_by_user_id は指定 user の本数だけ数える（他人の本数は混ざらない）"""
    other = user_repo.create(
        db, email="other@x.com", password_hash="x", display_name="Other"
    )
    # owner 3 本、other 2 本
    for i in range(3):
        video_repo.create(db, user_id=user.id, title=f"u-{i}", storage_path=f"u{i}")
    for i in range(2):
        video_repo.create(db, user_id=other.id, title=f"o-{i}", storage_path=f"o{i}")

    assert video_repo.count_by_user_id(db, user.id) == 3
    assert video_repo.count_by_user_id(db, other.id) == 2


def test_count_by_user_id_returns_zero_for_user_with_no_videos(db, user):
    assert video_repo.count_by_user_id(db, user.id) == 0


# ----------------------------------------------------------------------
# get_expired (created_at による時系列フィルタ)
# ----------------------------------------------------------------------

def test_get_expired_returns_only_videos_before_threshold(db, user):
    """threshold より古い created_at を持つ動画だけ返る。

    created_at は server_default=now() で DB が埋めるため、
    テストでは作った後に created_at を上書きして古い動画を演出する。
    """
    old = video_repo.create(db, user_id=user.id, title="old", storage_path="o")
    new = video_repo.create(db, user_id=user.id, title="new", storage_path="n")

    # old だけ created_at を 10 日前にずらす
    old.created_at = datetime.now(timezone.utc) - timedelta(days=10)
    db.commit()

    threshold = datetime.now(timezone.utc) - timedelta(days=7)
    expired = video_repo.get_expired(db, threshold)
    assert len(expired) == 1
    assert expired[0].id == old.id


# ----------------------------------------------------------------------
# update_status / update_output_path / update_duration
# ----------------------------------------------------------------------

def test_update_status_transitions_through_lifecycle(db, user):
    """uploaded → queued → processing → completed の状態遷移を反映できる"""
    video = video_repo.create(db, user_id=user.id, title="t", storage_path="p")
    assert video.status == VideoStatus.uploaded

    for next_status in [
        VideoStatus.queued,
        VideoStatus.processing,
        VideoStatus.completed,
    ]:
        updated = video_repo.update_status(db, video.id, next_status)
        assert updated is not None
        assert updated.status == next_status


def test_update_status_returns_none_for_unknown_id(db):
    """対象 video が無ければ None が返る（例外ではない）"""
    assert video_repo.update_status(db, uuid.uuid4(), VideoStatus.processing) is None


def test_update_output_path_sets_path(db, user):
    """ML 処理完了後の output_path を保存できる"""
    video = video_repo.create(db, user_id=user.id, title="t", storage_path="p")
    assert video.output_path is None

    updated = video_repo.update_output_path(db, video.id, "videos/out.mp4")
    assert updated is not None
    assert updated.output_path == "videos/out.mp4"


def test_update_duration_sets_duration(db, user):
    video = video_repo.create(db, user_id=user.id, title="t", storage_path="p")
    assert video.duration is None

    updated = video_repo.update_duration(db, video.id, 60.0)
    assert updated is not None
    assert updated.duration == 60.0


# ----------------------------------------------------------------------
# delete
# ----------------------------------------------------------------------

def test_delete_removes_video_and_returns_true(db, user):
    video = video_repo.create(db, user_id=user.id, title="t", storage_path="p")
    assert video_repo.delete(db, video.id) is True
    assert video_repo.get_by_id(db, video.id) is None


def test_delete_returns_false_when_not_found(db):
    assert video_repo.delete(db, uuid.uuid4()) is False


# ----------------------------------------------------------------------
# source_duration（元動画長）
# ----------------------------------------------------------------------

def test_create_source_duration_defaults_none(db, user):
    """source_duration は引数省略時 None"""
    video = video_repo.create(db, user_id=user.id, title="t", storage_path="p")
    assert video.source_duration is None


def test_create_with_source_duration_records_it(db, user):
    """source_duration を渡せば保存される（duration とは独立）"""
    video = video_repo.create(
        db=db, user_id=user.id, title="t", storage_path="p",
        duration=10.0, source_duration=88.0,
    )
    assert video.source_duration == 88.0
    assert video.duration == 10.0


# ----------------------------------------------------------------------
# get_processing_without_running_job（中断された書き出しの検出）
# ----------------------------------------------------------------------

def test_get_processing_without_running_job_detects_interrupted_export(db, user):
    """processing かつ実行中 job なし（＝中断された書き出し）の動画だけを返す"""
    # 中断された書き出し: job は completed 済みで video だけ processing
    interrupted = video_repo.create(db, user_id=user.id, title="中断", storage_path="videos/i.mp4")
    job_done = job_repo.create(db, video_id=interrupted.id)
    job_repo.update_status(db, job_done.id, JobStatus.completed)
    video_repo.update_status(db, interrupted.id, VideoStatus.processing)

    # ML 解析中: processing だが実行中 job があるので対象外
    analyzing = video_repo.create(db, user_id=user.id, title="解析中", storage_path="videos/a.mp4")
    job_running = job_repo.create(db, video_id=analyzing.id)
    job_repo.update_status(db, job_running.id, JobStatus.processing)
    video_repo.update_status(db, analyzing.id, VideoStatus.processing)

    # processing 以外は job が無くても対象外
    ready = video_repo.create(db, user_id=user.id, title="ready", storage_path="videos/r.mp4")
    video_repo.update_status(db, ready.id, VideoStatus.ready)

    result = video_repo.get_processing_without_running_job(db)
    assert [v.id for v in result] == [interrupted.id]


def test_get_processing_without_running_job_empty_when_all_running(db, user):
    """実行中 job を伴う processing 動画しか無ければ空を返す"""
    video = video_repo.create(db, user_id=user.id, title="t", storage_path="p")
    job = job_repo.create(db, video_id=video.id)
    job_repo.update_status(db, job.id, JobStatus.queued)
    video_repo.update_status(db, video.id, VideoStatus.processing)

    assert video_repo.get_processing_without_running_job(db) == []
