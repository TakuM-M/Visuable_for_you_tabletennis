"""clip リポジトリの実 DB テスト。

video / job への FK を持つため、conftest の video・job fixture を使って検証する。
"""
import uuid

import pytest
from sqlalchemy.exc import IntegrityError

from app.repositories import clip as clip_repo
from app.repositories import job as job_repo
from app.repositories import video as video_repo


def test_create_sets_fields(db, video, job):
    clip = clip_repo.create(
        db=db,
        video_id=video.id,
        job_id=job.id,
        start_time=1.5,
        end_time=12.5,
        storage_path="clips/abc.mp4",
    )

    assert isinstance(clip.id, uuid.UUID)
    assert clip.video_id == video.id
    assert clip.job_id == job.id
    assert clip.start_time == 1.5
    assert clip.end_time == 12.5
    assert clip.storage_path == "clips/abc.mp4"
    assert clip.created_at is not None


def test_create_with_unknown_video_id_violates_fk(db, job):
    """job は実在するが video_id が存在しない場合 FK 違反になる"""
    with pytest.raises(IntegrityError):
        clip_repo.create(
            db=db,
            video_id=uuid.uuid4(),
            job_id=job.id,
            start_time=0.0,
            end_time=1.0,
            storage_path="",
        )
    db.rollback()


def test_create_with_unknown_job_id_violates_fk(db, video):
    """video は実在するが job_id が存在しない場合 FK 違反になる"""
    with pytest.raises(IntegrityError):
        clip_repo.create(
            db=db,
            video_id=video.id,
            job_id=uuid.uuid4(),
            start_time=0.0,
            end_time=1.0,
            storage_path="",
        )
    db.rollback()


def test_get_by_video_id_returns_only_that_videos_clips(db, user, video, job):
    """別の動画のクリップが混ざらないこと"""
    # video / job fixture に紐づくクリップ
    clip_repo.create(
        db=db, video_id=video.id, job_id=job.id,
        start_time=0.0, end_time=5.0, storage_path="a",
    )
    # 別の動画 + その動画のジョブ・クリップ
    other_video = video_repo.create(
        db=db, user_id=user.id, title="other", storage_path="videos/other.mp4",
    )
    other_job = job_repo.create(db=db, video_id=other_video.id)
    clip_repo.create(
        db=db, video_id=other_video.id, job_id=other_job.id,
        start_time=0.0, end_time=5.0, storage_path="b",
    )

    result = clip_repo.get_by_video_id(db, video.id)
    assert len(result) == 1
    assert result[0].storage_path == "a"


def test_get_by_video_id_returns_empty_when_none(db, video):
    assert clip_repo.get_by_video_id(db, video.id) == []


def test_get_by_job_id_returns_only_that_jobs_clips(db, video, job):
    """同じ動画でもジョブが違えば絞り込まれること"""
    clip_repo.create(
        db=db, video_id=video.id, job_id=job.id,
        start_time=0.0, end_time=5.0, storage_path="a",
    )
    other_job = job_repo.create(db=db, video_id=video.id)
    clip_repo.create(
        db=db, video_id=video.id, job_id=other_job.id,
        start_time=0.0, end_time=5.0, storage_path="b",
    )

    result = clip_repo.get_by_job_id(db, job.id)
    assert len(result) == 1
    assert result[0].storage_path == "a"


def test_delete_by_video_id_removes_all_and_returns_count(db, video, job):
    clip_repo.create(
        db=db, video_id=video.id, job_id=job.id,
        start_time=0.0, end_time=5.0, storage_path="a",
    )
    clip_repo.create(
        db=db, video_id=video.id, job_id=job.id,
        start_time=5.0, end_time=10.0, storage_path="b",
    )

    count = clip_repo.delete_by_video_id(db, video.id)
    assert count == 2
    assert clip_repo.get_by_video_id(db, video.id) == []


def test_delete_by_video_id_returns_zero_when_none(db, video):
    assert clip_repo.delete_by_video_id(db, video.id) == 0
