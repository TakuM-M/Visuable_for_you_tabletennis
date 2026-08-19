"""clip リポジトリの実 DB テスト。

video / job への FK を持つため、conftest の video・job fixture を使って検証する。
"""

import uuid

import pytest
from sqlalchemy.exc import IntegrityError

from app.repositories import clip as clip_repo
from app.repositories.job import job_repository as job_repo
from app.repositories.video import video_repository as video_repo


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
        db=db,
        video_id=video.id,
        job_id=job.id,
        start_time=0.0,
        end_time=5.0,
        storage_path="a",
    )
    # 別の動画 + その動画のジョブ・クリップ
    other_video = video_repo.create(
        db=db,
        user_id=user.id,
        title="other",
        storage_path="videos/other.mp4",
    )
    other_job = job_repo.create(db=db, video_id=other_video.id)
    clip_repo.create(
        db=db,
        video_id=other_video.id,
        job_id=other_job.id,
        start_time=0.0,
        end_time=5.0,
        storage_path="b",
    )

    result = clip_repo.get_by_video_id(db, video.id)
    assert len(result) == 1
    assert result[0].storage_path == "a"


def test_get_by_video_id_returns_empty_when_none(db, video):
    assert clip_repo.get_by_video_id(db, video.id) == []


def test_get_by_job_id_returns_only_that_jobs_clips(db, video, job):
    """同じ動画でもジョブが違えば絞り込まれること"""
    clip_repo.create(
        db=db,
        video_id=video.id,
        job_id=job.id,
        start_time=0.0,
        end_time=5.0,
        storage_path="a",
    )
    other_job = job_repo.create(db=db, video_id=video.id)
    clip_repo.create(
        db=db,
        video_id=video.id,
        job_id=other_job.id,
        start_time=0.0,
        end_time=5.0,
        storage_path="b",
    )

    result = clip_repo.get_by_job_id(db, job.id)
    assert len(result) == 1
    assert result[0].storage_path == "a"


def test_delete_by_video_id_removes_all_and_returns_count(db, video, job):
    clip_repo.create(
        db=db,
        video_id=video.id,
        job_id=job.id,
        start_time=0.0,
        end_time=5.0,
        storage_path="a",
    )
    clip_repo.create(
        db=db,
        video_id=video.id,
        job_id=job.id,
        start_time=5.0,
        end_time=10.0,
        storage_path="b",
    )

    count = clip_repo.delete_by_video_id(db, video.id)
    assert count == 2
    assert clip_repo.get_by_video_id(db, video.id) == []


def test_delete_by_video_id_returns_zero_when_none(db, video):
    assert clip_repo.delete_by_video_id(db, video.id) == 0


# ----------------------------------------------------------------------
# sort_order / replace_for_video（ユーザー編集による一括置換）
# ----------------------------------------------------------------------


def test_create_records_sort_order(db, video, job):
    """create() に渡した sort_order が保存される"""
    clip = clip_repo.create(
        db=db,
        video_id=video.id,
        job_id=job.id,
        start_time=0.0,
        end_time=5.0,
        storage_path="",
        sort_order=3,
    )
    assert clip.sort_order == 3


def test_get_by_video_id_orders_by_sort_order(db, video, job):
    """get_by_video_id は sort_order 昇順で返す（連結順の保証）"""
    clip_repo.create(
        db=db,
        video_id=video.id,
        job_id=job.id,
        start_time=0.0,
        end_time=1.0,
        storage_path="c",
        sort_order=2,
    )
    clip_repo.create(
        db=db,
        video_id=video.id,
        job_id=job.id,
        start_time=0.0,
        end_time=1.0,
        storage_path="a",
        sort_order=0,
    )
    clip_repo.create(
        db=db,
        video_id=video.id,
        job_id=job.id,
        start_time=0.0,
        end_time=1.0,
        storage_path="b",
        sort_order=1,
    )

    result = clip_repo.get_by_video_id(db, video.id)
    assert [c.storage_path for c in result] == ["a", "b", "c"]


def test_replace_for_video_replaces_all_and_assigns_order(db, video, job):
    """既存 clip を全削除し、与えた配列順に sort_order を採番して作り直す"""
    clip_repo.create(
        db=db,
        video_id=video.id,
        job_id=job.id,
        start_time=0.0,
        end_time=1.0,
        storage_path="old1",
    )
    clip_repo.create(
        db=db,
        video_id=video.id,
        job_id=job.id,
        start_time=1.0,
        end_time=2.0,
        storage_path="old2",
    )

    new_clips = clip_repo.replace_for_video(
        db,
        video.id,
        job.id,
        [
            {"start_time": 0.0, "end_time": 3.0},
            {"start_time": 4.0, "end_time": 6.0},
            {"start_time": 7.0, "end_time": 9.0},
        ],
    )

    assert len(new_clips) == 3
    result = clip_repo.get_by_video_id(db, video.id)
    assert len(result) == 3
    assert [c.sort_order for c in result] == [0, 1, 2]
    assert [c.start_time for c in result] == [0.0, 4.0, 7.0]


def test_replace_for_video_with_empty_clears_all(db, video, job):
    """空配列で置換すると全 clip が消える（全削除のユースケース）"""
    clip_repo.create(
        db=db,
        video_id=video.id,
        job_id=job.id,
        start_time=0.0,
        end_time=1.0,
        storage_path="x",
    )

    new_clips = clip_repo.replace_for_video(db, video.id, job.id, [])
    assert new_clips == []
    assert clip_repo.get_by_video_id(db, video.id) == []
