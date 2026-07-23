"""video_service の mock テスト（quota チェック以外）。

ffprobe / R2 / ML(httpx) / DB を差し替え、再生時間取得・ML ディスパッチ・
アップロード・チャンク結合・削除の各分岐を検証する。
ファイル I/O を伴う関数は LOCAL_TMP_DIR を pytest の tmp_path に差し替えて隔離する。
"""

import io
import json
import subprocess
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from app.models.job import JobStatus
from app.models.video import VideoStatus
from app.services import video_service


def _upload_file(content: bytes = b"abc", filename: str = "v.mp4") -> SimpleNamespace:
    return SimpleNamespace(filename=filename, file=io.BytesIO(content))


def _make_video(**kw) -> SimpleNamespace:
    defaults = dict(
        id=uuid.uuid4(),
        storage_path="videos/a.mp4",
        output_path=None,
    )
    defaults.update(kw)
    return SimpleNamespace(**defaults)


def _httpx_client_cm(chunks=(b"data",)) -> MagicMock:
    """httpx.Client() のコンテキストマネージャ + stream() をモックする"""
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.iter_bytes = MagicMock(return_value=list(chunks))
    stream_cm = MagicMock()
    stream_cm.__enter__.return_value = response
    stream_cm.__exit__.return_value = False
    client = MagicMock()
    client.stream.return_value = stream_cm
    client_cm = MagicMock()
    client_cm.__enter__.return_value = client
    client_cm.__exit__.return_value = False
    return client_cm


# --- _extract_duration ------------------------------------------------------


def test_extract_duration_returns_float_on_success() -> None:
    with patch(
        "app.services.video_service.subprocess.run",
        return_value=SimpleNamespace(stdout="12.5\n"),
    ):
        assert video_service._extract_duration("/tmp/x.mp4") == 12.5


def test_extract_duration_returns_none_on_failure() -> None:
    with patch(
        "app.services.video_service.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="ffprobe", timeout=30),
    ):
        assert video_service._extract_duration("/tmp/x.mp4") is None


# --- call_ml_service --------------------------------------------------------


def test_call_ml_service_posts_to_mock_when_not_runpod() -> None:
    job_id, video_id = str(uuid.uuid4()), str(uuid.uuid4())
    with (
        patch("app.services.video_service.SessionLocal") as sl,
        patch("app.services.video_service.job_repo.update_status"),
        patch("app.services.video_service.video_repo.update_status"),
        patch(
            "app.services.video_service.storage_service.generate_presigned_url",
            return_value="http://dl",
        ),
        patch("app.services.video_service.USE_RUNPOD", False),
        patch("app.services.video_service.httpx") as mock_httpx,
    ):
        sl.return_value.__enter__.return_value = MagicMock()
        client = mock_httpx.Client.return_value.__enter__.return_value
        video_service.call_ml_service("videos/a.mp4", job_id, video_id)

    posted_url = client.post.call_args.args[0]
    assert posted_url.endswith("/process")


def test_call_ml_service_posts_to_runpod_when_enabled() -> None:
    job_id, video_id = str(uuid.uuid4()), str(uuid.uuid4())
    with (
        patch("app.services.video_service.SessionLocal") as sl,
        patch("app.services.video_service.job_repo.update_status"),
        patch("app.services.video_service.video_repo.update_status"),
        patch(
            "app.services.video_service.storage_service.generate_presigned_url",
            return_value="http://dl",
        ),
        patch("app.services.video_service.USE_RUNPOD", True),
        patch("app.services.video_service.RUNPOD_ENDPOINT_ID", "ep"),
        patch("app.services.video_service.httpx") as mock_httpx,
    ):
        sl.return_value.__enter__.return_value = MagicMock()
        client = mock_httpx.Client.return_value.__enter__.return_value
        client.post.return_value.json.return_value = {"id": "rp-1"}
        video_service.call_ml_service("videos/a.mp4", job_id, video_id)

    posted_url = client.post.call_args.args[0]
    assert "api.runpod.ai" in posted_url


def test_call_ml_service_delegates_failure_to_handler() -> None:
    job_id, video_id = str(uuid.uuid4()), str(uuid.uuid4())
    with (
        patch("app.services.video_service.SessionLocal") as sl,
        patch("app.services.video_service.job_repo.update_status"),
        patch("app.services.video_service.video_repo.update_status"),
        patch(
            "app.services.video_service.storage_service.generate_presigned_url",
            side_effect=RuntimeError("R2 down"),
        ),
        patch("app.services.job_service.handle_ml_failure") as handle,
    ):
        sl.return_value.__enter__.return_value = MagicMock()
        video_service.call_ml_service("videos/a.mp4", job_id, video_id)

    handle.assert_called_once()
    assert "ML呼び出し失敗" in handle.call_args.args[2]


# --- _register_video_and_start_ml ------------------------------------------


def test_register_video_creates_records_and_schedules_ml() -> None:
    db, bt = MagicMock(), MagicMock()
    video = _make_video()
    job = SimpleNamespace(id=uuid.uuid4())
    with (
        patch("app.services.video_service.video_repo.create", return_value=video),
        patch("app.services.video_service.video_repo.update_status"),
        patch("app.services.video_service.job_repo.create", return_value=job),
    ):
        result = video_service._register_video_and_start_ml(
            db, uuid.uuid4(), "t", "videos/a.mp4", bt
        )

    assert result is video
    bt.add_task.assert_called_once()


# --- upload_video -----------------------------------------------------------


def test_upload_video_saves_uploads_and_registers(tmp_path) -> None:
    db, bt = MagicMock(), MagicMock()
    video = _make_video()
    with (
        patch("app.services.video_service.LOCAL_TMP_DIR", tmp_path),
        patch("app.services.video_service.video_repo.count_by_user_id", return_value=0),
        patch("app.services.video_service.settings.user_video_quota", 10),
        patch("app.services.video_service.storage_service.upload_file") as upload,
        patch("app.services.video_service._extract_duration", return_value=10.0),
        patch(
            "app.services.video_service._register_video_and_start_ml",
            return_value=video,
        ) as register,
    ):
        result = video_service.upload_video(db, uuid.uuid4(), "t", _upload_file(), bt)

    assert result is video
    upload.assert_called_once()
    register.assert_called_once()


def test_upload_video_raises_when_quota_exceeded(tmp_path) -> None:
    db, bt = MagicMock(), MagicMock()
    with (
        patch("app.services.video_service.LOCAL_TMP_DIR", tmp_path),
        patch(
            "app.services.video_service.video_repo.count_by_user_id", return_value=10
        ),
        patch("app.services.video_service.settings.user_video_quota", 10),
        patch("app.services.video_service.storage_service.upload_file") as upload,
    ):
        with pytest.raises(video_service.QuotaExceededError):
            video_service.upload_video(db, uuid.uuid4(), "t", _upload_file(), bt)

    upload.assert_not_called()


# --- init_chunk_upload / save_chunk / complete_chunk_upload -----------------


def test_init_chunk_upload_writes_meta(tmp_path) -> None:
    db = MagicMock()
    with (
        patch("app.services.video_service.LOCAL_TMP_DIR", tmp_path),
        patch("app.services.video_service.video_repo.count_by_user_id", return_value=0),
        patch("app.services.video_service.settings.user_video_quota", 10),
    ):
        upload_id = video_service.init_chunk_upload(
            db, uuid.uuid4(), "タイトル", "v.mp4", 3
        )

    meta_path = tmp_path / upload_id / "meta.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta == {"title": "タイトル", "filename": "v.mp4", "total_chunks": 3}


def test_save_chunk_raises_when_upload_dir_missing(tmp_path) -> None:
    with patch("app.services.video_service.LOCAL_TMP_DIR", tmp_path):
        with pytest.raises(FileNotFoundError):
            video_service.save_chunk("missing", 0, _upload_file())


def test_save_chunk_writes_chunk_file(tmp_path) -> None:
    (tmp_path / "up1").mkdir()
    with patch("app.services.video_service.LOCAL_TMP_DIR", tmp_path):
        video_service.save_chunk("up1", 0, _upload_file(content=b"chunk0"))

    assert (tmp_path / "up1" / "0").read_bytes() == b"chunk0"


def test_complete_chunk_upload_raises_when_dir_missing(tmp_path) -> None:
    db, bt = MagicMock(), MagicMock()
    with (
        patch("app.services.video_service.LOCAL_TMP_DIR", tmp_path),
        patch("app.services.video_service.video_repo.count_by_user_id", return_value=0),
        patch("app.services.video_service.settings.user_video_quota", 10),
    ):
        with pytest.raises(FileNotFoundError):
            video_service.complete_chunk_upload(db, uuid.uuid4(), "missing", bt)


def test_complete_chunk_upload_raises_when_chunk_missing(tmp_path) -> None:
    db, bt = MagicMock(), MagicMock()
    upload_id = "up1"
    upload_dir = tmp_path / upload_id
    upload_dir.mkdir()
    (upload_dir / "meta.json").write_text(
        json.dumps({"title": "t", "filename": "v.mp4", "total_chunks": 2})
    )
    (upload_dir / "0").write_bytes(b"aa")  # chunk 1 が欠落

    with (
        patch("app.services.video_service.LOCAL_TMP_DIR", tmp_path),
        patch("app.services.video_service.video_repo.count_by_user_id", return_value=0),
        patch("app.services.video_service.settings.user_video_quota", 10),
    ):
        with pytest.raises(FileNotFoundError):
            video_service.complete_chunk_upload(db, uuid.uuid4(), upload_id, bt)


def test_complete_chunk_upload_schedules_background_task(tmp_path) -> None:
    db, bt = MagicMock(), MagicMock()
    upload_id = "up1"
    upload_dir = tmp_path / upload_id
    upload_dir.mkdir()
    (upload_dir / "meta.json").write_text(
        json.dumps({"title": "t", "filename": "v.mp4", "total_chunks": 2})
    )
    (upload_dir / "0").write_bytes(b"aa")
    (upload_dir / "1").write_bytes(b"bb")
    video = _make_video()

    with (
        patch("app.services.video_service.LOCAL_TMP_DIR", tmp_path),
        patch("app.services.video_service.video_repo.count_by_user_id", return_value=0),
        patch("app.services.video_service.settings.user_video_quota", 10),
        patch("app.services.video_service.storage_service.upload_file") as upload_file,
        patch(
            "app.services.video_service.video_repo.create", return_value=video
        ) as video_create,
        patch(
            "app.services.video_service.video_repo.update_status", return_value=video
        ) as video_update_status,
        patch(
            "app.services.video_service.job_repo.create",
            return_value=SimpleNamespace(id=uuid.uuid4()),
        ) as job_create,
    ):
        result = video_service.complete_chunk_upload(db, uuid.uuid4(), upload_id, bt)

    assert result is video
    upload_file.assert_not_called()
    video_create.assert_called_once()
    video_update_status.assert_called_once_with(db, video.id, VideoStatus.queued)
    job_create.assert_called_once()
    bt.add_task.assert_called_once()  # バックグラウンドタスクがスケジュールされる


def test_process_chunk_upload_merges_and_uploads(tmp_path) -> None:
    upload_id = "up1"
    upload_dir = tmp_path / upload_id
    upload_dir.mkdir()
    (upload_dir / "meta.json").write_text(
        json.dumps({"title": "t", "filename": "v.mp4", "total_chunks": 2})
    )
    (upload_dir / "0").write_bytes(b"aa")
    (upload_dir / "1").write_bytes(b"bb")
    r2_key = "videos/f1.mp4"
    video_id, job_id = uuid.uuid4(), uuid.uuid4()
    db = MagicMock()

    with (
        patch("app.services.video_service.LOCAL_TMP_DIR", tmp_path),
        patch("app.services.video_service.SessionLocal") as sl,
        patch("app.services.video_service.storage_service.upload_file") as upload_file,
        patch("app.services.video_service._extract_duration", return_value=10.0),
        patch(
            "app.services.video_service.video_repo.update_source_duration"
        ) as update_source_duration,
        patch(
            "app.services.video_service.video_repo.update_status"
        ) as video_update_status,
        patch("app.services.video_service.call_ml_service") as ml,
    ):
        sl.return_value.__enter__.return_value = db
        video_service.process_chunk_upload(
            upload_id, r2_key, str(video_id), str(job_id)
        )

    # 結合ファイルが R2 に上がる
    upload_file.assert_called_once()
    assert upload_file.call_args.args[1] == r2_key
    # 元動画長が保存される
    update_source_duration.assert_called_once_with(db, video_id, 10.0)
    # 一時ファイル（チャンク・結合ファイル）が残らない
    assert list(tmp_path.iterdir()) == []
    # ML キックは背景内で直接呼ばれる
    ml.assert_called_once_with(r2_key, str(job_id), str(video_id))
    # processing への遷移は call_ml_service の責務なのでここでは行わない
    video_update_status.assert_not_called()


def test_process_chunk_upload_marks_failed_on_error(tmp_path) -> None:
    upload_id = "up1"
    upload_dir = tmp_path / upload_id
    upload_dir.mkdir()
    (upload_dir / "meta.json").write_text(
        json.dumps({"title": "t", "filename": "v.mp4", "total_chunks": 2})
    )
    (upload_dir / "0").write_bytes(b"aa")
    (upload_dir / "1").write_bytes(b"bb")
    r2_key = "videos/f1.mp4"
    video_id, job_id = uuid.uuid4(), uuid.uuid4()
    db = MagicMock()

    with (
        patch("app.services.video_service.LOCAL_TMP_DIR", tmp_path),
        patch("app.services.video_service.SessionLocal") as sl,
        patch(
            "app.services.video_service.storage_service.upload_file",
            side_effect=RuntimeError("R2 down"),
        ),
        patch(
            "app.services.video_service.video_repo.update_status"
        ) as video_update_status,
        patch("app.services.video_service.job_repo.update_status") as job_update_status,
        patch("app.services.video_service.call_ml_service") as ml,
    ):
        sl.return_value.__enter__.return_value = db
        # 例外は外に漏れず、関数内で処理される
        video_service.process_chunk_upload(
            upload_id, r2_key, str(video_id), str(job_id)
        )

    # video / job とも failed になる
    video_update_status.assert_called_once_with(db, video_id, VideoStatus.failed)
    assert job_update_status.call_args.kwargs["status"] is JobStatus.failed
    # 失敗したら ML はキックしない
    ml.assert_not_called()


# --- delete_video -----------------------------------------------------------


def test_delete_video_returns_false_when_missing() -> None:
    db = MagicMock()
    with (
        patch("app.services.video_service.video_repo.get_by_id", return_value=None),
        patch("app.services.video_service.storage_service.delete_file") as delete_file,
    ):
        assert video_service.delete_video(db, uuid.uuid4()) is False
    delete_file.assert_not_called()


def test_delete_video_removes_records_and_r2_files() -> None:
    db = MagicMock()
    video = _make_video(storage_path="videos/a.mp4", output_path="outputs/a.mp4")
    jobs = [SimpleNamespace(id=uuid.uuid4()), SimpleNamespace(id=uuid.uuid4())]
    with (
        patch("app.services.video_service.video_repo.get_by_id", return_value=video),
        patch("app.services.video_service.job_repo.get_by_video_id", return_value=jobs),
        patch(
            "app.services.video_service.notification_log_repo.delete_by_job_id"
        ) as del_logs,
        patch("app.services.video_service.clip_repo.delete_by_video_id") as del_clips,
        patch("app.services.video_service.job_repo.delete_by_video_id") as del_jobs,
        patch("app.services.video_service.video_repo.delete") as del_video,
        patch("app.services.video_service.storage_service.delete_file") as delete_file,
    ):
        result = video_service.delete_video(db, video.id)

    assert result is True
    assert del_logs.call_count == 2
    del_clips.assert_called_once()
    del_jobs.assert_called_once()
    del_video.assert_called_once()
    deleted_keys = {call.args[0] for call in delete_file.call_args_list}
    assert deleted_keys == {"videos/a.mp4", "outputs/a.mp4"}


def test_delete_video_skips_output_when_no_output_path() -> None:
    db = MagicMock()
    video = _make_video(storage_path="videos/a.mp4", output_path=None)
    with (
        patch("app.services.video_service.video_repo.get_by_id", return_value=video),
        patch("app.services.video_service.job_repo.get_by_video_id", return_value=[]),
        patch("app.services.video_service.clip_repo.delete_by_video_id"),
        patch("app.services.video_service.job_repo.delete_by_video_id"),
        patch("app.services.video_service.video_repo.delete"),
        patch("app.services.video_service.storage_service.delete_file") as delete_file,
    ):
        video_service.delete_video(db, video.id)

    delete_file.assert_called_once_with("videos/a.mp4")


def test_delete_video_keeps_db_records_when_r2_delete_fails() -> None:
    """R2 削除に失敗したら例外を送出し、DB の行（R2 キーへの唯一のポインタ）を残す。

    先に DB を消すと、R2 削除失敗時にキーを知る手段が失われ、誰からも
    参照されないオブジェクトが R2 に残り続ける（オーファン化）ため。
    """
    db = MagicMock()
    video = _make_video(storage_path="videos/a.mp4", output_path=None)
    with (
        patch("app.services.video_service.video_repo.get_by_id", return_value=video),
        patch("app.services.video_service.job_repo.get_by_video_id", return_value=[]),
        patch("app.services.video_service.clip_repo.delete_by_video_id") as del_clips,
        patch("app.services.video_service.job_repo.delete_by_video_id") as del_jobs,
        patch("app.services.video_service.video_repo.delete") as del_video,
        patch(
            "app.services.video_service.storage_service.delete_file",
            side_effect=RuntimeError("R2 down"),
        ),
    ):
        with pytest.raises(RuntimeError):
            video_service.delete_video(db, video.id)

    # DB の削除には一切到達しない（再削除の手がかりが残る）
    del_clips.assert_not_called()
    del_jobs.assert_not_called()
    del_video.assert_not_called()


# --- replace_clips ----------------------------------------------------------


def test_replace_clips_calls_repo_with_latest_job() -> None:
    db = MagicMock()
    video = _make_video(source_duration=None)
    job = SimpleNamespace(id=uuid.uuid4())
    clips_input = [SimpleNamespace(start_time=0.0, end_time=5.0)]
    with (
        patch(
            "app.services.video_service.job_repo.get_latest_by_video_id",
            return_value=job,
        ),
        patch(
            "app.services.video_service.clip_repo.replace_for_video", return_value=["c"]
        ) as repl,
    ):
        result = video_service.replace_clips(db, video, clips_input)

    assert result == ["c"]
    # replace_for_video(db, video_id, job_id, clips_data) の 3 番目に最新 job の id を流用
    assert repl.call_args.args[2] == job.id


def test_replace_clips_rejects_range_over_source_duration() -> None:
    db = MagicMock()
    video = _make_video(source_duration=10.0)
    clips_input = [SimpleNamespace(start_time=0.0, end_time=20.0)]
    with patch("app.services.video_service.job_repo.get_latest_by_video_id") as gj:
        with pytest.raises(HTTPException) as exc:
            video_service.replace_clips(db, video, clips_input)
    assert exc.value.status_code == 422
    gj.assert_not_called()


def test_replace_clips_409_when_no_job() -> None:
    db = MagicMock()
    video = _make_video(source_duration=None)
    clips_input = [SimpleNamespace(start_time=0.0, end_time=5.0)]
    with patch(
        "app.services.video_service.job_repo.get_latest_by_video_id", return_value=None
    ):
        with pytest.raises(HTTPException) as exc:
            video_service.replace_clips(db, video, clips_input)
    assert exc.value.status_code == 409


# --- rebuild_output ---------------------------------------------------------


def test_rebuild_output_with_clips_uploads_and_completes() -> None:
    db = MagicMock()
    video = _make_video(storage_path="videos/a.mp4")
    clips = [{"start_time": 0.0, "end_time": 5.0}]
    with (
        patch("app.services.video_service.video_repo.get_by_id", return_value=video),
        patch(
            "app.services.video_service.storage_service.generate_presigned_url",
            return_value="http://signed",
        ),
        patch("app.services.video_service.storage_service.upload_file") as upload,
        patch("httpx.Client", return_value=_httpx_client_cm()),
        patch("app.services.video_service.clip_video") as clipv,
        patch("app.services.video_service._extract_duration", return_value=12.0),
        patch("app.services.video_service.video_repo.update_output_path"),
        patch("app.services.video_service.video_repo.update_duration") as vdur,
        patch("app.services.video_service.video_repo.update_status") as vupd,
    ):
        key = video_service.rebuild_output(db, video.id, clips)

    clipv.assert_called_once()
    assert key == f"outputs/{video.id}/play_scenes.mp4"
    assert upload.call_args.args[1] == f"outputs/{video.id}/play_scenes.mp4"
    vdur.assert_called_once_with(db, video.id, 12.0)
    vupd.assert_called_once_with(db, video.id, VideoStatus.completed)


def test_rebuild_output_empty_clips_sets_empty_output() -> None:
    db = MagicMock()
    video = _make_video()
    with (
        patch("app.services.video_service.video_repo.get_by_id", return_value=video),
        patch("app.services.video_service.clip_video") as clipv,
        patch("app.services.video_service.storage_service.upload_file") as upload,
        patch("app.services.video_service.video_repo.update_output_path") as out_path,
        patch("app.services.video_service.video_repo.update_status") as vupd,
    ):
        key = video_service.rebuild_output(db, video.id, [])

    clipv.assert_not_called()
    upload.assert_not_called()
    out_path.assert_called_once_with(db, video.id, "")
    vupd.assert_called_once_with(db, video.id, VideoStatus.completed)
    assert key == ""


# --- process_export ---------------------------------------------------------


def test_process_export_rebuilds_from_current_clips() -> None:
    video_id = uuid.uuid4()
    clip_objs = [SimpleNamespace(start_time=0.0, end_time=5.0)]
    with (
        patch("app.services.video_service.SessionLocal") as sl,
        patch(
            "app.services.video_service.clip_repo.get_by_video_id",
            return_value=clip_objs,
        ),
        patch("app.services.video_service.rebuild_output") as rebuild,
    ):
        sl.return_value.__enter__.return_value = MagicMock()
        video_service.process_export(video_id)

    rebuild.assert_called_once()
    # 現在の clip 区間が dict 化されて rebuild_output に渡る
    assert rebuild.call_args.args[2] == [{"start_time": 0.0, "end_time": 5.0}]


def test_process_export_sets_ready_on_failure() -> None:
    video_id = uuid.uuid4()
    db = MagicMock()
    with (
        patch("app.services.video_service.SessionLocal") as sl,
        patch("app.services.video_service.clip_repo.get_by_video_id", return_value=[]),
        patch(
            "app.services.video_service.rebuild_output",
            side_effect=RuntimeError("boom"),
        ),
        patch("app.services.video_service.video_repo.update_status") as vupd,
    ):
        sl.return_value.__enter__.return_value = db
        video_service.process_export(video_id)

    # 失敗時は ready に戻して再書き出しできるようにする
    vupd.assert_called_once_with(db, video_id, VideoStatus.ready)


# --- export_video -----------------------------------------------------------


def test_export_video_schedules_background_task() -> None:
    db, bt = MagicMock(), MagicMock()
    video = _make_video(status=VideoStatus.ready)
    with (
        patch(
            "app.services.video_service.clip_repo.get_by_video_id",
            return_value=[SimpleNamespace()],
        ),
        patch(
            "app.services.video_service.video_repo.update_status", return_value=video
        ) as vupd,
    ):
        video_service.export_video(db, video, bt)

    vupd.assert_called_once_with(db, video.id, VideoStatus.processing)
    bt.add_task.assert_called_once()


def test_export_video_400_when_no_clips() -> None:
    db, bt = MagicMock(), MagicMock()
    video = _make_video(status=VideoStatus.ready)
    with patch("app.services.video_service.clip_repo.get_by_video_id", return_value=[]):
        with pytest.raises(HTTPException) as exc:
            video_service.export_video(db, video, bt)
    assert exc.value.status_code == 400


def test_export_video_409_when_processing() -> None:
    db, bt = MagicMock(), MagicMock()
    video = _make_video(status=VideoStatus.processing)
    with patch(
        "app.services.video_service.clip_repo.get_by_video_id",
        return_value=[SimpleNamespace()],
    ):
        with pytest.raises(HTTPException) as exc:
            video_service.export_video(db, video, bt)
    assert exc.value.status_code == 409


# --- recover_interrupted_exports --------------------------------------------


def test_recover_interrupted_exports_resets_to_ready() -> None:
    videos = [SimpleNamespace(id=uuid.uuid4()), SimpleNamespace(id=uuid.uuid4())]
    db = MagicMock()
    with (
        patch("app.services.video_service.SessionLocal") as sl,
        patch(
            "app.services.video_service.video_repo.get_processing_without_running_job",
            return_value=videos,
        ),
        patch("app.services.video_service.video_repo.update_status") as vupd,
    ):
        sl.return_value.__enter__.return_value = db
        video_service.recover_interrupted_exports()

    # 中断された書き出しはすべて ready に戻る
    assert [c.args for c in vupd.call_args_list] == [
        (db, videos[0].id, VideoStatus.ready),
        (db, videos[1].id, VideoStatus.ready),
    ]


def test_recover_interrupted_exports_noop_when_none() -> None:
    with (
        patch("app.services.video_service.SessionLocal") as sl,
        patch(
            "app.services.video_service.video_repo.get_processing_without_running_job",
            return_value=[],
        ),
        patch("app.services.video_service.video_repo.update_status") as vupd,
    ):
        sl.return_value.__enter__.return_value = MagicMock()
        video_service.recover_interrupted_exports()

    vupd.assert_not_called()


def test_recover_interrupted_to_failed_marks_orphaned_queued_as_failed() -> None:
    job = SimpleNamespace(id=uuid.uuid4(), video_id=uuid.uuid4())
    db = MagicMock()

    with (
        patch("app.services.video_service.SessionLocal") as sl,
        patch(
            "app.services.video_service.job_repo.get_queued_started_null_jobs",
            return_value=[job],
        ),
        patch("app.services.video_service.job_repo.update_status") as jupd,
        patch("app.services.video_service.video_repo.update_status") as vupd,
    ):
        sl.return_value.__enter__.return_value = db
        video_service.recover_interrupted_to_failed()

    assert jupd.call_args.kwargs["status"] is JobStatus.failed
    vupd.assert_called_once_with(db, job.video_id, VideoStatus.failed)


def test_recover_interrupted_to_failed_noop_when_none() -> None:
    job = SimpleNamespace(id=uuid.uuid4(), video_id=uuid.uuid4())
    db = MagicMock()

    with (
        patch("app.services.video_service.SessionLocal") as sl,
        patch(
            "app.services.video_service.job_repo.get_queued_started_null_jobs",
            return_value=[],
        ),
        patch("app.services.video_service.job_repo.update_status") as jupd,
        patch("app.services.video_service.video_repo.update_status") as vupd,
    ):
        sl.return_value.__enter__.return_value = db
        video_service.recover_interrupted_to_failed()

    jupd.assert_not_called()
    vupd.assert_not_called()
