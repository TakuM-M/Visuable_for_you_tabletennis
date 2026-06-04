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
    with patch("app.services.video_service.SessionLocal") as sl, \
         patch("app.services.video_service.job_repo.update_status"), \
         patch("app.services.video_service.video_repo.update_status"), \
         patch("app.services.video_service.storage_service.generate_presigned_url", return_value="http://dl"), \
         patch("app.services.video_service.USE_RUNPOD", False), \
         patch("app.services.video_service.httpx") as mock_httpx:
        sl.return_value.__enter__.return_value = MagicMock()
        client = mock_httpx.Client.return_value.__enter__.return_value
        video_service.call_ml_service("videos/a.mp4", job_id, video_id)

    posted_url = client.post.call_args.args[0]
    assert posted_url.endswith("/process")


def test_call_ml_service_posts_to_runpod_when_enabled() -> None:
    job_id, video_id = str(uuid.uuid4()), str(uuid.uuid4())
    with patch("app.services.video_service.SessionLocal") as sl, \
         patch("app.services.video_service.job_repo.update_status"), \
         patch("app.services.video_service.video_repo.update_status"), \
         patch("app.services.video_service.storage_service.generate_presigned_url", return_value="http://dl"), \
         patch("app.services.video_service.USE_RUNPOD", True), \
         patch("app.services.video_service.RUNPOD_ENDPOINT_ID", "ep"), \
         patch("app.services.video_service.httpx") as mock_httpx:
        sl.return_value.__enter__.return_value = MagicMock()
        client = mock_httpx.Client.return_value.__enter__.return_value
        client.post.return_value.json.return_value = {"id": "rp-1"}
        video_service.call_ml_service("videos/a.mp4", job_id, video_id)

    posted_url = client.post.call_args.args[0]
    assert "api.runpod.ai" in posted_url


def test_call_ml_service_delegates_failure_to_handler() -> None:
    job_id, video_id = str(uuid.uuid4()), str(uuid.uuid4())
    with patch("app.services.video_service.SessionLocal") as sl, \
         patch("app.services.video_service.job_repo.update_status"), \
         patch("app.services.video_service.video_repo.update_status"), \
         patch("app.services.video_service.storage_service.generate_presigned_url",
               side_effect=RuntimeError("R2 down")), \
         patch("app.services.job_service.handle_ml_failure") as handle:
        sl.return_value.__enter__.return_value = MagicMock()
        video_service.call_ml_service("videos/a.mp4", job_id, video_id)

    handle.assert_called_once()
    assert "ML呼び出し失敗" in handle.call_args.args[2]


# --- _register_video_and_start_ml ------------------------------------------

def test_register_video_creates_records_and_schedules_ml() -> None:
    db, bt = MagicMock(), MagicMock()
    video = _make_video()
    job = SimpleNamespace(id=uuid.uuid4())
    with patch("app.services.video_service.video_repo.create", return_value=video), \
         patch("app.services.video_service.video_repo.update_status"), \
         patch("app.services.video_service.job_repo.create", return_value=job):
        result = video_service._register_video_and_start_ml(
            db, uuid.uuid4(), "t", "videos/a.mp4", bt
        )

    assert result is video
    bt.add_task.assert_called_once()


# --- upload_video -----------------------------------------------------------

def test_upload_video_saves_uploads_and_registers(tmp_path) -> None:
    db, bt = MagicMock(), MagicMock()
    video = _make_video()
    with patch("app.services.video_service.LOCAL_TMP_DIR", tmp_path), \
         patch("app.services.video_service.video_repo.count_by_user_id", return_value=0), \
         patch("app.services.video_service.settings.user_video_quota", 10), \
         patch("app.services.video_service.storage_service.upload_file") as upload, \
         patch("app.services.video_service._register_video_and_start_ml", return_value=video) as register:
        result = video_service.upload_video(db, uuid.uuid4(), "t", _upload_file(), bt)

    assert result is video
    upload.assert_called_once()
    register.assert_called_once()


def test_upload_video_raises_when_quota_exceeded(tmp_path) -> None:
    db, bt = MagicMock(), MagicMock()
    with patch("app.services.video_service.LOCAL_TMP_DIR", tmp_path), \
         patch("app.services.video_service.video_repo.count_by_user_id", return_value=10), \
         patch("app.services.video_service.settings.user_video_quota", 10), \
         patch("app.services.video_service.storage_service.upload_file") as upload:
        with pytest.raises(video_service.QuotaExceededError):
            video_service.upload_video(db, uuid.uuid4(), "t", _upload_file(), bt)

    upload.assert_not_called()


# --- init_chunk_upload / save_chunk / complete_chunk_upload -----------------

def test_init_chunk_upload_writes_meta(tmp_path) -> None:
    db = MagicMock()
    with patch("app.services.video_service.LOCAL_TMP_DIR", tmp_path), \
         patch("app.services.video_service.video_repo.count_by_user_id", return_value=0), \
         patch("app.services.video_service.settings.user_video_quota", 10):
        upload_id = video_service.init_chunk_upload(db, uuid.uuid4(), "タイトル", "v.mp4", 3)

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


def test_complete_chunk_upload_merges_and_registers(tmp_path) -> None:
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

    with patch("app.services.video_service.LOCAL_TMP_DIR", tmp_path), \
         patch("app.services.video_service.video_repo.count_by_user_id", return_value=0), \
         patch("app.services.video_service.settings.user_video_quota", 10), \
         patch("app.services.video_service.storage_service.upload_file") as upload, \
         patch("app.services.video_service._register_video_and_start_ml", return_value=video) as register:
        result = video_service.complete_chunk_upload(db, uuid.uuid4(), upload_id, bt)

    assert result is video
    upload.assert_called_once()
    register.assert_called_once()
    assert not upload_dir.exists()  # 結合後にクリーンアップされる


def test_complete_chunk_upload_raises_when_dir_missing(tmp_path) -> None:
    db, bt = MagicMock(), MagicMock()
    with patch("app.services.video_service.LOCAL_TMP_DIR", tmp_path), \
         patch("app.services.video_service.video_repo.count_by_user_id", return_value=0), \
         patch("app.services.video_service.settings.user_video_quota", 10):
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

    with patch("app.services.video_service.LOCAL_TMP_DIR", tmp_path), \
         patch("app.services.video_service.video_repo.count_by_user_id", return_value=0), \
         patch("app.services.video_service.settings.user_video_quota", 10):
        with pytest.raises(FileNotFoundError):
            video_service.complete_chunk_upload(db, uuid.uuid4(), upload_id, bt)


# --- delete_video -----------------------------------------------------------

def test_delete_video_returns_false_when_missing() -> None:
    db = MagicMock()
    with patch("app.services.video_service.video_repo.get_by_id", return_value=None), \
         patch("app.services.video_service.storage_service.delete_file") as delete_file:
        assert video_service.delete_video(db, uuid.uuid4()) is False
    delete_file.assert_not_called()


def test_delete_video_removes_records_and_r2_files() -> None:
    db = MagicMock()
    video = _make_video(storage_path="videos/a.mp4", output_path="outputs/a.mp4")
    jobs = [SimpleNamespace(id=uuid.uuid4()), SimpleNamespace(id=uuid.uuid4())]
    with patch("app.services.video_service.video_repo.get_by_id", return_value=video), \
         patch("app.services.video_service.job_repo.get_by_video_id", return_value=jobs), \
         patch("app.services.video_service.notification_log_repo.delete_by_job_id") as del_logs, \
         patch("app.services.video_service.clip_repo.delete_by_video_id") as del_clips, \
         patch("app.services.video_service.job_repo.delete_by_video_id") as del_jobs, \
         patch("app.services.video_service.video_repo.delete") as del_video, \
         patch("app.services.video_service.storage_service.delete_file") as delete_file:
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
    with patch("app.services.video_service.video_repo.get_by_id", return_value=video), \
         patch("app.services.video_service.job_repo.get_by_video_id", return_value=[]), \
         patch("app.services.video_service.clip_repo.delete_by_video_id"), \
         patch("app.services.video_service.job_repo.delete_by_video_id"), \
         patch("app.services.video_service.video_repo.delete"), \
         patch("app.services.video_service.storage_service.delete_file") as delete_file:
        video_service.delete_video(db, video.id)

    delete_file.assert_called_once_with("videos/a.mp4")


def test_delete_video_is_resilient_to_r2_errors() -> None:
    db = MagicMock()
    video = _make_video(storage_path="videos/a.mp4", output_path=None)
    with patch("app.services.video_service.video_repo.get_by_id", return_value=video), \
         patch("app.services.video_service.job_repo.get_by_video_id", return_value=[]), \
         patch("app.services.video_service.clip_repo.delete_by_video_id"), \
         patch("app.services.video_service.job_repo.delete_by_video_id"), \
         patch("app.services.video_service.video_repo.delete"), \
         patch("app.services.video_service.storage_service.delete_file",
               side_effect=RuntimeError("R2 down")):
        # R2 削除が失敗しても DB 削除は完了しているので True
        assert video_service.delete_video(db, video.id) is True
