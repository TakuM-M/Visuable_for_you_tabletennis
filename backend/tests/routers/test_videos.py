"""videos ルーターのテスト。

ルーターテストの方針:
  - 検証対象は HTTP 層だけ（ルーティング / 認証要否 / ステータスコード /
    レスポンス形 / 下位レイヤを正しく呼ぶか）。service・repo・DB の中身は見ない。
  - get_db / get_current_user は Depends なので app.dependency_overrides で差し替える。
  - 本体内で呼ぶ service / repo / storage は patch で差し替える
    （patch 先は「使われている場所」= app.routers.videos.* を指定する）。
"""
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.deps import get_current_user
from app.db.session import get_db
from app.models.video import VideoStatus
from app.routers import videos
from app.services import video_service


def _make_app() -> FastAPI:
    """videos ルーターだけを載せた使い捨てアプリ。get_db は None に固定。"""
    app = FastAPI()
    app.include_router(videos.router)
    app.dependency_overrides[get_db] = lambda: None
    return app


def _make_user(**kw) -> SimpleNamespace:
    """ニセの「ログイン中ユーザー」。ルーターは属性アクセスしかしないので十分。"""
    defaults = dict(
        id=uuid.uuid4(),
        email="user@example.com",
        password_hash="hashed",
        display_name="ユーザー",
        email_verified=True,
        created_at=datetime.now(timezone.utc),
    )
    defaults.update(kw)
    return SimpleNamespace(**defaults)


def _make_video(**kw) -> SimpleNamespace:
    """VideoResponse にシリアライズできるニセ動画オブジェクト。"""
    defaults = dict(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        title="テスト動画",
        storage_path="videos/test.mp4",
        output_path=None,
        duration=None,
        source_duration=None,
        status=VideoStatus.uploaded,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    defaults.update(kw)
    return SimpleNamespace(**defaults)


def _authed_client(user: SimpleNamespace | None = None) -> TestClient:
    """get_current_user を差し替えて「ログイン済み」にした TestClient を返す。"""
    app = _make_app()
    app.dependency_overrides[get_current_user] = lambda: user or _make_user()
    return TestClient(app)


# ---------------------------------------------------------------------------
# POST /videos （通常アップロード）
# ---------------------------------------------------------------------------
def test_upload_video_returns_201() -> None:
    """成功時は 201 と作成された動画を返し、service を1回呼ぶ"""
    user = _make_user()
    client = _authed_client(user)
    created = _make_video(user_id=user.id, title="新動画")

    with patch(
        "app.routers.videos.video_service.upload_video", return_value=created
    ) as upload:
        resp = client.post(
            "/videos",
            data={"title": "新動画"},                          # Form(...) フィールド
            files={"file": ("a.mp4", b"dummy", "video/mp4")},   # File(...) フィールド
        )

    assert resp.status_code == 201
    assert resp.json()["title"] == "新動画"
    upload.assert_called_once()


def test_upload_video_quota_exceeded_returns_409() -> None:
    """QuotaExceededError はルーターで 409 に変換される"""
    client = _authed_client()
    with patch(
        "app.routers.videos.video_service.upload_video",
        side_effect=video_service.QuotaExceededError("上限超過"),
    ):
        resp = client.post(
            "/videos",
            data={"title": "x"},
            files={"file": ("a.mp4", b"dummy", "video/mp4")},
        )
    assert resp.status_code == 409


# ---------------------------------------------------------------------------
# POST /videos/upload/init （チャンクアップロード初期化）
# ---------------------------------------------------------------------------
def test_chunk_upload_init_returns_upload_id() -> None:
    client = _authed_client()
    with patch(
        "app.routers.videos.video_service.init_chunk_upload",
        return_value="upload-123",
    ):
        resp = client.post(
            "/videos/upload/init",
            json={"title": "t", "filename": "a.mp4", "total_chunks": 3},
        )
    assert resp.status_code == 200
    assert resp.json()["upload_id"] == "upload-123"


def test_chunk_upload_init_quota_exceeded_returns_409() -> None:
    client = _authed_client()
    with patch(
        "app.routers.videos.video_service.init_chunk_upload",
        side_effect=video_service.QuotaExceededError("上限超過"),
    ):
        resp = client.post(
            "/videos/upload/init",
            json={"title": "t", "filename": "a.mp4", "total_chunks": 3},
        )
    assert resp.status_code == 409


# ---------------------------------------------------------------------------
# POST /videos/upload/{id}/chunk （チャンクデータ受信）
# ---------------------------------------------------------------------------
def test_chunk_upload_returns_204() -> None:
    client = _authed_client()
    with patch("app.routers.videos.video_service.save_chunk") as save:
        resp = client.post(
            "/videos/upload/abc/chunk",
            params={"index": 0},  # index はクエリパラメータ
            files={"file": ("chunk", b"data", "application/octet-stream")},
        )
    assert resp.status_code == 204
    save.assert_called_once()


def test_chunk_upload_unknown_upload_returns_404() -> None:
    """存在しない upload_id は FileNotFoundError → 404"""
    client = _authed_client()
    with patch(
        "app.routers.videos.video_service.save_chunk",
        side_effect=FileNotFoundError(),
    ):
        resp = client.post(
            "/videos/upload/abc/chunk",
            params={"index": 0},
            files={"file": ("chunk", b"data", "application/octet-stream")},
        )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /videos/upload/{id}/complete （チャンク完了・結合）
# ---------------------------------------------------------------------------
def test_chunk_upload_complete_returns_201() -> None:
    user = _make_user()
    client = _authed_client(user)
    created = _make_video(user_id=user.id)
    with patch(
        "app.routers.videos.video_service.complete_chunk_upload",
        return_value=created,
    ):
        resp = client.post("/videos/upload/abc/complete")
    assert resp.status_code == 201
    assert resp.json()["id"] == str(created.id)


def test_chunk_upload_complete_unknown_returns_404() -> None:
    client = _authed_client()
    with patch(
        "app.routers.videos.video_service.complete_chunk_upload",
        side_effect=FileNotFoundError("not found"),
    ):
        resp = client.post("/videos/upload/abc/complete")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /videos （ログインユーザーの一覧）
# ---------------------------------------------------------------------------
def test_list_videos_returns_user_videos() -> None:
    user = _make_user()
    client = _authed_client(user)
    items = [_make_video(user_id=user.id), _make_video(user_id=user.id)]
    with patch(
        "app.routers.videos.video_repo.get_by_user_id", return_value=items
    ):
        resp = client.get("/videos")
    assert resp.status_code == 200
    assert len(resp.json()) == 2


def test_list_videos_requires_auth() -> None:
    """get_current_user を override しない → 本物が動いて 401"""
    client = TestClient(_make_app())
    resp = client.get("/videos")
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# GET /videos/{id} （詳細）
#   所有者チェックは get_owned_video（app.core.deps）が担う。
#   テストは deps が使う app.core.deps.video_repo.get_by_id を patch し、
#   ニセ動画の user_id を「本人 / 他人」に振り分けて 200 / 403 / 404 を確認する。
# ---------------------------------------------------------------------------
def test_get_video_returns_video() -> None:
    user = _make_user()
    client = _authed_client(user)
    video = _make_video(user_id=user.id, title="練習試合")
    with patch("app.core.deps.video_repo.get_by_id", return_value=video):
        resp = client.get(f"/videos/{video.id}")
    assert resp.status_code == 200
    assert resp.json()["title"] == "練習試合"


def test_get_video_not_found_returns_404() -> None:
    client = _authed_client()
    with patch("app.core.deps.video_repo.get_by_id", return_value=None):
        resp = client.get(f"/videos/{uuid.uuid4()}")
    assert resp.status_code == 404


def test_get_video_other_user_returns_403() -> None:
    """他人の動画は UUID を知っていても 403（IDOR 対策）。"""
    user = _make_user()
    client = _authed_client(user)
    video = _make_video(user_id=uuid.uuid4())  # 別人の動画
    with patch("app.core.deps.video_repo.get_by_id", return_value=video):
        resp = client.get(f"/videos/{video.id}")
    assert resp.status_code == 403


def test_get_video_requires_auth() -> None:
    client = TestClient(_make_app())
    resp = client.get(f"/videos/{uuid.uuid4()}")
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# GET /videos/{id}/output （連結済み動画の presigned URL を JSON で返す・認証＋所有者必須）
# ---------------------------------------------------------------------------
def test_get_output_returns_presigned_url() -> None:
    user = _make_user()
    client = _authed_client(user)
    video = _make_video(user_id=user.id, output_path="outputs/done.mp4")
    with patch("app.core.deps.video_repo.get_by_id", return_value=video), \
         patch(
             "app.routers.videos.storage_service.generate_presigned_url",
             return_value="https://r2.example/signed",
         ):
        resp = client.get(f"/videos/{video.id}/output")
    assert resp.status_code == 200
    assert resp.json()["url"] == "https://r2.example/signed"


def test_get_output_requires_auth() -> None:
    """認証必須になった（旧仕様は認証不要だった）。"""
    client = TestClient(_make_app())
    resp = client.get(f"/videos/{uuid.uuid4()}/output")
    assert resp.status_code == 401


def test_get_output_other_user_returns_403() -> None:
    user = _make_user()
    client = _authed_client(user)
    video = _make_video(user_id=uuid.uuid4(), output_path="outputs/done.mp4")
    with patch("app.core.deps.video_repo.get_by_id", return_value=video):
        resp = client.get(f"/videos/{video.id}/output")
    assert resp.status_code == 403


def test_get_output_not_found_returns_404() -> None:
    client = _authed_client()
    with patch("app.core.deps.video_repo.get_by_id", return_value=None):
        resp = client.get(f"/videos/{uuid.uuid4()}/output")
    assert resp.status_code == 404


def test_get_output_not_generated_yet_returns_404() -> None:
    """動画はあるが output_path がまだ無い場合も 404"""
    user = _make_user()
    client = _authed_client(user)
    video = _make_video(user_id=user.id, output_path=None)
    with patch("app.core.deps.video_repo.get_by_id", return_value=video):
        resp = client.get(f"/videos/{video.id}/output")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /videos/{id}/source （元動画の presigned URL を JSON で返す・認証＋所有者必須）
#   新規切り抜きのトリミング UI 用。storage_path は常にあるので 404 は所有判定のみ。
# ---------------------------------------------------------------------------
def test_get_source_returns_presigned_url() -> None:
    user = _make_user()
    client = _authed_client(user)
    video = _make_video(user_id=user.id, storage_path="videos/src.mp4")
    with patch("app.core.deps.video_repo.get_by_id", return_value=video), \
         patch(
             "app.routers.videos.storage_service.generate_presigned_url",
             return_value="https://r2.example/source-signed",
         ) as gen_mock:
        resp = client.get(f"/videos/{video.id}/source")
    assert resp.status_code == 200
    assert resp.json()["url"] == "https://r2.example/source-signed"
    gen_mock.assert_called_once_with("videos/src.mp4")  # 元動画パスで署名している


def test_get_source_requires_auth() -> None:
    client = TestClient(_make_app())
    resp = client.get(f"/videos/{uuid.uuid4()}/source")
    assert resp.status_code == 401


def test_get_source_other_user_returns_403() -> None:
    user = _make_user()
    client = _authed_client(user)
    video = _make_video(user_id=uuid.uuid4())  # 別人の動画
    with patch("app.core.deps.video_repo.get_by_id", return_value=video):
        resp = client.get(f"/videos/{video.id}/source")
    assert resp.status_code == 403


def test_get_source_not_found_returns_404() -> None:
    client = _authed_client()
    with patch("app.core.deps.video_repo.get_by_id", return_value=None):
        resp = client.get(f"/videos/{uuid.uuid4()}/source")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# DELETE /videos/{id} （削除）
# ---------------------------------------------------------------------------
def test_delete_video_returns_204() -> None:
    user = _make_user()
    client = _authed_client(user)
    video = _make_video(user_id=user.id)
    with patch("app.core.deps.video_repo.get_by_id", return_value=video), \
         patch("app.routers.videos.video_service.delete_video") as delete:
        resp = client.delete(f"/videos/{video.id}")
    assert resp.status_code == 204
    delete.assert_called_once()


def test_delete_video_not_found_returns_404() -> None:
    """存在しない動画の削除は 404。service の削除は呼ばれない。"""
    client = _authed_client()
    with patch("app.core.deps.video_repo.get_by_id", return_value=None), \
         patch("app.routers.videos.video_service.delete_video") as delete:
        resp = client.delete(f"/videos/{uuid.uuid4()}")
    assert resp.status_code == 404
    delete.assert_not_called()


def test_delete_video_other_user_returns_403() -> None:
    """他人の動画は削除できず 403。service の削除は呼ばれない。"""
    user = _make_user()
    client = _authed_client(user)
    video = _make_video(user_id=uuid.uuid4())  # 別人の動画
    with patch("app.core.deps.video_repo.get_by_id", return_value=video), \
         patch("app.routers.videos.video_service.delete_video") as delete:
        resp = client.delete(f"/videos/{video.id}")
    assert resp.status_code == 403
    delete.assert_not_called()


# ---------------------------------------------------------------------------
# POST /videos/{id}/export （現在の切り抜きから連結動画を書き出す・認証＋所有者必須）
#   ルーター: video = Depends(get_owned_video) → video_service.export_video(...)
# ---------------------------------------------------------------------------
def test_export_video_returns_202() -> None:
    """所有者なら 202 と processing 状態の動画を返し、service を1回呼ぶ。"""
    user = _make_user()
    client = _authed_client(user)
    video = _make_video(user_id=user.id)
    processing = _make_video(
        id=video.id, user_id=user.id, status=VideoStatus.processing
    )
    with patch("app.core.deps.video_repo.get_by_id", return_value=video), \
         patch(
             "app.routers.videos.video_service.export_video", return_value=processing
         ) as export:
        resp = client.post(f"/videos/{video.id}/export")
    assert resp.status_code == 202
    assert resp.json()["status"] == "processing"
    export.assert_called_once()


def test_export_video_other_user_returns_403() -> None:
    """他人の動画は書き出せず 403。service は呼ばれない。"""
    user = _make_user()
    client = _authed_client(user)
    video = _make_video(user_id=uuid.uuid4())
    with patch("app.core.deps.video_repo.get_by_id", return_value=video), \
         patch("app.routers.videos.video_service.export_video") as export:
        resp = client.post(f"/videos/{video.id}/export")
    assert resp.status_code == 403
    export.assert_not_called()


def test_export_video_not_found_returns_404() -> None:
    client = _authed_client()
    with patch("app.core.deps.video_repo.get_by_id", return_value=None), \
         patch("app.routers.videos.video_service.export_video") as export:
        resp = client.post(f"/videos/{uuid.uuid4()}/export")
    assert resp.status_code == 404
    export.assert_not_called()


def test_export_video_requires_auth() -> None:
    client = TestClient(_make_app())
    resp = client.post(f"/videos/{uuid.uuid4()}/export")
    assert resp.status_code == 401
