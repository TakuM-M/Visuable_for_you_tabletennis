"""clips ルーターのテスト。

clips ルーターはエンドポイントが1本だけ:
    GET /videos/{video_id}/clips → list_clips_by_video
      → clip_repo.get_by_video_id(db, video_id) の結果をそのまま返す（認証必須）

ルーターテストの方針は test_jobs.py / test_videos.py と同じ:
  - 検証対象は HTTP 層だけ（ルーティング / 認証要否 / ステータスコード / レスポンス形 /
    下位レイヤを正しく呼ぶか）。repo・DB の中身は見ない。
  - get_db / get_current_user は Depends なので app.dependency_overrides で差し替える。
  - 本体内で呼ぶ repo は patch で差し替える（patch 先は「使われている場所」=
    app.routers.clips.clip_repo.* を指定する）。
"""
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.deps import get_current_user
from app.db.session import get_db
from app.routers import clips


# ---------------------------------------------------------------------------
# 共通ヘルパー
# ---------------------------------------------------------------------------
def _make_app() -> FastAPI:
    """clips ルーターだけを載せた使い捨てアプリ。get_db は None に固定。"""
    app = FastAPI()
    app.include_router(clips.router)
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


def _make_clip(**kw) -> SimpleNamespace:
    """ClipResponse にシリアライズできるニセクリップオブジェクト。

    属性は app/schemas/clip.py の ClipResponse のフィールドと一致させる。
    """
    defaults = dict(
        id=uuid.uuid4(),
        video_id=uuid.uuid4(),
        job_id=uuid.uuid4(),
        start_time=0.0,
        end_time=10.0,
        sort_order=0,
        storage_path="clips/test.mp4",
        created_at=datetime.now(timezone.utc),
    )
    defaults.update(kw)
    return SimpleNamespace(**defaults)


def _make_video(**kw) -> SimpleNamespace:
    """get_owned_video が返すニセ動画。所有者チェックは user_id だけ見る。"""
    defaults = dict(id=uuid.uuid4(), user_id=uuid.uuid4())
    defaults.update(kw)
    return SimpleNamespace(**defaults)


def _authed_client(user: SimpleNamespace | None = None) -> TestClient:
    """get_current_user を差し替えて「ログイン済み」にした TestClient を返す。"""
    app = _make_app()
    app.dependency_overrides[get_current_user] = lambda: user or _make_user()
    return TestClient(app)


# ===========================================================================
# GET /videos/{video_id}/clips （動画に紐づくクリップ一覧）
#   ルーター: video = Depends(get_owned_video) → clip_repo.get_by_video_id(db, video.id)
#   所有者チェックは get_owned_video（app.core.deps）が担うので、
#   テストは app.core.deps.video_repo.get_by_id を patch し動画の所有者を制御する。
# ===========================================================================
def test_list_clips_by_video_returns_clips() -> None:
    """所有者なら、repo が返したクリップのリストがそのまま JSON 配列になる。"""
    user = _make_user()
    video = _make_video(user_id=user.id)
    items = [_make_clip(video_id=video.id), _make_clip(video_id=video.id)]

    with patch("app.core.deps.video_repo.get_by_id", return_value=video), \
         patch("app.routers.clips.clip_repo.get_by_video_id", return_value=items):
        client = _authed_client(user)
        resp = client.get(f"/videos/{video.id}/clips")

    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 2
    # UUID は JSON では文字列になる。全クリップが同じ video_id を持つことを確認。
    assert {item["video_id"] for item in body} == {str(video.id)}


def test_list_clips_by_video_empty_returns_empty_list() -> None:
    """クリップが1件も無い動画でも、エラーではなく空配列 [] を返す。"""
    user = _make_user()
    video = _make_video(user_id=user.id)
    with patch("app.core.deps.video_repo.get_by_id", return_value=video), \
         patch("app.routers.clips.clip_repo.get_by_video_id", return_value=[]):
        client = _authed_client(user)
        resp = client.get(f"/videos/{video.id}/clips")

    assert resp.status_code == 200
    assert resp.json() == []


def test_list_clips_by_video_other_user_returns_403() -> None:
    """他人の動画のクリップ一覧は 403。"""
    user = _make_user()
    video = _make_video(user_id=uuid.uuid4())  # 別人の動画
    with patch("app.core.deps.video_repo.get_by_id", return_value=video):
        client = _authed_client(user)
        resp = client.get(f"/videos/{video.id}/clips")
    assert resp.status_code == 403


def test_list_clips_by_video_not_found_returns_404() -> None:
    """動画が存在しなければ 404。"""
    with patch("app.core.deps.video_repo.get_by_id", return_value=None):
        client = _authed_client()
        resp = client.get(f"/videos/{uuid.uuid4()}/clips")
    assert resp.status_code == 404


def test_list_clips_by_video_requires_auth() -> None:
    """get_current_user を override しない → 本物の認証が動いて 401。"""
    client = TestClient(_make_app())
    resp = client.get(f"/videos/{uuid.uuid4()}/clips")
    assert resp.status_code == 401


# ===========================================================================
# PUT /videos/{video_id}/clips （切り抜きの一括置換）
#   ルーター: video = Depends(get_owned_video) → video_service.replace_clips(...)
#   所有者チェックは get_owned_video が担うので app.core.deps.video_repo.get_by_id を patch。
#   置換ロジック本体は video_service.replace_clips を patch して HTTP 層だけ見る。
# ===========================================================================
def test_put_clips_replaces_and_returns_list() -> None:
    """所有者なら、service が返したクリップ配列がそのまま JSON で返る。"""
    user = _make_user()
    video = _make_video(user_id=user.id)
    returned = [
        _make_clip(video_id=video.id, sort_order=0),
        _make_clip(video_id=video.id, sort_order=1),
    ]
    with patch("app.core.deps.video_repo.get_by_id", return_value=video), \
         patch("app.routers.clips.video_service.replace_clips", return_value=returned) as replace:
        client = _authed_client(user)
        resp = client.put(
            f"/videos/{video.id}/clips",
            json={"clips": [
                {"start_time": 0.0, "end_time": 5.0},
                {"start_time": 6.0, "end_time": 9.0},
            ]},
        )

    assert resp.status_code == 200
    assert len(resp.json()) == 2
    replace.assert_called_once()


def test_put_clips_other_user_returns_403() -> None:
    """他人の動画のクリップは置換できず 403。"""
    user = _make_user()
    video = _make_video(user_id=uuid.uuid4())
    with patch("app.core.deps.video_repo.get_by_id", return_value=video):
        client = _authed_client(user)
        resp = client.put(f"/videos/{video.id}/clips", json={"clips": []})
    assert resp.status_code == 403


def test_put_clips_not_found_returns_404() -> None:
    with patch("app.core.deps.video_repo.get_by_id", return_value=None):
        client = _authed_client()
        resp = client.put(f"/videos/{uuid.uuid4()}/clips", json={"clips": []})
    assert resp.status_code == 404


def test_put_clips_requires_auth() -> None:
    client = TestClient(_make_app())
    resp = client.put(f"/videos/{uuid.uuid4()}/clips", json={"clips": []})
    assert resp.status_code == 401


def test_put_clips_invalid_range_returns_422() -> None:
    """end_time <= start_time の区間はバリデーションで 422。"""
    user = _make_user()
    video = _make_video(user_id=user.id)
    with patch("app.core.deps.video_repo.get_by_id", return_value=video):
        client = _authed_client(user)
        resp = client.put(
            f"/videos/{video.id}/clips",
            json={"clips": [{"start_time": 5.0, "end_time": 2.0}]},
        )
    assert resp.status_code == 422
