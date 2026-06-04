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
        storage_path="clips/test.mp4",
        created_at=datetime.now(timezone.utc),
    )
    defaults.update(kw)
    return SimpleNamespace(**defaults)


def _authed_client(user: SimpleNamespace | None = None) -> TestClient:
    """get_current_user を差し替えて「ログイン済み」にした TestClient を返す。"""
    app = _make_app()
    app.dependency_overrides[get_current_user] = lambda: user or _make_user()
    return TestClient(app)


# ===========================================================================
# GET /videos/{video_id}/clips （動画に紐づくクリップ一覧）
# ===========================================================================
def test_list_clips_by_video_returns_clips() -> None:
    """repo が返したクリップのリストがそのまま JSON 配列になる。"""
    video_id = uuid.uuid4()
    items = [_make_clip(video_id=video_id), _make_clip(video_id=video_id)]

    with patch("app.routers.clips.clip_repo.get_by_video_id", return_value=items):
        client = _authed_client()
        resp = client.get(f"/videos/{video_id}/clips")

    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 2
    # UUID は JSON では文字列になる。全クリップが同じ video_id を持つことを確認。
    assert {item["video_id"] for item in body} == {str(video_id)}


def test_list_clips_by_video_empty_returns_empty_list() -> None:
    """クリップが1件も無い動画でも、エラーではなく空配列 [] を返す。"""
    with patch("app.routers.clips.clip_repo.get_by_video_id", return_value=[]):
        client = _authed_client()
        resp = client.get(f"/videos/{uuid.uuid4()}/clips")

    assert resp.status_code == 200
    assert resp.json() == []


def test_list_clips_by_video_requires_auth() -> None:
    """get_current_user を override しない → 本物の認証が動いて 401。"""
    client = TestClient(_make_app())
    resp = client.get(f"/videos/{uuid.uuid4()}/clips")
    assert resp.status_code == 401
