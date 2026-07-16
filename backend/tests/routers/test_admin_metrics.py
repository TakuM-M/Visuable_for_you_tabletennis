"""/admin/metrics エンドポイントの認証とレスポンス形のテスト"""

from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers import admin
from app.schemas.admin import StorageMetricsResponse


def _make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(admin.router)
    return app


def test_metrics_requires_internal_api_key() -> None:
    """ヘッダなしリクエストは 401"""
    app = _make_app()
    client = TestClient(app)
    with patch("app.core.deps.settings.internal_api_key", "secret"):
        resp = client.get("/admin/metrics")
    assert resp.status_code == 401


def test_metrics_rejects_wrong_key() -> None:
    """誤った API key は 401"""
    app = _make_app()
    client = TestClient(app)
    with patch("app.core.deps.settings.internal_api_key", "secret"):
        resp = client.get("/admin/metrics", headers={"X-Internal-Api-Key": "wrong"})
    assert resp.status_code == 401


def test_metrics_rejects_when_key_unset() -> None:
    """internal_api_key が未設定の環境は常に 401（誤公開防止）"""
    app = _make_app()
    client = TestClient(app)
    with patch("app.core.deps.settings.internal_api_key", ""):
        resp = client.get("/admin/metrics", headers={"X-Internal-Api-Key": "anything"})
    assert resp.status_code == 401


def test_metrics_returns_collected_data() -> None:
    """正しい API key で metrics_service の結果が返る"""
    app = _make_app()
    client = TestClient(app)
    fake = StorageMetricsResponse(
        r2_total_bytes=12345,
        r2_object_count=2,
        db_video_count=1,
        videos_per_user={"user-uuid": 1},
    )
    with (
        patch("app.core.deps.settings.internal_api_key", "secret"),
        patch(
            "app.routers.admin.metrics_service.collect_storage_metrics",
            return_value=fake,
        ),
        patch("app.routers.admin.get_db", return_value=iter([None])),
    ):
        # get_db Depends は generator 形式なので Depends override の方が綺麗だが
        # 今回は collect_storage_metrics をモックしているので DB は使われない
        app.dependency_overrides[admin.get_db] = lambda: None
        resp = client.get("/admin/metrics", headers={"X-Internal-Api-Key": "secret"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["r2_total_bytes"] == 12345
    assert body["r2_object_count"] == 2
    assert body["db_video_count"] == 1
    assert body["videos_per_user"] == {"user-uuid": 1}
