"""jobs ルーターのテスト。

ルーターテストの方針（test_videos.py と同じ考え方）:
  - 検証対象は HTTP 層だけ。具体的には次の4点に絞る:
      1. ルーティング（パス・メソッドが正しく繋がっているか）
      2. 認証の要否（ログイン or 内部APIキーが必要か）
      3. ステータスコード（200 / 202 / 401 / 404 / 409 ...）
      4. レスポンスの形 と「下位レイヤ（service / repo）を正しく呼ぶか」
    service / repo / DB の "中身" は見ない（それらは別レイヤのテストで担保する）。

  - get_db / get_current_user / require_internal_api_key は FastAPI の Depends なので、
    `app.dependency_overrides` で「ニセモノ」に差し替える。
      → DB に繋がず、ログイン済み・認証済みの状態を作れる。

  - ルーター本体の中で呼んでいる service / repo は `unittest.mock.patch` で差し替える。
    patch 先は「定義元」ではなく「使われている場所」を指定するのが鉄則。
    jobs ルーターは `from app.repositories.job import job_repository as job_repo` /
    `from app.services import job_service` で import しているので、
    patch 先は app.routers.jobs.job_repo.* / app.routers.jobs.job_service.* になる。
"""

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.deps import get_current_user, get_video_repository, require_internal_api_key
from app.db.session import get_db
from app.models.job import JobStatus
from app.routers import jobs
from tests.fakes import FakeVideoRepository


# ---------------------------------------------------------------------------
# 共通ヘルパー
# ---------------------------------------------------------------------------
def _make_app() -> FastAPI:
    """jobs ルーターだけを載せた使い捨てアプリ。

    本物の app（app.main:app）には他のルーターやミドルウェアが全部ぶら下がっていて
    重い。テストでは「今テストしたいルーターだけ」を載せた最小アプリを毎回作る。
    get_db は repo を patch するので実際には使われないが、Depends の解決を通すため
    None を返す関数に差し替えておく。
    """
    app = FastAPI()
    app.include_router(jobs.router)
    app.dependency_overrides[get_db] = lambda: None
    return app


def _make_user(**kw) -> SimpleNamespace:
    """ニセの「ログイン中ユーザー」。

    ルーターは current_user の属性（id など）にアクセスするだけなので、
    本物の User モデルを作らなくても SimpleNamespace（属性の入れ物）で十分。
    """
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


def _make_job(**kw) -> SimpleNamespace:
    """JobResponse にシリアライズできるニセジョブオブジェクト。

    重要: ここの属性は app/schemas/job.py の JobResponse のフィールドと
    過不足なく一致させる必要がある（JobResponse は from_attributes=True なので
    属性アクセスで値を読み取る）。フィールドが足りないと 500、
    余分なだけなら無視されるが、status は JobStatus の値しか受け付けない点に注意。
    """
    defaults = dict(
        id=uuid.uuid4(),
        video_id=uuid.uuid4(),
        status=JobStatus.queued,
        started_at=None,
        completed_at=None,
        error_message=None,
        retry_count=0,
        next_retry_at=None,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    defaults.update(kw)
    return SimpleNamespace(**defaults)


def _make_video(**kw) -> SimpleNamespace:
    """get_owned_video が返すニセ動画。所有者チェックは user_id だけ見る。"""
    defaults = dict(id=uuid.uuid4(), user_id=uuid.uuid4())
    defaults.update(kw)
    return SimpleNamespace(**defaults)


class _VideoRepositoryStub(FakeVideoRepository):
    """get_owned_video が使う get_by_id だけ答える。他は呼ばれたら落ちる。"""

    def __init__(self, video: SimpleNamespace | None = None) -> None:
        self.video = video

    def get_by_id(self, db: Session, video_id: uuid.UUID):
        return self.video


def _authed_client(
    user: SimpleNamespace | None = None,
    *,
    owned_video: SimpleNamespace | None = None,
) -> TestClient:
    """get_current_user を差し替えて「ログイン済み」にした TestClient を返す。

    こうしておくと、各テストでトークンを用意しなくても認証を通過できる。
    認証そのものを試したいテストでは、この override をしない素の TestClient を使う。
    """
    app = _make_app()
    app.dependency_overrides[get_current_user] = lambda: user or _make_user()
    # get_owned_video が参照するリポジトリも差し替える。owned_video を渡さなければ
    # 「動画が見つからない」状態（404）になる。
    app.dependency_overrides[get_video_repository] = lambda: _VideoRepositoryStub(
        owned_video
    )
    return TestClient(app)


# ===========================================================================
# GET /videos/{video_id}/jobs （動画に紐づくジョブ一覧）
#   ルーター: video = Depends(get_owned_video) → job_repo.get_by_video_id(db, video.id)
#   所有者チェックは get_owned_video（app.core.deps）が担うので、
#   テストは _authed_client(owned_video=...) で動画の所有者を制御する。
# ===========================================================================
def test_list_jobs_by_video_returns_jobs() -> None:
    """所有者なら、repo が返したジョブのリストがそのまま JSON 配列になる。"""
    user = _make_user()
    video = _make_video(user_id=user.id)
    items = [_make_job(video_id=video.id), _make_job(video_id=video.id)]

    with patch("app.routers.jobs.job_repo.get_by_video_id", return_value=items):
        client = _authed_client(user, owned_video=video)
        resp = client.get(f"/videos/{video.id}/jobs")

    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 2
    assert {item["video_id"] for item in body} == {str(video.id)}


def test_list_jobs_by_video_empty_returns_empty_list() -> None:
    """ジョブが1件も無い動画でも、エラーではなく空配列 [] を返す。"""
    user = _make_user()
    video = _make_video(user_id=user.id)
    with patch("app.routers.jobs.job_repo.get_by_video_id", return_value=[]):
        client = _authed_client(user, owned_video=video)
        resp = client.get(f"/videos/{video.id}/jobs")

    assert resp.status_code == 200
    assert resp.json() == []


def test_list_jobs_by_video_other_user_returns_403() -> None:
    """他人の動画のジョブ一覧は 403。"""
    user = _make_user()
    video = _make_video(user_id=uuid.uuid4())  # 別人の動画
    client = _authed_client(user, owned_video=video)
    resp = client.get(f"/videos/{video.id}/jobs")
    assert resp.status_code == 403


def test_list_jobs_by_video_not_found_returns_404() -> None:
    """動画が存在しなければ 404。"""
    client = _authed_client()
    resp = client.get(f"/videos/{uuid.uuid4()}/jobs")
    assert resp.status_code == 404


def test_list_jobs_by_video_requires_auth() -> None:
    """get_current_user を override しない → 本物の認証が動いて 401。"""
    client = TestClient(_make_app())
    resp = client.get(f"/videos/{uuid.uuid4()}/jobs")
    assert resp.status_code == 401


# ===========================================================================
# POST /jobs/{job_id}/retry （失敗ジョブの手動再実行）
#   ルーター:
#     job_service.retry_job(...)        ← 権限/状態の検証。NG なら HTTPException を投げる
#     job = job_repo.get_by_id(...)     ← 再取得
#     job is None なら 404 / それ以外は返す
# ===========================================================================
def test_retry_job_returns_200() -> None:
    """retry_job が成功し、再取得したジョブを 200 で返す。"""
    job = _make_job(status=JobStatus.queued)

    # retry_job は副作用（DB 更新・background_tasks 登録）だけで戻り値を使わないので、
    # patch すると自動で MagicMock になり「何もしない・例外も投げない」関数になる。
    with (
        patch("app.routers.jobs.job_service.retry_job") as retry,
        patch("app.routers.jobs.job_repo.get_by_id", return_value=job),
    ):
        client = _authed_client()
        resp = client.post(f"/jobs/{job.id}/retry")

    assert resp.status_code == 200
    assert resp.json()["id"] == str(job.id)
    retry.assert_called_once()  # service がちゃんと1回呼ばれたことも確認


def test_retry_job_propagates_service_error_409() -> None:
    """service が投げた HTTPException は、そのままレスポンスのステータスになる。

    例: 失敗していないジョブを再実行しようとすると service が 409 を投げる。
    ルーターは握りつぶさないので、クライアントには 409 が返る。
    """
    with patch(
        "app.routers.jobs.job_service.retry_job",
        side_effect=HTTPException(
            status_code=409, detail="失敗したジョブのみ再実行できます"
        ),
    ):
        client = _authed_client()
        resp = client.post(f"/jobs/{uuid.uuid4()}/retry")

    assert resp.status_code == 409
    assert resp.json()["detail"] == "失敗したジョブのみ再実行できます"


def test_retry_job_requires_auth() -> None:
    """認証必須。override しなければ 401。"""
    client = TestClient(_make_app())
    resp = client.post(f"/jobs/{uuid.uuid4()}/retry")
    assert resp.status_code == 401


# ===========================================================================
# POST /internal/jobs/{job_id}/complete （ML からの完了コールバック）
#   - 認証は get_current_user ではなく require_internal_api_key（内部APIキー）
#   - 重い結合処理は background_tasks に逃がし、即 202 を返すのがポイント
#   ルーター:
#     background_tasks.add_task(job_service.process_complete_job, job_id, clips)
#     return {"message": "受付完了"}  (status_code=202)
# ===========================================================================
def test_complete_job_returns_202_and_schedules_background_task() -> None:
    """正しい内部APIキー前提で 202 を返し、重い処理を背景タスクに委譲する。"""
    app = _make_app()
    # このエンドポイントは get_current_user ではなく内部APIキー認証。
    # 認証を通過させたいので require_internal_api_key を no-op に差し替える。
    app.dependency_overrides[require_internal_api_key] = lambda: None
    client = TestClient(app)

    job_id = uuid.uuid4()
    payload = {
        "job_id": str(job_id),
        "clips": [
            {"start_time": 0.0, "end_time": 5.0},
            {"start_time": 10.0, "end_time": 12.5},
        ],
    }

    # 実際の結合処理（process_complete_job）は重い & 外部 I/O を伴うので patch で潰す。
    # TestClient は「レスポンス返却後に background task を同期実行」するため、
    # client.post(...) を抜けた時点で mock が呼ばれている。
    with patch("app.routers.jobs.job_service.process_complete_job") as proc:
        resp = client.post(f"/internal/jobs/{job_id}/complete", json=payload)

    assert resp.status_code == 202
    assert resp.json() == {"message": "受付完了"}
    # 背景タスクとして process_complete_job が1回スケジュール・実行されたことを確認。
    proc.assert_called_once()


def test_complete_job_requires_internal_key_returns_401() -> None:
    """内部APIキーが無ければ 401。背景タスクも実行されない。

    require_internal_api_key を override しない素のアプリに、
    X-Internal-Api-Key ヘッダ無しでリクエストする → 認証で弾かれる。
    """
    client = TestClient(_make_app())  # require_internal_api_key は本物が動く
    payload = {"job_id": str(uuid.uuid4()), "clips": []}

    with patch("app.routers.jobs.job_service.process_complete_job") as proc:
        resp = client.post(f"/internal/jobs/{uuid.uuid4()}/complete", json=payload)

    assert resp.status_code == 401
    proc.assert_not_called()  # 認証で止まるので重い処理は呼ばれない


# ===========================================================================
# POST /internal/jobs/{job_id}/fail （ML からの失敗コールバック）
#   - complete と対になる入口。ML が自分の失敗を自覚できたときに叩かれる。
#   - 認証・背景タスク委譲・202 という形は complete と揃えている。
#   ルーター:
#     background_tasks.add_task(job_service.process_fail_job, job_id, request.error)
# ===========================================================================
def test_fail_job_returns_202_and_schedules_background_task() -> None:
    """202 を返し、job_id と error をそのまま背景タスクへ渡す。"""
    app = _make_app()
    app.dependency_overrides[require_internal_api_key] = lambda: None
    client = TestClient(app)

    job_id = uuid.uuid4()
    payload = {"error": "HTTPStatusError: 403 Forbidden"}

    with patch("app.routers.jobs.job_service.process_fail_job") as proc:
        resp = client.post(f"/internal/jobs/{job_id}/fail", json=payload)

    assert resp.status_code == 202
    assert resp.json() == {"message": "受付完了"}
    proc.assert_called_once_with(job_id, payload["error"])


def test_fail_job_requires_internal_key_returns_401() -> None:
    """内部APIキーが無ければ 401。失敗扱いの処理も走らない。"""
    client = TestClient(_make_app())

    with patch("app.routers.jobs.job_service.process_fail_job") as proc:
        resp = client.post(
            f"/internal/jobs/{uuid.uuid4()}/fail", json={"error": "boom"}
        )

    assert resp.status_code == 401
    proc.assert_not_called()


def test_fail_job_without_error_field_returns_422() -> None:
    """error は必須。欠けていれば Pydantic のバリデーションで 422。"""
    app = _make_app()
    app.dependency_overrides[require_internal_api_key] = lambda: None
    client = TestClient(app)

    with patch("app.routers.jobs.job_service.process_fail_job") as proc:
        resp = client.post(f"/internal/jobs/{uuid.uuid4()}/fail", json={})

    assert resp.status_code == 422
    proc.assert_not_called()
