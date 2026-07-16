"""共有テスト fixture。

- **mock テスト**: services / routers のロジックをテストする際、DB アクセスを
  `unittest.mock.patch` でニセモノに差し替える（既存の test_video_quota.py 等）。
  この conftest.py の fixture は使わない。

- **実 DB テスト**: repositories 層のテストでは、実際の PostgreSQL を相手に
  書き込み・読み出しを行って SQL の正しさを確認する。テスト専用 DB
  `tabletennis_test` をセッション開始時に作成し、各テスト後に全テーブルを
  truncate して独立性を保つ。

実 DB テストは backend コンテナ内で実行する想定:

    docker compose -f docker-compose.dev.yml exec backend uv run pytest
"""

from urllib.parse import urlparse

import pytest
import sqlalchemy as sa
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings
from app.models import Base  # import 副作用で全モデルが Base.metadata に登録される
from app.models.clip import Clip
from app.models.job import Job
from app.models.notification_log import NotificationLog
from app.models.user import User
from app.models.video import Video
from app.repositories import clip as clip_repo
from app.repositories import job as job_repo
from app.repositories import notification_log as notification_log_repo
from app.repositories import user as user_repo
from app.repositories import video as video_repo

TEST_DB_NAME = "tabletennis_test"


def _replace_db_name(url: str, db_name: str) -> str:
    """DATABASE_URL の DB 名部分だけ差し替えたものを返す"""
    parsed = urlparse(url)
    return parsed._replace(path=f"/{db_name}").geturl()


@pytest.fixture(scope="session")
def engine() -> Engine:
    """
    1. postgres システム DB に管理者として接続
    2. 既存接続を切ってから tabletennis_test を DROP → CREATE
    3. テスト DB に再接続し、Base.metadata.create_all で全テーブル作成
    """
    # 管理者権限としてシステムに接続するための engine
    admin_url = _replace_db_name(settings.database_url, "postgres")
    admin_engine = create_engine(admin_url, isolation_level="AUTOCOMMIT")
    with admin_engine.connect() as conn:
        conn.execute(
            sa.text(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                f"WHERE datname = '{TEST_DB_NAME}' AND pid <> pg_backend_pid()"
            )
        )
        conn.execute(sa.text(f'DROP DATABASE IF EXISTS "{TEST_DB_NAME}"'))
        conn.execute(sa.text(f'CREATE DATABASE "{TEST_DB_NAME}"'))
    admin_engine.dispose()

    # 本物のテストの engine
    test_engine = create_engine(_replace_db_name(settings.database_url, TEST_DB_NAME))
    Base.metadata.create_all(test_engine)
    yield test_engine  # テストの実行
    test_engine.dispose()


@pytest.fixture
def db(engine: Engine) -> Session:
    """各テストにフレッシュな Session を渡し、終了後に全テーブルを truncate する。

    function スコープ（デフォルト）なのでテスト 1 件ごとに新しい Session。
    truncate は FK の依存順に逆順で実行し、CASCADE で安全に消す。
    """
    SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
        with engine.begin() as conn:
            for table in reversed(Base.metadata.sorted_tables):
                conn.execute(sa.text(f'TRUNCATE TABLE "{table.name}" CASCADE'))


@pytest.fixture
def user(db: Session) -> User:
    """テスト用に 1 人ユーザーを作る fixture。

    videos / jobs / clips など FK で users.id を要求するテーブルを
    扱うテストで使い回す。fixture 自体が `db` を要求しているので、
    pytest が自動的に db → user の順に解決して渡してくれる。
    """
    return user_repo.create(
        db=db,
        email="owner@example.com",
        password_hash="hashed",
        display_name="Owner",
    )


@pytest.fixture
def video(db: Session, user: User) -> Video:
    """`user` に紐づいたテスト用 video を作る fixture。

    job / clip など videos.id を FK で要求するテーブルを扱うテストで使い回す。
    依存解決の流れは: db → user → video → test。
    """
    return video_repo.create(
        db=db,
        user_id=user.id,
        title="テスト動画",
        storage_path="videos/test.mp4",
    )


@pytest.fixture
def job(db: Session, video: Video) -> Job:
    """`video` に紐づいたテスト用 job を作る fixture。

    clip / notification_log は jobs.id を FK で要求するため、その FK 先として使う。
    依存解決の流れは: db → user → video → job。
    """
    return job_repo.create(db=db, video_id=video.id)


@pytest.fixture
def clip(db: Session, video: Video, job: Job) -> Clip:
    """`video` + `job` に紐づいたテスト用 clip を作る fixture。

    Clip は video_id / job_id の両方を FK に持つため、video と job の双方が要る。
    依存解決の流れは: db → user → video → job → clip。
    """
    return clip_repo.create(
        db=db,
        video_id=video.id,
        job_id=job.id,
        start_time=0.0,
        end_time=10.0,
        storage_path="clips/test.mp4",
    )


@pytest.fixture
def notification_log(db: Session, user: User, job: Job) -> NotificationLog:
    """`user` + `job` に紐づいたテスト用 notification_log を作る fixture。

    NotificationLog は user_id / job_id を FK に持つ。
    依存解決の流れは: db → user → video → job → notification_log。
    """
    return notification_log_repo.create(
        db=db,
        user_id=user.id,
        job_id=job.id,
        email="test@example.com",
    )
