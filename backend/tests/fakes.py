"""リポジトリ Protocol を満たす Fake の土台。"""

import uuid
from datetime import datetime

from sqlalchemy.orm import Session

from app.models.job import Job, JobStatus
from app.models.user import User
from app.models.video import Video, VideoStatus


class FakeUserRepository:
    """UserRepository Protocol の形だけを満たす土台"""

    def create(
        self, db: Session, email: str, password_hash: str, display_name: str
    ) -> User:
        raise NotImplementedError

    def get_by_id(self, db: Session, user_id: uuid.UUID) -> User | None:
        raise NotImplementedError

    def get_by_email(self, db: Session, email: str) -> User | None:
        raise NotImplementedError

    def verify_email(self, db: Session, user_id: uuid.UUID) -> None:
        raise NotImplementedError

    def update(
        self, db: Session, user_id: uuid.UUID, display_name: str, password_hash: str
    ) -> User | None:
        raise NotImplementedError


class FakeJobRepository:
    """JobRepository Protocol の形だけを満たす土台"""

    def create(self, db: Session, video_id: uuid.UUID) -> Job:
        raise NotImplementedError

    def get_by_id(self, db: Session, job_id: uuid.UUID) -> Job | None:
        raise NotImplementedError

    def get_by_video_id(self, db: Session, video_id: uuid.UUID) -> list[Job]:
        raise NotImplementedError

    def get_latest_by_video_id(self, db: Session, video_id: uuid.UUID) -> Job | None:
        raise NotImplementedError

    def update_status(
        self,
        db: Session,
        job_id: uuid.UUID,
        status: JobStatus,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
        error_message: str | None = None,
    ) -> Job | None:
        raise NotImplementedError

    def set_runpod_job_id(
        self, db: Session, job_id: uuid.UUID, runpod_job_id: str
    ) -> Job | None:
        raise NotImplementedError

    def get_running_runpod_jobs(self, db: Session) -> list[Job]:
        raise NotImplementedError

    def mark_failed(
        self,
        db: Session,
        job_id: uuid.UUID,
        error_message: str,
        next_retry_at: datetime | None,
    ) -> Job | None:
        raise NotImplementedError

    def get_timed_out_jobs(self, db: Session, threshold: datetime) -> list[Job]:
        raise NotImplementedError

    def get_queued_started_null_jobs(self, db: Session) -> list[Job]:
        raise NotImplementedError

    def get_jobs_ready_for_retry(
        self, db: Session, now: datetime, max_retries: int
    ) -> list[Job]:
        raise NotImplementedError

    def prepare_for_auto_retry(self, db: Session, job_id: uuid.UUID) -> Job | None:
        raise NotImplementedError

    def reset_for_manual_retry(self, db: Session, job_id: uuid.UUID) -> Job | None:
        raise NotImplementedError

    def delete_by_video_id(self, db: Session, video_id: uuid.UUID) -> int:
        raise NotImplementedError


class FakeVideoRepository:
    """VideoRepository Protocol の形だけを満たす土台"""

    def create(
        self,
        db: Session,
        user_id: uuid.UUID,
        title: str,
        storage_path: str,
        duration: float | None = None,
        source_duration: float | None = None,
        thumbnail_path: str | None = None,
    ) -> Video:
        raise NotImplementedError

    def get_by_id(self, db: Session, video_id: uuid.UUID) -> Video | None:
        raise NotImplementedError

    def get_by_user_id(self, db: Session, user_id: uuid.UUID) -> list[Video]:
        raise NotImplementedError

    def count_by_user_id(self, db: Session, user_id: uuid.UUID) -> int:
        raise NotImplementedError

    def get_expired(self, db: Session, threshold: datetime) -> list[Video]:
        raise NotImplementedError

    def get_processing_without_running_job(self, db: Session) -> list[Video]:
        raise NotImplementedError

    def update_status(
        self,
        db: Session,
        video_id: uuid.UUID,
        status: VideoStatus,
    ) -> Video | None:
        raise NotImplementedError

    def update_output_path(
        self,
        db: Session,
        video_id: uuid.UUID,
        output_path: str,
    ) -> Video | None:
        raise NotImplementedError

    def update_duration(
        self,
        db: Session,
        video_id: uuid.UUID,
        duration: float,
    ) -> Video | None:
        raise NotImplementedError

    def update_thumbnail_path(
        self,
        db: Session,
        video_id: uuid.UUID,
        thumbnail_path: str,
    ) -> Video | None:
        raise NotImplementedError

    def update_source_duration(
        self,
        db: Session,
        video_id: uuid.UUID,
        source_duration: float,
    ) -> Video | None:
        raise NotImplementedError

    def delete(self, db: Session, video_id: uuid.UUID) -> bool:
        raise NotImplementedError
