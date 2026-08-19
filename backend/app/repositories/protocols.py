import uuid
from datetime import datetime
from typing import Protocol

from sqlalchemy.orm import Session

from app.models.clip import Clip
from app.models.job import Job, JobStatus
from app.models.notification_log import NotificationLog, NotificationStatus
from app.models.user import User
from app.models.video import Video, VideoStatus


class UserRepository(Protocol):
    def create(
        self, db: Session, email: str, password_hash: str, display_name: str
    ) -> User:
        ...

    def get_by_id(self, db: Session, user_id: uuid.UUID) -> User | None:
        ...

    def get_by_email(self, db: Session, email: str) -> User | None:
        ...

    def verify_email(self, db: Session, user_id: uuid.UUID) -> None:
        ...

    def update(
        self, db: Session, user_id: uuid.UUID, display_name: str, password_hash: str
    ) -> User | None:
        ...


class JobRepository(Protocol):
    def create(self, db: Session, video_id: uuid.UUID) -> Job:
        ...

    def get_by_id(self, db: Session, job_id: uuid.UUID) -> Job | None:
        ...

    def get_by_video_id(self, db: Session, video_id: uuid.UUID) -> list[Job]:
        ...

    def get_latest_by_video_id(self, db: Session, video_id: uuid.UUID) -> Job | None:
        ...

    def update_status(
        self,
        db: Session,
        job_id: uuid.UUID,
        status: JobStatus,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
        error_message: str | None = None,
    ) -> Job | None:
        ...

    def set_runpod_job_id(
        self, db: Session, job_id: uuid.UUID, runpod_job_id: str
    ) -> Job | None:
        ...

    def get_running_runpod_jobs(self, db: Session) -> list[Job]:
        ...

    def mark_failed(
        self,
        db: Session,
        job_id: uuid.UUID,
        error_message: str,
        next_retry_at: datetime | None,
    ) -> Job | None:
        ...

    def get_timed_out_jobs(self, db: Session, threshold: datetime) -> list[Job]:
        ...

    def get_queued_started_null_jobs(self, db: Session) -> list[Job]:
        ...

    def get_jobs_ready_for_retry(
        self, db: Session, now: datetime, max_retries: int
    ) -> list[Job]:
        ...

    def prepare_for_auto_retry(self, db: Session, job_id: uuid.UUID) -> Job | None:
        ...

    def reset_for_manual_retry(self, db: Session, job_id: uuid.UUID) -> Job | None:
        ...

    def delete_by_video_id(self, db: Session, video_id: uuid.UUID) -> int:
        ...


class VideoRepository(Protocol):
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
        ...

    def get_by_id(self, db: Session, video_id: uuid.UUID) -> Video | None:
        ...

    def get_by_user_id(self, db: Session, user_id: uuid.UUID) -> list[Video]:
        ...

    def count_by_user_id(self, db: Session, user_id: uuid.UUID) -> int:
        ...

    def get_expired(self, db: Session, threshold: datetime) -> list[Video]:
        ...

    def get_processing_without_running_job(self, db: Session) -> list[Video]:
        ...

    def update_status(
        self,
        db: Session,
        video_id: uuid.UUID,
        status: VideoStatus,
    ) -> Video | None:
        ...

    def update_output_path(
        self,
        db: Session,
        video_id: uuid.UUID,
        output_path: str,
    ) -> Video | None:
        ...

    def update_duration(
        self,
        db: Session,
        video_id: uuid.UUID,
        duration: float,
    ) -> Video | None:
        ...

    def update_thumbnail_path(
        self,
        db: Session,
        video_id: uuid.UUID,
        thumbnail_path: str,
    ) -> Video | None:
        ...

    def update_source_duration(
        self,
        db: Session,
        video_id: uuid.UUID,
        source_duration: float,
    ) -> Video | None:
        ...

    def delete(self, db: Session, video_id: uuid.UUID) -> bool:
        ...


class ClipRepository(Protocol):
    def create(
        self,
        db: Session,
        video_id: uuid.UUID,
        job_id: uuid.UUID,
        start_time: float,
        end_time: float,
        storage_path: str,
        sort_order: int = 0,
    ) -> Clip:
        ...

    def get_by_video_id(self, db: Session, video_id: uuid.UUID) -> list[Clip]:
        ...

    def get_by_job_id(self, db: Session, job_id: uuid.UUID) -> list[Clip]:
        ...

    def delete_by_video_id(self, db: Session, video_id: uuid.UUID) -> int:
        ...

    def replace_for_video(
        self,
        db: Session,
        video_id: uuid.UUID,
        job_id: uuid.UUID,
        clips_data: list[dict],
    ) -> list[Clip]:
        ...


class NotificationLogRepository(Protocol):
    def create(
        self,
        db: Session,
        user_id: uuid.UUID,
        job_id: uuid.UUID,
        email: str,
    ) -> NotificationLog:
        ...

    def get_by_job_id(self, db: Session, job_id: uuid.UUID) -> list[NotificationLog]:
        ...

    def update_status(
        self,
        db: Session,
        log_id: int,
        status: NotificationStatus,
        sent_at: datetime | None = None,
    ) -> NotificationLog | None:
        ...

    def delete_by_job_id(self, db: Session, job_id: uuid.UUID) -> int:
        ...
