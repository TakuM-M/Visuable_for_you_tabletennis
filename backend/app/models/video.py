import enum
import uuid
from datetime import datetime

from sqlalchemy import UUID, Enum as SAEnum, Float, ForeignKey, String
from sqlalchemy.dialects.postgresql import TIMESTAMP
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.models.base import Base


class VideoStatus(str, enum.Enum):
    uploaded = "uploaded"
    queued = "queued"
    processing = "processing"
    ready = "ready"  # ML 解析完了・編集可能・未書き出し
    completed = "completed"  # 書き出し済み（output_path あり）
    failed = "failed"


class Video(Base):
    __tablename__ = "videos"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id")
    )
    title: Mapped[str] = mapped_column(String)
    storage_path: Mapped[str] = mapped_column(String)
    output_path: Mapped[str | None] = mapped_column(String, nullable=True)
    duration: Mapped[float | None] = mapped_column(Float, nullable=True)
    # 元動画の再生時間（秒）。編集時の区間バリデーションやタイムライン表示に使う。
    # duration は書き出し済み出力動画の長さが入るため別フィールドで保持する。
    source_duration: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[VideoStatus] = mapped_column(
        SAEnum(VideoStatus), default=VideoStatus.uploaded
    )
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    user: Mapped["User"] = relationship(back_populates="videos")
    jobs: Mapped[list["Job"]] = relationship(back_populates="video")
    clips: Mapped[list["Clip"]] = relationship(back_populates="video")
