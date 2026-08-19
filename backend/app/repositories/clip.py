import uuid

from sqlalchemy.orm import Session

from app.models.clip import Clip
from app.repositories.protocols import ClipRepository


class ClipRepositoryImpl:
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
        clip = Clip(
            video_id=video_id,
            job_id=job_id,
            start_time=start_time,
            end_time=end_time,
            sort_order=sort_order,
            storage_path=storage_path,
        )
        db.add(clip)
        db.commit()
        db.refresh(clip)
        return clip

    def get_by_video_id(self, db: Session, video_id: uuid.UUID) -> list[Clip]:
        return (
            db.query(Clip)
            .filter(Clip.video_id == video_id)
            .order_by(Clip.sort_order)
            .all()
        )

    def get_by_job_id(self, db: Session, job_id: uuid.UUID) -> list[Clip]:
        return db.query(Clip).filter(Clip.job_id == job_id).all()

    def delete_by_video_id(self, db: Session, video_id: uuid.UUID) -> int:
        clips = self.get_by_video_id(db, video_id)
        count = len(clips)
        for clip in clips:
            db.delete(clip)
        db.commit()
        return count

    def replace_for_video(
        self,
        db: Session,
        video_id: uuid.UUID,
        job_id: uuid.UUID,
        clips_data: list[dict],
    ) -> list[Clip]:
        """動画に紐づく既存 clip を全削除し、与えられた区間で作り直す。

        sort_order は clips_data の並び順（0 始まり）で採番する。
        削除と再作成を 1 トランザクションで行い、中間状態を残さない。
        clips_data は {"start_time": float, "end_time": float} のリスト。
        """
        for clip in db.query(Clip).filter(Clip.video_id == video_id).all():
            db.delete(clip)

        new_clips: list[Clip] = []
        for i, data in enumerate(clips_data):
            clip = Clip(
                video_id=video_id,
                job_id=job_id,
                start_time=data["start_time"],
                end_time=data["end_time"],
                sort_order=i,
                storage_path="",
            )
            db.add(clip)
            new_clips.append(clip)

        db.commit()
        for clip in new_clips:
            db.refresh(clip)
        return new_clips


clip_repository: ClipRepository = ClipRepositoryImpl()
