import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.models.video import VideoStatus


class VideoCreate(BaseModel):
    """動画アップロードリクエスト（ファイル本体は別途 multipart で受け取る）"""

    title: str


class ChunkUploadInitRequest(BaseModel):
    """チャンクアップロード初期化リクエスト"""

    title: str
    filename: str
    total_chunks: int


class ChunkUploadInitResponse(BaseModel):
    """チャンクアップロード初期化レスポンス"""

    upload_id: str


class VideoResponse(BaseModel):
    """動画情報レスポンス"""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    user_id: uuid.UUID
    title: str
    storage_path: str
    output_path: str | None
    duration: float | None
    source_duration: float | None
    status: VideoStatus
    created_at: datetime
    updated_at: datetime


class VideoOutputResponse(BaseModel):
    """連結済み動画の取得用レスポンス。

    R2 の presigned URL（短命・推測不可）を返し、フロントは <video src> /
    ダウンロード href にこの URL を直接セットする。認可はこのエンドポイントで
    行い、バイト本体は R2 から直接配信する。

    url はインライン再生用、download_url は Content-Disposition: attachment 付きで
    ブラウザに保存させる用（クロスオリジン URL では <a download> が無視されるため、
    ヘッダ側で attachment を指定しないとモバイルで再生画面が開くだけになる）。
    """

    url: str
    download_url: str | None = None
