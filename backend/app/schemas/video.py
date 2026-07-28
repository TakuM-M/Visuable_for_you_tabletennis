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
    # 保持ポリシー（settings.video_retention_days）による自動削除の予定時刻。
    # DB には持たず created_at から算出する導出値で、組み立ては
    # routers/videos.py の _to_response が一手に引き受ける。
    expires_at: datetime
    # サムネイル画像の presigned URL。未生成（生成失敗・機能追加以前の動画）なら
    # None で、フロントはプレースホルダ表示にフォールバックする。
    # <img src> は Authorization ヘッダを送れないため、R2 キーではなく
    # 署名済み URL を一覧レスポンスに同梱して追加リクエストを不要にしている。
    thumbnail_url: str | None = None


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
