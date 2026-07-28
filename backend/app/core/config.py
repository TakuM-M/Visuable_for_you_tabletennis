from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str
    secret_key: str
    debug: bool = False

    # ジョブ耐障害性
    # 自動リトライは1回まで（初回 + リトライ1回 = 計2回実行）。
    # 失敗の多くは動画そのものに起因して何度やっても同じ結果になるため、
    # GPU を無駄に立ち上げないよう回数を絞っている
    job_max_retries: int = 1
    job_timeout_hours: float = 24.0
    # リトライ間隔（秒）。リスト長を超えるリトライ要求では末尾値を使う
    job_retry_backoff_seconds: list[int] = [60, 600]
    reaper_interval_seconds: int = 60
    tmp_cleaner_interval_seconds: int = 3600
    tmp_retention_hours: float = 24.0
    # RunPod のジョブ状態を突き合わせる間隔（秒）。コールバックが届かないまま
    # GPU が死んだケースを job_timeout_hours を待たずに検知するためのもの
    runpod_poll_interval_seconds: int = 300

    # 動画保持ポリシー・容量管理
    video_retention_days: float = 7.0
    video_retention_cleanup_interval_seconds: int = 3600
    user_video_quota: int = 10
    metrics_log_interval_seconds: int = 3600

    # 受け入れ上限（長時間・大容量動画のガード）
    # ここを超える動画は GPU 実行時間・書き出し時間・ディスクのいずれかが
    # 現実的でなくなるため、R2 に上げる前・GPU に流す前に落とす。
    # max_upload_bytes は frontend/src/lib/chunkedUpload.ts の MAX_UPLOAD_BYTES と
    # 揃えること（フロントで先に弾き、backend で最終的に担保する二段構え）
    max_upload_bytes: int = 5 * 1024**3  # 5GB
    max_video_duration_seconds: float = 3600.0  # 60分
    # 1 リクエストで受け取るチャンクの上限。フロントの CHUNK_SIZE（50MB）と
    # nginx の client_max_body_size（60M）の内側に収まる値
    max_chunk_bytes: int = 55 * 1024**2

    # ML ディスパッチ
    # 元動画 presigned URL の有効期限。RunPod のキュー待ち（コールドスタート・
    # ワーカー枯渇）とダウンロード時間を合わせて賄える長さが必要
    ml_presigned_url_expires_seconds: int = 21600  # 6時間
    # RunPod の実行時間上限（秒）。推論時間は動画長にほぼ比例するので
    # base + ratio × 動画長 で見積もる。未指定だとエンドポイント既定値が効き、
    # 長い動画が TIMED_OUT で落ちる
    runpod_execution_timeout_base_seconds: int = 900
    runpod_execution_timeout_ratio: float = 3.0
    runpod_execution_timeout_max_seconds: int = 10800  # 3時間

    # 書き出し（FFmpeg）
    # 同時に走らせる書き出し数。長い動画ほど 1 件が CPU を長時間占有するため、
    # 並走を絞らないと VPS 全体（API 応答含む）が巻き込まれる
    export_max_concurrency: int = 1
    # 順番待ちの上限。ここを超えたら ready に戻して手動で再実行してもらう
    # （背景タスクのスレッドを無期限に抱え込ませない）
    export_queue_timeout_seconds: int = 1800
    # セグメント切り出しの並列数。区間が短いほどプロセス起動・シークの
    # オーバーヘッドが支配的になるので、複数本を重ねて詰める
    ffmpeg_segment_workers: int = 2
    # FFmpeg のハング対策。到達したら書き出しを失敗させて ready に戻す
    ffmpeg_segment_timeout_seconds: int = 1800
    ffmpeg_concat_timeout_seconds: int = 3600
    # R2 からの元動画ダウンロードで「無応答」とみなすまでの秒数。
    # ダウンロード全体の上限ではなく 1 回のソケット操作の上限なので、
    # GB 級でもこの値で足りる（全体に上限を掛けると大きい動画が落ちる）
    source_download_read_timeout_seconds: float = 120.0
    # /admin/* と内部コールバックの認証で共有
    internal_api_key: str = ""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
