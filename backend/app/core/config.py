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
    # /admin/* と内部コールバックの認証で共有
    internal_api_key: str = ""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
