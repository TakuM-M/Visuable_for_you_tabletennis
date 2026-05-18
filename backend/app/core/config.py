from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str
    secret_key: str
    debug: bool = False

    # ジョブ耐障害性
    job_max_retries: int = 2
    job_timeout_hours: float = 24.0
    # リトライ間隔（秒）。リスト長を超えるリトライ要求では末尾値を使う
    job_retry_backoff_seconds: list[int] = [60, 600]
    reaper_interval_seconds: int = 60
    tmp_cleaner_interval_seconds: int = 3600
    tmp_retention_hours: float = 24.0

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
