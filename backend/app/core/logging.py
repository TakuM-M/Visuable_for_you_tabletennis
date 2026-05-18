import logging

from app.core.config import settings

_configured = False


def setup_logging() -> None:
    """アプリ全体のロギング設定を一度だけ初期化する"""
    global _configured
    if _configured:
        return
    logging.basicConfig(
        level=logging.DEBUG if settings.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    _configured = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
