import os
from contextlib import asynccontextmanager

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging import get_logger, setup_logging
from app.routers import admin, auth, clips, jobs, users, videos
from app.services import job_reaper, video_service

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """アプリ起動/停止時のフック。APScheduler を立ち上げて reaper を回す"""
    setup_logging()
    scheduler = AsyncIOScheduler(timezone="UTC")
    scheduler.add_job(
        job_reaper.reap_timeouts,
        "interval",
        seconds=settings.reaper_interval_seconds,
        id="reap_timeouts",
    )
    scheduler.add_job(
        job_reaper.dispatch_retries,
        "interval",
        seconds=settings.reaper_interval_seconds,
        id="dispatch_retries",
    )
    scheduler.add_job(
        job_reaper.clean_tmp_dir,
        "interval",
        seconds=settings.tmp_cleaner_interval_seconds,
        id="clean_tmp_dir",
    )
    scheduler.add_job(
        job_reaper.cleanup_expired_videos,
        "interval",
        seconds=settings.video_retention_cleanup_interval_seconds,
        id="cleanup_expired_videos",
    )
    scheduler.add_job(
        job_reaper.log_storage_metrics,
        "interval",
        seconds=settings.metrics_log_interval_seconds,
        id="log_storage_metrics",
    )
    # 起動時に tmp を一度掃除し、中断された書き出しを ready に戻す
    job_reaper.clean_tmp_dir()
    video_service.recover_interrupted_exports()
    scheduler.start()
    logger.info("APScheduler 起動")
    try:
        yield
    finally:
        scheduler.shutdown(wait=False)
        logger.info("APScheduler 停止")


app = FastAPI(
    title="Visuable for You Table Tennis API",
    version="0.1.0",
    lifespan=lifespan,
)

allowed_origins = ["http://localhost:5173"]
frontend_url = os.getenv("FRONTEND_URL")
if frontend_url:
    allowed_origins.append(frontend_url)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Internal-Api-Key"],
    allow_credentials=False,
)

app.include_router(auth.router)
app.include_router(users.router)
app.include_router(videos.router)
app.include_router(jobs.router)
app.include_router(clips.router)
app.include_router(admin.router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
