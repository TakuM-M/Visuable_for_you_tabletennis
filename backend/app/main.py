import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import auth, clips, jobs, users, videos

app = FastAPI(
    title="Visuable for You Table Tennis API",
    version="0.1.0",
)

allowed_origins = ["http://localhost:5173"]
frontend_url = os.getenv("FRONTEND_URL")
if frontend_url:
    allowed_origins.append(frontend_url)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(users.router)
app.include_router(videos.router)
app.include_router(jobs.router)
app.include_router(clips.router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}