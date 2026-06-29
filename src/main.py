"""
Hospital AI Command Center
FastAPI application entry point — v3 with LangGraph + XGBoost + RAG.
"""
import asyncio
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

log = structlog.get_logger(__name__)


def _init_pipeline_background():
    """Pre-warm pipeline singletons (scoring engine + RAG) in the background."""
    try:
        from src.api.command_center import _get_pipeline
        p = _get_pipeline()
        if p:
            log.info("pipeline.warmed")
    except Exception as e:
        log.warning("pipeline.warm_failed", error=str(e))


@asynccontextmanager
async def lifespan(app: FastAPI):
    from src.api.command_center import load_data, live_feed_loop
    log.info("app.startup", version="3.0.0")
    load_data()
    # Warm up pipeline in background thread without blocking the event loop
    asyncio.create_task(asyncio.to_thread(_init_pipeline_background))
    task = asyncio.create_task(live_feed_loop())
    log.info("live_feed.scheduled")
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    log.info("app.shutdown")


app = FastAPI(
    title="Hospital AI Command Center",
    description="Real-time hospital operations with LangGraph multi-agent pipeline, XGBoost risk scoring, and ChromaDB RAG",
    version="3.0.0",
    lifespan=lifespan,
)

from src.config import get_settings as _get_settings
_cfg = _get_settings()
_cors_origins = (
    ["*"] if _cfg.environment != "production"
    else [
        "https://hospital-ai-command-center-production.up.railway.app",
        "http://localhost:8000",
    ]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Accept"],
)

from src.api.command_center import router
app.include_router(router)
