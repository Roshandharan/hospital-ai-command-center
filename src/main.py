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
    # Warm up pipeline in thread pool so it doesn't block startup
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _init_pipeline_background)
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from src.api.command_center import router
app.include_router(router)
