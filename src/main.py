"""
Hospital AI Command Center
FastAPI application entry point.
"""
import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

log = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from src.api.command_center import load_data, live_feed_loop
    log.info("app.startup", version="2.0.0")
    load_data()
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
    description="Real-time hospital operations and AI-powered clinical decision support",
    version="2.0.0",
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
