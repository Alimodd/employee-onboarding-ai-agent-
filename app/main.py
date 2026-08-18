"""FastAPI application entry point (Day 23).

Run with:  uvicorn app.main:app --reload

This is the full agent-based API. The legacy Day 1-7 prompt-only app still lives
in the root ``main.py`` for reference; this one supersedes it.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.config import configure_logging
from app.database import tickets

configure_logging()
logger = logging.getLogger("app")


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Ensure the SQLite ticket table exists before serving requests."""
    logger.info("Application startup: initializing ticket database.")
    tickets.init_db()
    yield


app = FastAPI(
    title="Employee Onboarding AI Agent",
    description="Educational RAG + tool-calling agent over synthetic HR policies.",
    version="1.0.0",
    lifespan=lifespan,
)
app.include_router(router)
