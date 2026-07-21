"""FastAPI application entry point."""

from __future__ import annotations

from fastapi import FastAPI

from src.api.routes import router
from src.logging_config import configure_logging

configure_logging()

app = FastAPI(
    title="Financial RAG API",
    description="Multi-agent RAG system for SEC filing analysis",
    version="0.1.0",
)
app.include_router(router)
