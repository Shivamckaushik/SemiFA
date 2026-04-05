"""FastAPI application entry-point."""

from __future__ import annotations

import logging

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

from src.api.routes import health, inspection, reports
from src.config import settings

# ── Logging ───────────────────────────────────────────────────────────────────
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Multi-Modal VLM Semiconductor FA System",
    description=(
        "Autonomous Failure Analysis report generation from semiconductor "
        "inspection images (SEM, optical, wafer map) using LLaVA-1.6 + "
        "DINOv2 + LangGraph agentic pipeline."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(health.router)
app.include_router(inspection.router)
app.include_router(reports.router)


# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event() -> None:
    logger.info("Loading models…")
    from src.models.model_registry import registry
    registry.initialise(load_llava=True, load_detector=True)
    logger.info("Models loaded. FA API ready.")


@app.get("/")
async def root() -> dict:
    return {
        "service": "Multi-Modal VLM Semiconductor FA System",
        "version": "1.0.0",
        "docs": "/docs",
    }
