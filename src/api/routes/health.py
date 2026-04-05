"""Health-check endpoint."""

from __future__ import annotations

import logging

from fastapi import APIRouter

from src.api.schemas import HealthResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/health", tags=["Health"])


@router.get("", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    from src.models.model_registry import registry
    from src.config import settings

    models_loaded = registry._initialised if hasattr(registry, "_initialised") else False

    qdrant_ok = False
    try:
        from src.data.qdrant_store import DefectVectorStore
        DefectVectorStore()
        qdrant_ok = True
    except Exception:
        pass

    minio_ok = False
    try:
        from src.data.minio_store import ImageStore
        ImageStore()
        minio_ok = True
    except Exception:
        pass

    timescale_ok = False
    try:
        import asyncpg, asyncio
        conn = await asyncpg.connect(
            f"postgresql://{settings.timescale_user}:{settings.timescale_password}"
            f"@{settings.timescale_host}:{settings.timescale_port}/{settings.timescale_db}"
        )
        await conn.close()
        timescale_ok = True
    except Exception:
        pass

    all_ok = models_loaded and qdrant_ok and minio_ok and timescale_ok
    return HealthResponse(
        status="ok" if all_ok else "degraded",
        models_loaded=models_loaded,
        qdrant_ok=qdrant_ok,
        minio_ok=minio_ok,
        timescaledb_ok=timescale_ok,
    )
