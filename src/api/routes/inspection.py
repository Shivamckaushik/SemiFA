"""Inspection endpoint — accepts image upload, runs FA pipeline, returns report."""

from __future__ import annotations

import io
import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image

from src.agents.orchestrator import run_fa_pipeline
from src.agents.state import FAState
from src.api.schemas import FAReportResponse
from src.data.minio_store import ImageStore

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/inspection", tags=["Inspection"])


@router.post("/analyze", response_model=FAReportResponse)
async def analyze_inspection_image(
    image: Annotated[UploadFile, File(description="Inspection image (PNG/JPEG/TIFF)")],
    equipment_id: Annotated[str, Form()] = "UNKNOWN",
    lot_id: Annotated[str, Form()] = "LOT-000",
    wafer_id: Annotated[str, Form()] = "W00",
    image_modality: Annotated[str, Form()] = "optical",
) -> FAReportResponse:
    """
    Upload an inspection image and run the full autonomous FA pipeline.

    Returns a structured FA report including defect description,
    root cause, severity, and recipe recommendations.
    """
    # ── Validate image ────────────────────────────────────────────────────
    allowed_types = {"image/png", "image/jpeg", "image/tiff", "image/bmp"}
    if image.content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported image type: {image.content_type}",
        )

    image_bytes = await image.read()
    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Cannot decode image: {exc}")

    # ── Upload to MinIO ────────────────────────────────────────────────────
    object_key = f"{equipment_id}/{lot_id}/{wafer_id}/{uuid.uuid4().hex}.png"
    try:
        store = ImageStore()
        store.upload_image(object_key, image_bytes)
        logger.info("Image stored: %s", object_key)
    except Exception as exc:
        logger.warning("MinIO upload failed (continuing): %s", exc)
        object_key = ""

    # ── Run FA pipeline ────────────────────────────────────────────────────
    initial_state: FAState = {
        "image": pil_image,
        "image_modality": image_modality,
        "equipment_id": equipment_id,
        "lot_id": lot_id,
        "wafer_id": wafer_id,
        "image_object_key": object_key,
        "errors": [],
    }

    try:
        final_state = run_fa_pipeline(initial_state)
    except Exception as exc:
        logger.exception("FA pipeline crashed: %s", exc)
        raise HTTPException(status_code=500, detail=f"FA pipeline error: {exc}")

    report = final_state.get("fa_report", {})
    defect_info = report.get("defect", {})
    rc_info = report.get("root_cause", {})
    sev_info = report.get("severity", {})
    rec_info = report.get("recommendations", {})

    return FAReportResponse(
        report_id=report.get("report_id", str(uuid.uuid4())),
        generated_at=report.get("generated_at", ""),
        equipment_id=equipment_id,
        lot_id=lot_id,
        wafer_id=wafer_id,
        defect={
            "defect_class": defect_info.get("class", "unknown"),
            "confidence": defect_info.get("confidence", 0.0),
            "description": defect_info.get("description", ""),
            "wafer_map_stats": defect_info.get("wafer_map_stats", {}),
        },
        root_cause={
            "hypotheses": rc_info.get("hypotheses", []),
            "summary": rc_info.get("summary", ""),
            "similar_defect_count": rc_info.get("similar_defect_count", 0),
        },
        severity={
            "level": sev_info.get("level", "MINOR"),
            "yield_impact_pct": sev_info.get("yield_impact_pct", 0.0),
            "reasoning": sev_info.get("reasoning", ""),
        },
        recommendations={
            "actions": rec_info.get("actions", []),
            "process_parameters": rec_info.get("process_parameters", {}),
        },
        report_pdf_path=final_state.get("report_path", ""),
        elapsed_seconds=final_state.get("elapsed_seconds", 0.0),
        errors=final_state.get("errors", []),
    )
