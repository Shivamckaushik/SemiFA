"""Pydantic request/response schemas for the FA API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ── Request ───────────────────────────────────────────────────────────────────

class InspectionRequest(BaseModel):
    equipment_id: str = Field(example="EQ-INSP-01")
    lot_id: str = Field(example="LOT-2024-001")
    wafer_id: str = Field(example="W05")
    image_modality: str = Field(
        default="optical",
        description="sem | optical | wafer_map",
        example="optical",
    )


# ── Response ──────────────────────────────────────────────────────────────────

class DefectInfo(BaseModel):
    defect_class: str
    confidence: float
    description: str
    wafer_map_stats: dict[str, Any] = {}


class RootCauseInfo(BaseModel):
    hypotheses: list[str]
    summary: str
    similar_defect_count: int


class SeverityInfo(BaseModel):
    level: str
    yield_impact_pct: float
    reasoning: str


class RecommendationsInfo(BaseModel):
    actions: list[str]
    process_parameters: dict[str, Any]


class FAReportResponse(BaseModel):
    report_id: str
    generated_at: str
    equipment_id: str
    lot_id: str
    wafer_id: str
    defect: DefectInfo
    root_cause: RootCauseInfo
    severity: SeverityInfo
    recommendations: RecommendationsInfo
    report_pdf_path: str
    elapsed_seconds: float
    errors: list[str] = []


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    qdrant_ok: bool
    minio_ok: bool
    timescaledb_ok: bool
