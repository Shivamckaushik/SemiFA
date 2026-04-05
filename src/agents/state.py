"""Shared LangGraph state definition for the FA pipeline."""

from __future__ import annotations

from typing import Any, TypedDict

from PIL import Image


class FAState(TypedDict, total=False):
    """State that flows through the LangGraph FA pipeline."""

    # ── Inputs ───────────────────────────────────────────────────────────────
    image: Image.Image                    # inspection image (PIL)
    image_modality: str                   # "sem" | "optical" | "wafer_map"
    equipment_id: str
    lot_id: str
    wafer_id: str
    image_object_key: str                 # MinIO object key

    # ── Defect Describer outputs ──────────────────────────────────────────────
    defect_description: str               # natural-language description
    defect_class: str                     # e.g. "scratch"
    defect_confidence: float
    defect_embedding: list[float]         # DINOv2 embedding (768-d)
    wafer_map_stats: dict[str, Any]       # from WaferMapAnalyzer

    # ── Root Cause Analyzer outputs ───────────────────────────────────────────
    equipment_logs: list[dict[str, Any]]  # fetched SECS/GEM / TimescaleDB records
    similar_historical_defects: list[dict[str, Any]]  # Qdrant nearest neighbours
    root_cause_hypotheses: list[str]
    root_cause_summary: str

    # ── Severity Classifier outputs ───────────────────────────────────────────
    severity: str                         # "critical" | "major" | "minor" | "none"
    severity_reasoning: str
    yield_impact_pct: float

    # ── Recipe Advisor outputs ────────────────────────────────────────────────
    recipe_recommendations: list[str]
    process_parameter_adjustments: dict[str, Any]

    # ── Report ────────────────────────────────────────────────────────────────
    fa_report: dict[str, Any]             # structured final report dict
    report_path: str                      # saved PDF path

    # ── Meta ──────────────────────────────────────────────────────────────────
    errors: list[str]
    elapsed_seconds: float
