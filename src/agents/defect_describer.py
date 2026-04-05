"""Agent 1 — Defect Describer.

Responsibilities:
  1. Run DINOv2 defect classifier to get class + confidence + embedding
  2. Run WaferMapAnalyzer (if wafer_map modality)
  3. Query LLaVA-1.6 for natural-language defect description
  4. Populate FAState with results
"""

from __future__ import annotations

import logging
from typing import Any

from PIL import Image

from src.agents.state import FAState
from src.models.model_registry import registry
from src.vision.image_preprocessor import ImageModality

logger = logging.getLogger(__name__)

# ── LLaVA prompts ────────────────────────────────────────────────────────────

_DESCRIBE_PROMPT = (
    "Examine this semiconductor inspection image carefully. "
    "Describe:\n"
    "1. The type and morphology of any defects visible.\n"
    "2. The spatial location on the die or wafer.\n"
    "3. Any surface texture anomalies (contamination, cracks, scratches).\n"
    "4. Estimated defect size category (micro / meso / macro).\n"
    "Be specific and use semiconductor FA terminology."
)

_WAFER_MAP_PROMPT = (
    "This is a wafer map showing defect distribution across a 300mm wafer. "
    "Describe the defect pattern: its spatial distribution, density, and "
    "what process-level failure mode it likely indicates."
)


def defect_describer_node(state: FAState) -> FAState:
    """LangGraph node: Defect Describer."""
    image: Image.Image = state["image"]
    modality = state.get("image_modality", "optical")
    errors: list[str] = list(state.get("errors", []))

    try:
        # ── Step 1: DINOv2 classification ─────────────────────────────────
        det_result = registry.detector.detect(image)
        defect_class = det_result["defect_class"]
        confidence = det_result["confidence"]
        embedding = det_result["embedding"]
        logger.info(
            "DefectDescriber: class=%s confidence=%.3f", defect_class, confidence
        )

        # ── Step 2: Wafer map spatial stats ───────────────────────────────
        wafer_stats: dict[str, Any] = {}
        if modality == ImageModality.WAFER_MAP:
            stats_obj = registry.wafer_analyzer.analyze(image)
            wafer_stats = {
                "pattern_class": stats_obj.pattern_class,
                "defect_density": stats_obj.defect_density,
                "defect_count": stats_obj.defect_count,
                "cluster_count": stats_obj.cluster_count,
                "radial_distribution": stats_obj.radial_distribution,
                "spatial_entropy": stats_obj.spatial_entropy,
            }
            logger.info("WaferMap pattern: %s", stats_obj.pattern_class)

        # ── Step 3: LLaVA natural-language description ───────────────────
        prompt = (
            _WAFER_MAP_PROMPT
            if modality == ImageModality.WAFER_MAP
            else _DESCRIBE_PROMPT
        )
        description = registry.llava.query(image, prompt, max_new_tokens=400)
        logger.info("DefectDescriber: description generated (%d chars)", len(description))

    except Exception as exc:
        logger.exception("DefectDescriber failed: %s", exc)
        errors.append(f"DefectDescriber: {exc}")
        return {
            **state,
            "defect_description": "Error during defect description.",
            "defect_class": "unknown",
            "defect_confidence": 0.0,
            "defect_embedding": [],
            "wafer_map_stats": {},
            "errors": errors,
        }

    return {
        **state,
        "defect_description": description,
        "defect_class": defect_class,
        "defect_confidence": confidence,
        "defect_embedding": embedding,
        "wafer_map_stats": wafer_stats,
        "errors": errors,
    }
