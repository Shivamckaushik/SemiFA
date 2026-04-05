"""LangGraph FA pipeline orchestrator.

Wires together the four agentic nodes into a directed graph:

    [START]
       │
       ▼
  DefectDescriber          ← image + metadata
       │
       ▼
  RootCauseAnalyzer        ← description + embedding + equipment logs + Qdrant
       │
       ▼
  SeverityClassifier       ← image + defect context
       │
       ▼
  RecipeAdvisor            ← full context
       │
       ▼
  ReportGenerator          ← assemble structured FA report
       │
       ▼
    [END]

Each node returns an updated FAState dict.
Total wall-clock target: < 2 minutes.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from langgraph.graph import END, START, StateGraph

from src.agents.state import FAState
from src.agents.defect_describer import defect_describer_node
from src.agents.root_cause_analyzer import root_cause_analyzer_node
from src.agents.severity_classifier import severity_classifier_node
from src.agents.recipe_advisor import recipe_advisor_node
from src.reports.fa_report_generator import build_fa_report, save_report_pdf

logger = logging.getLogger(__name__)


# ── Report assembly node ─────────────────────────────────────────────────────

def report_generator_node(state: FAState) -> FAState:
    """Assemble the final structured FA report and persist to disk."""
    try:
        report_dict = build_fa_report(state)
        report_path = save_report_pdf(report_dict)
        logger.info("FA report saved: %s", report_path)
    except Exception as exc:
        logger.exception("ReportGenerator failed: %s", exc)
        report_dict = {"error": str(exc)}
        report_path = ""

    return {
        **state,
        "fa_report": report_dict,
        "report_path": report_path,
    }


# ── Graph construction ────────────────────────────────────────────────────────

def _build_graph() -> StateGraph:
    graph = StateGraph(FAState)

    graph.add_node("DefectDescriber", defect_describer_node)
    graph.add_node("RootCauseAnalyzer", root_cause_analyzer_node)
    graph.add_node("SeverityClassifier", severity_classifier_node)
    graph.add_node("RecipeAdvisor", recipe_advisor_node)
    graph.add_node("ReportGenerator", report_generator_node)

    graph.add_edge(START, "DefectDescriber")
    graph.add_edge("DefectDescriber", "RootCauseAnalyzer")
    graph.add_edge("RootCauseAnalyzer", "SeverityClassifier")
    graph.add_edge("SeverityClassifier", "RecipeAdvisor")
    graph.add_edge("RecipeAdvisor", "ReportGenerator")
    graph.add_edge("ReportGenerator", END)

    return graph


# Compile once at module import
_compiled_graph = _build_graph().compile()


# ── Public API ────────────────────────────────────────────────────────────────

def run_fa_pipeline(initial_state: FAState) -> FAState:
    """
    Execute the full FA pipeline synchronously.

    Args:
        initial_state: Must contain at minimum 'image', 'equipment_id',
                       'lot_id', 'wafer_id', 'image_modality'.
    Returns:
        Final FAState with all fields populated.
    """
    t0 = time.perf_counter()
    initial_state.setdefault("errors", [])

    logger.info(
        "FA pipeline START — equipment=%s lot=%s wafer=%s",
        initial_state.get("equipment_id"),
        initial_state.get("lot_id"),
        initial_state.get("wafer_id"),
    )

    final_state: FAState = _compiled_graph.invoke(initial_state)
    elapsed = round(time.perf_counter() - t0, 2)
    final_state["elapsed_seconds"] = elapsed

    logger.info(
        "FA pipeline DONE in %.1fs — severity=%s defect=%s",
        elapsed,
        final_state.get("severity"),
        final_state.get("defect_class"),
    )

    # Persist defect embedding to Qdrant for future similarity searches
    _store_embedding(final_state)

    return final_state


def _store_embedding(state: FAState) -> None:
    """Store defect embedding + FA results back into Qdrant."""
    embedding = state.get("defect_embedding")
    if not embedding:
        return
    try:
        from src.data.qdrant_store import DefectVectorStore
        store = DefectVectorStore()
        store.upsert_defect(
            embedding=embedding,
            metadata={
                "equipment_id": state.get("equipment_id", ""),
                "lot_id": state.get("lot_id", ""),
                "wafer_id": state.get("wafer_id", ""),
                "defect_type": state.get("defect_class", ""),
                "severity": state.get("severity", ""),
                "root_cause": state.get("root_cause_summary", ""),
                "image_key": state.get("image_object_key", ""),
            },
        )
    except Exception as exc:
        logger.warning("Failed to store embedding: %s", exc)
