"""Agent 2 — Root Cause Analyzer.

Responsibilities:
  1. Fetch recent SECS/GEM equipment logs from TimescaleDB
  2. Query Qdrant for historically similar defects
  3. Correlate image description + logs + history via LLaVA
  4. Generate ranked root cause hypotheses
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from src.agents.state import FAState
from src.data.qdrant_store import DefectVectorStore
from src.data.timescaledb_client import TelemetryDB
from src.models.model_registry import registry

logger = logging.getLogger(__name__)

_ROOT_CAUSE_PROMPT_TEMPLATE = """
You are a semiconductor failure analysis expert.

DEFECT DESCRIPTION:
{description}

DEFECT CLASS: {defect_class} (confidence: {confidence:.1%})

RECENT EQUIPMENT LOGS (last 2 hours):
{equipment_logs}

HISTORICALLY SIMILAR DEFECTS:
{similar_defects}

Based on all evidence above, generate:
1. Top 3 ranked root cause hypotheses (most likely first)
2. Supporting evidence for each hypothesis
3. A concise root cause summary (2-3 sentences)

Format your response as:
HYPOTHESIS 1: <cause>
EVIDENCE: <supporting evidence>

HYPOTHESIS 2: <cause>
EVIDENCE: <supporting evidence>

HYPOTHESIS 3: <cause>
EVIDENCE: <supporting evidence>

ROOT CAUSE SUMMARY: <concise summary>
"""


def _format_logs(logs: list[dict]) -> str:
    if not logs:
        return "No equipment logs available."
    lines = []
    for log in logs[:10]:  # limit to 10 entries for prompt length
        ts = log.get("time", "")
        param = log.get("parameter_name", "")
        val = log.get("value", "")
        alarm = log.get("alarm_code", "")
        alarm_str = f" [ALARM: {alarm}]" if alarm else ""
        lines.append(f"  {ts} | {param}={val}{alarm_str}")
    return "\n".join(lines)


def _format_similar(similar: list[dict]) -> str:
    if not similar:
        return "No historical matches found."
    lines = []
    for i, s in enumerate(similar[:5], 1):
        score = s.get("score", 0)
        dtype = s.get("defect_type", "unknown")
        rc = s.get("root_cause", "N/A")
        lines.append(f"  {i}. [{score:.2f}] {dtype} — Root cause: {rc}")
    return "\n".join(lines)


def _parse_hypotheses(llava_response: str) -> tuple[list[str], str]:
    """Extract hypothesis list and summary from LLaVA text response."""
    lines = llava_response.strip().split("\n")
    hypotheses = []
    summary = ""
    for line in lines:
        if line.startswith("HYPOTHESIS"):
            hyp = line.split(":", 1)[-1].strip()
            if hyp:
                hypotheses.append(hyp)
        elif line.startswith("ROOT CAUSE SUMMARY"):
            summary = line.split(":", 1)[-1].strip()
    if not hypotheses:
        hypotheses = [llava_response[:200]]
    if not summary:
        summary = hypotheses[0] if hypotheses else llava_response[:100]
    return hypotheses, summary


def root_cause_analyzer_node(state: FAState) -> FAState:
    """LangGraph node: Root Cause Analyzer."""
    errors: list[str] = list(state.get("errors", []))
    equipment_id = state.get("equipment_id", "UNKNOWN")
    description = state.get("defect_description", "")
    defect_class = state.get("defect_class", "unknown")
    confidence = state.get("defect_confidence", 0.0)
    embedding = state.get("defect_embedding", [])

    # ── Fetch equipment logs (sync wrapper around async client) ───────────

    equipment_logs: list[dict[str, Any]] = []
    try:
        db = TelemetryDB()
        since = datetime.now(timezone.utc) - timedelta(hours=2)

        async def _fetch() -> list[dict]:
            await db.connect()
            try:
                return await db.fetch_alarm_history(equipment_id, since)
            finally:
                await db.close()

        equipment_logs = asyncio.run(_fetch())
        logger.info("RootCause: fetched %d equipment log entries", len(equipment_logs))
    except Exception as exc:
        logger.warning("Failed to fetch equipment logs: %s", exc)
        errors.append(f"RootCause/DB: {exc}")

    # ── Qdrant similarity search ──────────────────────────────────────────

    similar: list[dict[str, Any]] = []
    if embedding:
        try:
            store = DefectVectorStore()
            similar = store.search_similar(embedding, top_k=5)
            logger.info("RootCause: found %d similar historical defects", len(similar))
        except Exception as exc:
            logger.warning("Failed to query Qdrant: %s", exc)
            errors.append(f"RootCause/Qdrant: {exc}")

    # ── LLaVA root cause reasoning ────────────────────────────────────────

    try:
        prompt = _ROOT_CAUSE_PROMPT_TEMPLATE.format(
            description=description,
            defect_class=defect_class,
            confidence=confidence,
            equipment_logs=_format_logs(equipment_logs),
            similar_defects=_format_similar(similar),
        )
        # For root cause we use text-only (no image needed at this stage)
        # Reuse the LLaVA engine with a placeholder 1×1 image
        from PIL import Image as PILImage
        blank = PILImage.new("RGB", (4, 4), color=(128, 128, 128))
        response = registry.llava.query(blank, prompt, max_new_tokens=600)
        hypotheses, summary = _parse_hypotheses(response)
        logger.info("RootCause: %d hypotheses generated", len(hypotheses))
    except Exception as exc:
        logger.exception("LLaVA root cause analysis failed: %s", exc)
        errors.append(f"RootCause/LLaVA: {exc}")
        hypotheses = ["Unable to generate root cause — model error."]
        summary = "Model inference error."

    return {
        **state,
        "equipment_logs": equipment_logs,
        "similar_historical_defects": similar,
        "root_cause_hypotheses": hypotheses,
        "root_cause_summary": summary,
        "errors": errors,
    }
