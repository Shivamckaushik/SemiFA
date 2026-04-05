"""Agent 4 — Recipe Advisor.

Suggests process recipe adjustments and corrective actions based on
defect type, root cause, and equipment context.
"""

from __future__ import annotations

import logging
from typing import Any

from src.agents.state import FAState
from src.models.model_registry import registry

logger = logging.getLogger(__name__)

_RECIPE_PROMPT_TEMPLATE = """
You are a senior process integration engineer in a semiconductor fab.

DEFECT ANALYSIS SUMMARY:
- Defect type: {defect_class}
- Severity: {severity}
- Root cause: {root_cause}
- Equipment ID: {equipment_id}
- Yield impact: {yield_impact:.1f}%

Recent equipment parameter deviations:
{param_deviations}

Based on this failure analysis, provide:

RECIPE_RECOMMENDATIONS:
1. <immediate corrective action>
2. <process parameter adjustment with specific values>
3. <preventive action for future lots>
4. <monitoring/SPC recommendation>

PROCESS_PARAMETERS:
- parameter_name: suggested_value (units)
(list key parameters to adjust)

PRIORITY: <IMMEDIATE_HOLD | NEXT_MAINTENANCE | MONITOR>
"""


def _extract_recommendations(response: str) -> tuple[list[str], dict[str, Any], str]:
    """Parse recommendations, parameter adjustments, and priority."""
    lines = response.splitlines()
    recommendations: list[str] = []
    params: dict[str, Any] = {}
    priority = "MONITOR"
    in_recs = False
    in_params = False

    for line in lines:
        stripped = line.strip()
        if stripped.upper().startswith("RECIPE_RECOMMENDATIONS"):
            in_recs = True
            in_params = False
            continue
        if stripped.upper().startswith("PROCESS_PARAMETERS"):
            in_recs = False
            in_params = True
            continue
        if stripped.upper().startswith("PRIORITY"):
            in_recs = False
            in_params = False
            pval = stripped.split(":", 1)[-1].strip().upper()
            if pval in {"IMMEDIATE_HOLD", "NEXT_MAINTENANCE", "MONITOR"}:
                priority = pval
            continue

        if in_recs and stripped and stripped[0].isdigit():
            rec = stripped.lstrip("0123456789. ").strip()
            if rec:
                recommendations.append(rec)
        elif in_params and ":" in stripped:
            parts = stripped.lstrip("-").split(":", 1)
            if len(parts) == 2:
                params[parts[0].strip()] = parts[1].strip()

    if not recommendations:
        recommendations = [response[:300]]
    return recommendations, params, priority


def _extract_param_deviations(logs: list[dict]) -> str:
    """Summarise alarm codes and parameter deviations from equipment logs."""
    if not logs:
        return "No recent deviations detected."
    alarms = [
        f"  {l.get('time', '')} | {l.get('parameter_name', '')}={l.get('value', '')} [ALARM: {l.get('alarm_code', '')}]"
        for l in logs
        if l.get("alarm_code")
    ]
    return "\n".join(alarms[:5]) if alarms else "No alarms; all parameters within spec."


def recipe_advisor_node(state: FAState) -> FAState:
    """LangGraph node: Recipe Advisor."""
    errors: list[str] = list(state.get("errors", []))
    image = state["image"]

    try:
        prompt = _RECIPE_PROMPT_TEMPLATE.format(
            defect_class=state.get("defect_class", "unknown"),
            severity=state.get("severity", "MINOR"),
            root_cause=state.get("root_cause_summary", ""),
            equipment_id=state.get("equipment_id", "N/A"),
            yield_impact=state.get("yield_impact_pct", 0.0),
            param_deviations=_extract_param_deviations(
                state.get("equipment_logs", [])
            ),
        )
        response = registry.llava.query(image, prompt, max_new_tokens=500, temperature=0.1)
        recommendations, params, priority = _extract_recommendations(response)
        logger.info(
            "RecipeAdvisor: %d recommendations, priority=%s",
            len(recommendations),
            priority,
        )
    except Exception as exc:
        logger.exception("RecipeAdvisor failed: %s", exc)
        errors.append(f"RecipeAdvisor: {exc}")
        recommendations = ["Manual review required."]
        params = {}
        priority = "MONITOR"

    return {
        **state,
        "recipe_recommendations": recommendations,
        "process_parameter_adjustments": params,
        "errors": errors,
    }
