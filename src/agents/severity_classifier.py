"""Agent 3 — Severity Classifier.

Classifies defect severity as: critical | major | minor | none
and estimates yield impact %.

Uses LLaVA with image + defect context for final classification.
"""

from __future__ import annotations

import logging
import re

from src.agents.state import FAState
from src.models.model_registry import registry

logger = logging.getLogger(__name__)

_SEVERITY_PROMPT_TEMPLATE = """
You are a semiconductor yield engineer.

DEFECT INFORMATION:
- Type: {defect_class}
- Description: {description}
- Root cause summary: {root_cause}
- Wafer map stats: {wafer_stats}

Classify the defect severity for this semiconductor inspection:

SEVERITY LEVELS:
- CRITICAL: Will cause immediate device failure; yield impact >15%
- MAJOR: Likely to cause parametric failure; yield impact 5-15%
- MINOR: May cause marginal performance degradation; yield impact <5%
- NONE: Cosmetic or within process spec; yield impact ~0%

Provide:
SEVERITY: <CRITICAL|MAJOR|MINOR|NONE>
YIELD_IMPACT_PCT: <number between 0 and 100>
REASONING: <2-3 sentence explanation referencing the image and defect characteristics>
"""


def _parse_severity_response(
    response: str,
) -> tuple[str, float, str]:
    """Parse SEVERITY, YIELD_IMPACT_PCT, REASONING from LLaVA text."""
    severity = "MINOR"
    yield_pct = 0.0
    reasoning = response

    for line in response.splitlines():
        line = line.strip()
        if line.upper().startswith("SEVERITY:"):
            val = line.split(":", 1)[-1].strip().upper()
            if val in {"CRITICAL", "MAJOR", "MINOR", "NONE"}:
                severity = val
        elif line.upper().startswith("YIELD_IMPACT_PCT:"):
            try:
                nums = re.findall(r"[\d.]+", line)
                if nums:
                    yield_pct = min(100.0, float(nums[0]))
            except ValueError:
                pass
        elif line.upper().startswith("REASONING:"):
            reasoning = line.split(":", 1)[-1].strip()

    return severity, yield_pct, reasoning


def severity_classifier_node(state: FAState) -> FAState:
    """LangGraph node: Severity Classifier."""
    errors: list[str] = list(state.get("errors", []))
    image = state["image"]
    defect_class = state.get("defect_class", "unknown")
    description = state.get("defect_description", "")
    root_cause = state.get("root_cause_summary", "")
    wafer_stats = state.get("wafer_map_stats", {})

    wafer_stats_str = (
        f"Pattern={wafer_stats.get('pattern_class', 'N/A')}, "
        f"density={wafer_stats.get('defect_density', 0):.1%}, "
        f"clusters={wafer_stats.get('cluster_count', 'N/A')}"
        if wafer_stats else "N/A"
    )

    try:
        prompt = _SEVERITY_PROMPT_TEMPLATE.format(
            defect_class=defect_class,
            description=description[:300],
            root_cause=root_cause[:200],
            wafer_stats=wafer_stats_str,
        )
        response = registry.llava.query(image, prompt, max_new_tokens=300, temperature=0.1)
        severity, yield_pct, reasoning = _parse_severity_response(response)
        logger.info("Severity: %s (yield impact %.1f%%)", severity, yield_pct)
    except Exception as exc:
        logger.exception("Severity classification failed: %s", exc)
        errors.append(f"SeverityClassifier: {exc}")
        severity, yield_pct, reasoning = "MINOR", 0.0, "Classification error."

    return {
        **state,
        "severity": severity,
        "severity_reasoning": reasoning,
        "yield_impact_pct": yield_pct,
        "errors": errors,
    }
