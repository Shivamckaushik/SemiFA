"""FA Report generator — builds structured dict and renders PDF.

Output structure mirrors industry standard FA report format:
  - Header (lot, wafer, equipment, timestamp)
  - Executive Summary
  - Defect Description (with image)
  - Root Cause Analysis
  - Severity Assessment
  - Recipe / Corrective Action Recommendations
  - Appendix: Equipment Logs, Similar Defects
"""

from __future__ import annotations

import io
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    HRFlowable,
    PageBreak,
)

from src.config import settings

# ---------------------------------------------------------------------------
# JSON / dict builder
# ---------------------------------------------------------------------------

def build_fa_report(state: dict[str, Any]) -> dict[str, Any]:
    """Construct a structured FA report dict from the final pipeline state."""
    now = datetime.now(timezone.utc).isoformat()
    return {
        "report_id": str(uuid.uuid4()),
        "generated_at": now,
        "header": {
            "equipment_id": state.get("equipment_id", "N/A"),
            "lot_id": state.get("lot_id", "N/A"),
            "wafer_id": state.get("wafer_id", "N/A"),
            "image_modality": state.get("image_modality", "N/A"),
            "image_object_key": state.get("image_object_key", ""),
        },
        "defect": {
            "class": state.get("defect_class", "unknown"),
            "confidence": state.get("defect_confidence", 0.0),
            "description": state.get("defect_description", ""),
            "wafer_map_stats": state.get("wafer_map_stats", {}),
        },
        "root_cause": {
            "hypotheses": state.get("root_cause_hypotheses", []),
            "summary": state.get("root_cause_summary", ""),
            "similar_defect_count": len(state.get("similar_historical_defects", [])),
        },
        "severity": {
            "level": state.get("severity", "MINOR"),
            "yield_impact_pct": state.get("yield_impact_pct", 0.0),
            "reasoning": state.get("severity_reasoning", ""),
        },
        "recommendations": {
            "actions": state.get("recipe_recommendations", []),
            "process_parameters": state.get("process_parameter_adjustments", {}),
        },
        "pipeline_meta": {
            "elapsed_seconds": state.get("elapsed_seconds", 0.0),
            "errors": state.get("errors", []),
        },
    }


# ---------------------------------------------------------------------------
# PDF renderer
# ---------------------------------------------------------------------------

# Severity colour map
_SEV_COLOURS = {
    "CRITICAL": colors.HexColor("#C0392B"),
    "MAJOR": colors.HexColor("#E67E22"),
    "MINOR": colors.HexColor("#F1C40F"),
    "NONE": colors.HexColor("#27AE60"),
}


def save_report_pdf(report: dict[str, Any], output_dir: str | None = None) -> str:
    """Render FA report dict to PDF; returns file path."""
    out_dir = Path(output_dir or settings.report_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report_id = report.get("report_id", str(uuid.uuid4()))
    filename = out_dir / f"FA_Report_{report_id[:8]}.pdf"

    doc = SimpleDocTemplate(
        str(filename),
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
    )

    styles = getSampleStyleSheet()
    story = _build_story(report, styles)
    doc.build(story)
    return str(filename)


def _build_story(report: dict, styles: Any) -> list:
    story = []
    h = report.get("header", {})
    defect = report.get("defect", {})
    rc = report.get("root_cause", {})
    sev = report.get("severity", {})
    recs = report.get("recommendations", {})
    meta = report.get("pipeline_meta", {})

    severity_level = sev.get("level", "MINOR").upper()
    sev_colour = _SEV_COLOURS.get(severity_level, colors.grey)

    title_style = ParagraphStyle(
        "FATitle",
        parent=styles["Title"],
        fontSize=18,
        spaceAfter=4 * mm,
    )
    heading_style = ParagraphStyle(
        "FAHeading",
        parent=styles["Heading2"],
        fontSize=13,
        spaceBefore=6 * mm,
        spaceAfter=2 * mm,
        textColor=colors.HexColor("#2C3E50"),
    )
    normal = styles["Normal"]
    body_style = ParagraphStyle("FABody", parent=normal, fontSize=10, leading=14)

    # ── Title ────────────────────────────────────────────────────────────────
    story.append(Paragraph("Failure Analysis Report", title_style))
    story.append(Paragraph(f"Report ID: {report.get('report_id', '')[:8]}", normal))
    story.append(Paragraph(f"Generated: {report.get('generated_at', '')}", normal))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2C3E50")))
    story.append(Spacer(1, 4 * mm))

    # ── Header table ─────────────────────────────────────────────────────────
    story.append(Paragraph("Inspection Details", heading_style))
    header_data = [
        ["Equipment ID", h.get("equipment_id", "N/A"),
         "Lot ID", h.get("lot_id", "N/A")],
        ["Wafer ID", h.get("wafer_id", "N/A"),
         "Modality", h.get("image_modality", "N/A")],
    ]
    t = Table(header_data, colWidths=[40 * mm, 50 * mm, 30 * mm, 50 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#ECF0F1")),
        ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#ECF0F1")),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#BDC3C7")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t)
    story.append(Spacer(1, 4 * mm))

    # ── Severity banner ───────────────────────────────────────────────────────
    story.append(Paragraph("Severity Assessment", heading_style))
    sev_data = [
        [
            Paragraph(f"<b>SEVERITY: {severity_level}</b>", body_style),
            Paragraph(
                f"Yield Impact: <b>{sev.get('yield_impact_pct', 0):.1f}%</b>",
                body_style,
            ),
        ]
    ]
    sev_table = Table(sev_data, colWidths=[85 * mm, 85 * mm])
    sev_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), sev_colour),
        ("TEXTCOLOR", (0, 0), (0, 0), colors.white),
        ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#F8F9FA")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#BDC3C7")),
        ("PADDING", (0, 0), (-1, -1), 6),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
    ]))
    story.append(sev_table)
    story.append(Paragraph(sev.get("reasoning", ""), body_style))
    story.append(Spacer(1, 3 * mm))

    # ── Defect Description ────────────────────────────────────────────────────
    story.append(Paragraph("Defect Description", heading_style))
    story.append(Paragraph(
        f"<b>Class:</b> {defect.get('class', 'unknown')} "
        f"(confidence: {defect.get('confidence', 0):.1%})",
        body_style,
    ))
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph(defect.get("description", ""), body_style))

    wm = defect.get("wafer_map_stats", {})
    if wm:
        story.append(Spacer(1, 2 * mm))
        story.append(Paragraph("<b>Wafer Map Analysis:</b>", body_style))
        wm_data = [
            ["Pattern", wm.get("pattern_class", "N/A"),
             "Density", f"{wm.get('defect_density', 0):.2%}"],
            ["Clusters", str(wm.get("cluster_count", "N/A")),
             "Distribution", wm.get("radial_distribution", "N/A")],
        ]
        wmt = Table(wm_data, colWidths=[35 * mm, 50 * mm, 30 * mm, 55 * mm])
        wmt.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#ECF0F1")),
            ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#ECF0F1")),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#BDC3C7")),
            ("PADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(wmt)

    # ── Root Cause Analysis ───────────────────────────────────────────────────
    story.append(Paragraph("Root Cause Analysis", heading_style))
    story.append(Paragraph(
        f"<b>Summary:</b> {rc.get('summary', 'N/A')}", body_style
    ))
    story.append(Spacer(1, 2 * mm))
    hypotheses = rc.get("hypotheses", [])
    if hypotheses:
        story.append(Paragraph("<b>Root Cause Hypotheses:</b>", body_style))
        for i, hyp in enumerate(hypotheses, 1):
            story.append(Paragraph(f"  {i}. {hyp}", body_style))
    story.append(Paragraph(
        f"(Based on {rc.get('similar_defect_count', 0)} historically similar defects in vector DB)",
        ParagraphStyle("small", parent=normal, fontSize=8, textColor=colors.grey),
    ))

    # ── Recommendations ───────────────────────────────────────────────────────
    story.append(Paragraph("Corrective Action Recommendations", heading_style))
    actions = recs.get("actions", [])
    for i, action in enumerate(actions, 1):
        story.append(Paragraph(f"  {i}. {action}", body_style))

    params = recs.get("process_parameters", {})
    if params:
        story.append(Spacer(1, 2 * mm))
        story.append(Paragraph("<b>Process Parameter Adjustments:</b>", body_style))
        param_data = [[k, v] for k, v in params.items()]
        pt = Table(param_data, colWidths=[80 * mm, 90 * mm])
        pt.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#ECF0F1")),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#BDC3C7")),
            ("PADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(pt)

    # ── Pipeline meta ─────────────────────────────────────────────────────────
    story.append(Spacer(1, 6 * mm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
    story.append(Paragraph(
        f"Semiconductor FA System — Analysis completed in "
        f"{meta.get('elapsed_seconds', 0):.1f}s",
        ParagraphStyle("footer", parent=normal, fontSize=8, textColor=colors.grey),
    ))

    return story
