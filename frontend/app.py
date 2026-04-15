"""Streamlit frontend — Multi-Modal VLM Semiconductor FA System.

Features:
  - Upload inspection image (SEM / optical / wafer map)
  - Fill equipment / lot / wafer metadata
  - Call the FastAPI backend
  - Render FA report with defect details, root cause, severity badge, recommendations
  - Download generated PDF
"""

from __future__ import annotations

import io
import os
import time
from typing import Any

import httpx
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

API_BASE = os.getenv("FA_API_URL", "http://localhost:8000")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Semiconductor FA System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styles ────────────────────────────────────────────────────────────────────
SEVERITY_COLOURS = {
    "CRITICAL": "#C0392B",
    "MAJOR": "#E67E22",
    "MINOR": "#F1C40F",
    "NONE": "#27AE60",
}

st.markdown("""
<style>
.severity-badge {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 4px;
    font-weight: bold;
    font-size: 1.1em;
    color: white;
    margin-bottom: 8px;
}
.metric-card {
    background: #F8F9FA;
    border-radius: 8px;
    padding: 12px 16px;
    border-left: 4px solid #2C3E50;
    margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔬 Semiconductor FA")
    st.markdown("**Multi-Modal Vision-Language AI**")
    st.markdown("---")
    st.subheader("Equipment Details")
    equipment_id = st.text_input("Equipment ID", value="EQ-INSP-01")
    lot_id = st.text_input("Lot ID", value="LOT-2024-001")
    wafer_id = st.text_input("Wafer ID", value="W05")
    image_modality = st.selectbox(
        "Image Modality",
        options=["optical", "sem", "wafer_map"],
        help="Select the type of inspection image",
    )
    st.markdown("---")
    st.caption("Stack: LLaVA-1.6 · DINOv2 · LangGraph · Qdrant · MinIO · TimescaleDB")


# ── Main content ──────────────────────────────────────────────────────────────
st.title("Autonomous Failure Analysis Report Generator")
st.markdown(
    "Upload an inspection image (SEM, optical, or wafer map). "
    "The AI pipeline will analyse the defect, identify root cause, "
    "classify severity, and generate a structured FA report in under 2 minutes."
)

col_upload, col_preview = st.columns([1, 1])

with col_upload:
    uploaded_file = st.file_uploader(
        "Upload Inspection Image",
        type=["png", "jpg", "jpeg", "tiff", "bmp"],
        help="Accepted: SEM, optical microscope, or wafer map image",
    )

    if uploaded_file:
        run_btn = st.button("Run FA Pipeline", type="primary", use_container_width=True)
    else:
        st.info("Upload an image to begin.")
        run_btn = False

with col_preview:
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption=f"Uploaded: {uploaded_file.name}", use_column_width=True)


# ── Run pipeline ──────────────────────────────────────────────────────────────
if run_btn and uploaded_file:
    st.divider()
    progress = st.progress(0, text="Initialising FA pipeline…")

    try:
        uploaded_file.seek(0)
        image_bytes = uploaded_file.read()

        with st.spinner("Running multi-modal FA pipeline…"):
            progress.progress(10, text="Uploading image to backend…")
            t0 = time.time()

            resp = httpx.post(
                f"{API_BASE}/inspection/analyze",
                files={"image": (uploaded_file.name, image_bytes, "image/png")},
                data={
                    "equipment_id": equipment_id,
                    "lot_id": lot_id,
                    "wafer_id": wafer_id,
                    "image_modality": image_modality,
                },
                timeout=180,
            )

            elapsed = time.time() - t0
            progress.progress(90, text="Rendering report…")

        if resp.status_code != 200:
            st.error(f"API error {resp.status_code}: {resp.text}")
        else:
            report = resp.json()
            progress.progress(100, text="Done!")
            _render_report(report, elapsed)

    except httpx.ConnectError:
        st.error(
            f"Cannot connect to FA API at {API_BASE}. "
            "Make sure the backend is running (`make api-dev`)."
        )
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")


def _render_report(report: dict[str, Any], wall_time: float) -> None:
    """Render the FA report dict into Streamlit UI."""
    st.success(f"FA report generated in {report.get('elapsed_seconds', wall_time):.1f}s")

    sev = report.get("severity", {})
    severity_level = sev.get("level", "MINOR").upper()
    sev_colour = SEVERITY_COLOURS.get(severity_level, "#888")

    # ── Header KPIs ──────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Defect Class", report.get("defect", {}).get("defect_class", "N/A").title())
    k2.metric(
        "Confidence",
        f"{report.get('defect', {}).get('confidence', 0):.1%}",
    )
    k3.metric("Yield Impact", f"{sev.get('yield_impact_pct', 0):.1f}%")
    k4.metric("Report Time", f"{report.get('elapsed_seconds', 0):.1f}s")

    # Severity badge
    st.markdown(
        f'<div class="severity-badge" style="background:{sev_colour}">'
        f'SEVERITY: {severity_level}</div>',
        unsafe_allow_html=True,
    )
    st.caption(sev.get("reasoning", ""))

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Defect Description", "Root Cause", "Recommendations", "Raw JSON"]
    )

    with tab1:
        defect = report.get("defect", {})
        st.subheader("Defect Description")
        st.write(defect.get("description", "N/A"))

        wm = defect.get("wafer_map_stats", {})
        if wm:
            st.subheader("Wafer Map Analysis")
            wm_cols = st.columns(3)
            wm_cols[0].metric("Pattern", wm.get("pattern_class", "N/A"))
            wm_cols[1].metric(
                "Defect Density", f"{wm.get('defect_density', 0):.2%}"
            )
            wm_cols[2].metric("Clusters", wm.get("cluster_count", "N/A"))

            # Radial distribution indicator
            _render_wafer_map_chart(wm)

    with tab2:
        rc = report.get("root_cause", {})
        st.subheader("Root Cause Summary")
        st.info(rc.get("summary", "N/A"))
        st.subheader("Hypotheses")
        for i, hyp in enumerate(rc.get("hypotheses", []), 1):
            st.markdown(f"**{i}.** {hyp}")
        st.caption(
            f"Based on {rc.get('similar_defect_count', 0)} historically similar defects."
        )

    with tab3:
        recs = report.get("recommendations", {})
        st.subheader("Corrective Actions")
        for i, action in enumerate(recs.get("actions", []), 1):
            st.markdown(f"**{i}.** {action}")

        params = recs.get("process_parameters", {})
        if params:
            st.subheader("Process Parameter Adjustments")
            for param, value in params.items():
                st.markdown(f"- **{param}**: {value}")

        # Download PDF
        report_id = report.get("report_id", "")
        if report_id:
            try:
                pdf_resp = httpx.get(
                    f"{API_BASE}/reports/{report_id[:8]}/pdf", timeout=10
                )
                if pdf_resp.status_code == 200:
                    st.download_button(
                        label="Download FA Report PDF",
                        data=pdf_resp.content,
                        file_name=f"FA_Report_{report_id[:8]}.pdf",
                        mime="application/pdf",
                    )
            except Exception:
                pass

    with tab4:
        st.json(report)

    # Errors
    if report.get("errors"):
        with st.expander("Pipeline warnings"):
            for err in report["errors"]:
                st.warning(err)


def _render_wafer_map_chart(wm: dict) -> None:
    """Simple polar chart showing radial defect distribution."""
    density = wm.get("defect_density", 0)
    dist = wm.get("radial_distribution", "uniform")

    # Synthesise radial distribution for visualisation
    if dist == "center-heavy":
        r_vals = [density * 3, density * 2, density, density * 0.5]
    elif dist == "edge-heavy":
        r_vals = [density * 0.5, density, density * 2, density * 3]
    else:
        r_vals = [density] * 4

    fig = go.Figure(
        go.Scatterpolar(
            r=r_vals + [r_vals[0]],
            theta=["Center", "Inner", "Outer", "Edge", "Center"],
            fill="toself",
            name="Defect Density",
            line_color="#E74C3C",
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, max(r_vals) * 1.2 + 0.001])),
        showlegend=False,
        height=280,
        margin=dict(l=30, r=30, t=30, b=30),
        title="Radial Defect Distribution",
    )
    st.plotly_chart(fig, use_container_width=True)


# Call render here to avoid NameError (defined after usage in conditional block)
if "report" in dir() and "elapsed" in dir():  # type: ignore[name-defined]
    pass  # already rendered inline above
