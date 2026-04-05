"""Reports endpoint — retrieve historical FA reports and PDFs."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/reports", tags=["Reports"])


@router.get("/{report_id}/pdf")
async def download_report_pdf(report_id: str) -> FileResponse:
    """Download the PDF for a given report ID (first 8 chars)."""
    from src.config import settings
    out_dir = Path(settings.report_output_dir)
    matches = list(out_dir.glob(f"FA_Report_{report_id[:8]}*.pdf"))
    if not matches:
        raise HTTPException(status_code=404, detail=f"Report {report_id} not found.")
    return FileResponse(
        path=str(matches[0]),
        media_type="application/pdf",
        filename=matches[0].name,
    )


@router.get("/list")
async def list_reports() -> dict:
    """List all generated FA report PDFs."""
    from src.config import settings
    out_dir = Path(settings.report_output_dir)
    pdfs = sorted(out_dir.glob("FA_Report_*.pdf"), reverse=True)
    return {
        "count": len(pdfs),
        "reports": [
            {"filename": p.name, "size_kb": round(p.stat().st_size / 1024, 1)}
            for p in pdfs[:50]
        ],
    }
