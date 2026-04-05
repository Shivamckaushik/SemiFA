"""TimescaleDB async client — equipment telemetry reads and writes."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import asyncpg

from src.config import settings


class TelemetryDB:
    """Async PostgreSQL/TimescaleDB client for equipment telemetry."""

    def __init__(self) -> None:
        self._dsn = (
            f"postgresql://{settings.timescale_user}:{settings.timescale_password}"
            f"@{settings.timescale_host}:{settings.timescale_port}/{settings.timescale_db}"
        )
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        self._pool = await asyncpg.create_pool(self._dsn, min_size=2, max_size=10)

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()

    @property
    def pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("Call connect() before using the pool.")
        return self._pool

    # ── Telemetry ─────────────────────────────────────────────────────────────

    async def insert_telemetry(
        self,
        equipment_id: str,
        parameter_name: str,
        value: float,
        unit: str = "",
        alarm_code: str = "",
        time: datetime | None = None,
    ) -> None:
        ts = time or datetime.now(timezone.utc)
        await self.pool.execute(
            """
            INSERT INTO equipment_telemetry
                (time, equipment_id, parameter_name, value, unit, alarm_code)
            VALUES ($1, $2, $3, $4, $5, $6)
            """,
            ts, equipment_id, parameter_name, value, unit, alarm_code,
        )

    async def fetch_recent_telemetry(
        self,
        equipment_id: str,
        parameter_name: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        rows = await self.pool.fetch(
            """
            SELECT time, value, unit, alarm_code
            FROM equipment_telemetry
            WHERE equipment_id = $1 AND parameter_name = $2
            ORDER BY time DESC
            LIMIT $3
            """,
            equipment_id, parameter_name, limit,
        )
        return [dict(r) for r in rows]

    async def fetch_alarm_history(
        self, equipment_id: str, since: datetime
    ) -> list[dict[str, Any]]:
        rows = await self.pool.fetch(
            """
            SELECT time, parameter_name, value, alarm_code
            FROM equipment_telemetry
            WHERE equipment_id = $1 AND alarm_code != '' AND time >= $2
            ORDER BY time DESC
            """,
            equipment_id, since,
        )
        return [dict(r) for r in rows]

    # ── Inspection events ─────────────────────────────────────────────────────

    async def record_inspection(
        self,
        equipment_id: str,
        lot_id: str,
        wafer_id: str,
        image_path: str,
        defect_type: str,
        severity: str,
    ) -> str:
        row = await self.pool.fetchrow(
            """
            INSERT INTO inspection_events
                (equipment_id, lot_id, wafer_id, image_path, defect_type, severity)
            VALUES ($1, $2, $3, $4, $5, $6)
            RETURNING id
            """,
            equipment_id, lot_id, wafer_id, image_path, defect_type, severity,
        )
        return str(row["id"])

    async def record_fa_report(
        self,
        inspection_id: str,
        report_path: str,
        summary: str,
        root_cause: str,
        recommendation: str,
    ) -> str:
        row = await self.pool.fetchrow(
            """
            INSERT INTO fa_reports
                (inspection_id, report_path, summary, root_cause, recommendation)
            VALUES ($1, $2, $3, $4, $5)
            RETURNING id
            """,
            uuid.UUID(inspection_id), report_path, summary, root_cause, recommendation,
        )
        return str(row["id"])
