"""SECS/GEM client — HSMS passive mode connection to fab equipment.

Uses the secs4net-compatible Python protocol implementation.
Falls back to a simulation mode when SECSGEM_SIM=true.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class SECSMessage:
    stream: int
    function: int
    equipment_id: str
    body: dict[str, Any] = field(default_factory=dict)


@dataclass
class EquipmentEvent:
    """Parsed SECS/GEM event (S6F11 – Event Notification)."""
    equipment_id: str
    ceid: int          # Collection Event ID
    parameters: dict[str, Any]


# ---------------------------------------------------------------------------
# SECS/GEM simulation (used when real equipment is unavailable)
# ---------------------------------------------------------------------------

class _SECSGEMSimulator:
    """Generates synthetic equipment events for offline testing."""

    def __init__(
        self,
        equipment_id: str,
        event_callback: Callable[[EquipmentEvent], None],
    ) -> None:
        self._equipment_id = equipment_id
        self._callback = event_callback
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._emit_loop, daemon=True)
        self._thread.start()
        logger.info("SECS/GEM simulator started for %s", self._equipment_id)

    def stop(self) -> None:
        self._running = False

    def _emit_loop(self) -> None:
        import random, math

        ceid_cycle = [1, 2, 3, 100]  # process start, end, alarm, lot start
        tick = 0
        while self._running:
            ceid = ceid_cycle[tick % len(ceid_cycle)]
            event = EquipmentEvent(
                equipment_id=self._equipment_id,
                ceid=ceid,
                parameters={
                    "chuck_temp": round(22.5 + math.sin(tick * 0.1) * 1.5, 2),
                    "bond_force": round(50 + random.gauss(0, 2), 2),
                    "loop_height": round(120 + random.gauss(0, 5), 2),
                    "alarm_code": "A001" if ceid == 3 else "",
                    "lot_id": f"LOT-{tick // 4:04d}",
                    "wafer_id": f"W{(tick % 4) + 1:02d}",
                },
            )
            self._callback(event)
            tick += 1
            time.sleep(2)


# ---------------------------------------------------------------------------
# Public client
# ---------------------------------------------------------------------------

class SECSGEMClient:
    """
    Thin SECS/GEM equipment interface.

    In production: connects via HSMS to real KLA / K&S equipment.
    In simulation mode (env SECSGEM_SIM=true): uses _SECSGEMSimulator.
    """

    def __init__(
        self,
        equipment_id: str,
        host: str,
        port: int,
        session_id: int = 1,
        event_callback: Callable[[EquipmentEvent], None] | None = None,
    ) -> None:
        self._equipment_id = equipment_id
        self._host = host
        self._port = port
        self._session_id = session_id
        self._callback = event_callback
        self._sim_mode = os.getenv("SECSGEM_SIM", "true").lower() == "true"
        self._simulator: _SECSGEMSimulator | None = None
        self._connected = False

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def connect(self) -> None:
        if self._sim_mode:
            if self._callback is None:
                raise ValueError("event_callback required for simulator mode.")
            self._simulator = _SECSGEMSimulator(
                self._equipment_id, self._callback
            )
            self._simulator.start()
            self._connected = True
        else:
            self._connect_hsms()

    def disconnect(self) -> None:
        if self._sim_mode and self._simulator:
            self._simulator.stop()
        self._connected = False

    # ── HSMS connection (production) ─────────────────────────────────────────

    def _connect_hsms(self) -> None:
        """
        Real HSMS connection via secs4net or equivalent library.
        Stub implementation — replace with actual HSMS active/passive handler.
        """
        try:
            import secsgem.hsms as hsms  # type: ignore[import]

            logger.info(
                "Connecting HSMS to %s:%d session=%d",
                self._host, self._port, self._session_id,
            )
            # Equipment-specific connection logic would go here.
            self._connected = True
        except ImportError:
            logger.warning(
                "secsgem library not installed; falling back to simulation mode."
            )
            self._sim_mode = True
            self.connect()

    # ── Send/Receive helpers ─────────────────────────────────────────────────

    def send_remote_command(self, rcmd: str, params: dict[str, Any]) -> bool:
        """S2F41 – Host Command Send."""
        if not self._connected:
            raise RuntimeError("Not connected.")
        if self._sim_mode:
            logger.debug("SIM send_remote_command: rcmd=%s params=%s", rcmd, params)
            return True
        # Production: serialize to SECS-II and send via HSMS
        return True

    def request_equipment_status(self) -> dict[str, Any]:
        """S1F3 – Selected Equipment Status Request."""
        if self._sim_mode:
            return {
                "equipment_id": self._equipment_id,
                "state": "PROCESS",
                "alarm_count": 0,
            }
        return {}

    @property
    def is_connected(self) -> bool:
        return self._connected
