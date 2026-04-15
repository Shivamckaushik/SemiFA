"""MQTT ingestion — subscribes to equipment topic and forwards to TimescaleDB."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Callable

import paho.mqtt.client as mqtt

from src.config import settings

logger = logging.getLogger(__name__)


class MQTTIngestionClient:
    """
    Subscribes to fab/equipment/# topic.

    Expected message payload (JSON):
    {
        "equipment_id": "EQ-INSP-01",
        "parameter": "chuck_temp",
        "value": 23.4,
        "unit": "°C",
        "alarm_code": ""
    }
    """

    def __init__(
        self,
        on_message_callback: Callable[[dict], None] | None = None,
    ) -> None:
        self._callback = on_message_callback
        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

        if settings.mqtt_username:
            self._client.username_pw_set(
                settings.mqtt_username, settings.mqtt_password
            )

        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message
        self._client.on_disconnect = self._on_disconnect

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._client.connect(
            settings.mqtt_broker_host, settings.mqtt_broker_port, keepalive=60
        )
        self._client.loop_start()
        logger.info(
            "MQTT client started — broker=%s:%d",
            settings.mqtt_broker_host,
            settings.mqtt_broker_port,
        )

    def stop(self) -> None:
        self._client.loop_stop()
        self._client.disconnect()
        logger.info("MQTT client stopped.")

    # ── Callbacks ────────────────────────────────────────────────────────────

    def _on_connect(self, client, userdata, flags, reason_code, properties) -> None:  # noqa: ANN001
        if reason_code == 0:
            logger.info("MQTT connected — subscribing to %s", settings.mqtt_equipment_topic)
            client.subscribe(settings.mqtt_equipment_topic, qos=1)
        else:
            logger.error("MQTT connection failed: reason_code=%s", reason_code)

    def _on_message(self, client, userdata, msg: mqtt.MQTTMessage) -> None:  # noqa: ANN001
        try:
            payload = json.loads(msg.payload.decode())
            logger.debug("MQTT msg: topic=%s payload=%s", msg.topic, payload)
            if self._callback:
                self._callback(payload)
        except json.JSONDecodeError as exc:
            logger.warning("Invalid MQTT payload on %s: %s", msg.topic, exc)

    def _on_disconnect(self, client, userdata, disconnect_flags, reason_code, properties) -> None:  # noqa: ANN001
        logger.warning("MQTT disconnected: reason_code=%s", reason_code)

    # ── Publishing (for simulation / testing) ────────────────────────────────

    def publish_equipment_data(self, equipment_id: str, data: dict) -> None:
        topic = f"fab/equipment/{equipment_id}"
        self._client.publish(topic, json.dumps(data), qos=1)
