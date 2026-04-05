"""Model registry — singleton access to shared model instances.

All agents share the same loaded models to avoid reloading weights
on every request.
"""

from __future__ import annotations

import threading
import logging

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Thread-safe singleton registry for model instances."""

    _instance: "ModelRegistry | None" = None
    _lock = threading.Lock()

    def __new__(cls) -> "ModelRegistry":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialised = False
        return cls._instance

    def initialise(self, load_llava: bool = True, load_detector: bool = True) -> None:
        if self._initialised:
            return

        with self._lock:
            if self._initialised:
                return

            self._llava: "LLaVAInferenceEngine | None" = None
            self._detector: "DefectDetector | None" = None
            self._wafer_analyzer: "WaferMapAnalyzer | None" = None

            if load_llava:
                from src.models.llava_inference import LLaVAInferenceEngine
                self._llava = LLaVAInferenceEngine()
                self._llava.load()

            if load_detector:
                from src.vision.defect_detector import DefectDetector
                self._detector = DefectDetector()

            from src.vision.wafer_map_analyzer import WaferMapAnalyzer
            self._wafer_analyzer = WaferMapAnalyzer()

            self._initialised = True
            logger.info("ModelRegistry initialised.")

    # ── Accessors ────────────────────────────────────────────────────────────

    @property
    def llava(self) -> "LLaVAInferenceEngine":
        if self._llava is None:
            raise RuntimeError("LLaVA not loaded. Call initialise(load_llava=True).")
        return self._llava

    @property
    def detector(self) -> "DefectDetector":
        if self._detector is None:
            raise RuntimeError("Detector not loaded.")
        return self._detector

    @property
    def wafer_analyzer(self) -> "WaferMapAnalyzer":
        if self._wafer_analyzer is None:
            raise RuntimeError("WaferMapAnalyzer not loaded.")
        return self._wafer_analyzer


# Module-level singleton
registry = ModelRegistry()
