"""Wafer map analysis — pattern classification and spatial statistics.

Uses the WM-811K defect taxonomy (9 classes) and morphological analysis
to characterise defect spatial distributions on wafer maps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import cv2
from PIL import Image


WAFER_MAP_CLASSES = [
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Near-full",
    "Random",
    "Scratch",
    "none",
]


@dataclass
class WaferMapStats:
    pattern_class: str
    defect_density: float          # fraction of die with defects
    defect_count: int
    cluster_count: int
    radial_distribution: str       # "center-heavy" | "edge-heavy" | "uniform"
    spatial_entropy: float
    bounding_box: tuple[int, int, int, int] | None   # (x, y, w, h) of largest cluster
    raw_stats: dict[str, Any]


class WaferMapAnalyzer:
    """
    Analyse a wafer map image and return spatial statistics + pattern class.

    Wafer maps are expected to be:
      - Single-channel (grayscale) or binary
      - 0 = good die, 255 = defective die
      - Square (any resolution; internally rescaled to 256×256)
    """

    TARGET_SIZE = (256, 256)

    # ── Public API ───────────────────────────────────────────────────────────

    def analyze(self, image: Image.Image | np.ndarray) -> WaferMapStats:
        wmap = self._load_binary_map(image)
        stats = self._compute_stats(wmap)
        pattern = self._classify_pattern(wmap, stats)
        return WaferMapStats(
            pattern_class=pattern,
            defect_density=stats["defect_density"],
            defect_count=stats["defect_count"],
            cluster_count=stats["cluster_count"],
            radial_distribution=stats["radial_distribution"],
            spatial_entropy=stats["spatial_entropy"],
            bounding_box=stats["largest_bbox"],
            raw_stats=stats,
        )

    # ── Internals ────────────────────────────────────────────────────────────

    def _load_binary_map(self, image: Image.Image | np.ndarray) -> np.ndarray:
        if isinstance(image, Image.Image):
            img = np.array(image.convert("L"))
        else:
            img = image.copy()
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        img = cv2.resize(img, self.TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        return binary

    def _compute_stats(self, wmap: np.ndarray) -> dict[str, Any]:
        h, w = wmap.shape
        total_die = h * w
        defect_pixels = int(np.sum(wmap > 0))
        defect_density = defect_pixels / total_die

        # Connected components (clusters)
        num_labels, labels, stats_cc, centroids = cv2.connectedComponentsWithStats(
            wmap, connectivity=8
        )
        cluster_count = num_labels - 1  # subtract background

        # Largest cluster bounding box
        largest_bbox = None
        if cluster_count > 0:
            areas = stats_cc[1:, cv2.CC_STAT_AREA]
            largest_idx = int(np.argmax(areas)) + 1
            x = int(stats_cc[largest_idx, cv2.CC_STAT_LEFT])
            y = int(stats_cc[largest_idx, cv2.CC_STAT_TOP])
            bw = int(stats_cc[largest_idx, cv2.CC_STAT_WIDTH])
            bh = int(stats_cc[largest_idx, cv2.CC_STAT_HEIGHT])
            largest_bbox = (x, y, bw, bh)

        # Radial distribution
        cy, cx = h // 2, w // 2
        y_idx, x_idx = np.where(wmap > 0)
        radial_dist = "uniform"
        if len(y_idx) > 0:
            radii = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2)
            max_r = np.sqrt(cx**2 + cy**2)
            inner = np.sum(radii < max_r * 0.4)
            outer = np.sum(radii > max_r * 0.7)
            if inner > outer * 2:
                radial_dist = "center-heavy"
            elif outer > inner * 2:
                radial_dist = "edge-heavy"

        # Spatial entropy
        hist, _ = np.histogram(wmap[wmap > 0] if defect_pixels > 0 else [0], bins=8)
        hist = hist / (hist.sum() + 1e-9)
        entropy = float(-np.sum(hist * np.log2(hist + 1e-9)))

        return {
            "defect_density": round(defect_density, 4),
            "defect_count": defect_pixels,
            "cluster_count": cluster_count,
            "radial_distribution": radial_dist,
            "spatial_entropy": round(entropy, 4),
            "largest_bbox": largest_bbox,
            "num_cc": cluster_count,
        }

    def _classify_pattern(self, wmap: np.ndarray, stats: dict) -> str:
        """Rule-based pattern classification (fast, interpretable)."""
        density = stats["defect_density"]
        clusters = stats["cluster_count"]
        radial = stats["radial_distribution"]

        if density < 0.005:
            return "none"
        if density > 0.80:
            return "Near-full"
        if radial == "edge-heavy" and clusters <= 3:
            return "Edge-Ring"
        if radial == "edge-heavy" and clusters > 3:
            return "Edge-Loc"
        if radial == "center-heavy" and density < 0.15:
            return "Center"
        if radial == "center-heavy" and density >= 0.15:
            return "Donut"

        # Check for scratch (elongated single cluster)
        if stats["largest_bbox"] is not None:
            _, _, bw, bh = stats["largest_bbox"]
            aspect = max(bw, bh) / (min(bw, bh) + 1e-9)
            if aspect > 5 and clusters < 4:
                return "Scratch"

        if clusters <= 3 and density < 0.10:
            return "Loc"

        return "Random"
