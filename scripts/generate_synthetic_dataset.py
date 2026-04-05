"""
Synthetic semiconductor defect image generator.

Generates 10 images per defect class (90 total) across two modalities:
  - SEM-like grayscale (scratch, particle_contamination, edge_crack)
  - Wafer map binary (center_cluster, local_cluster, ring_pattern,
                      random_defects, near_full_wafer, no_defect)

Output layout:
  data/synthetic_dataset/
    images/
      scratch_00.png ... scratch_09.png
      particle_contamination_00.png ...
      ...
    annotations.jsonl   ← one JSON object per line, compatible with
                           training/prepare_dataset.py

Usage:
    python scripts/generate_synthetic_dataset.py
    python scripts/generate_synthetic_dataset.py --output data/synthetic_dataset --n 20
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# ── reproducibility ──────────────────────────────────────────────────────────
SEED = 42
rng = np.random.default_rng(SEED)
random.seed(SEED)

# ── defect class metadata ─────────────────────────────────────────────────────
DEFECT_META: dict[str, dict] = {
    "scratch": {
        "modality": "sem",
        "description_template": (
            "SEM image showing a linear scratch defect extending diagonally "
            "across the die surface. The scratch exhibits a dark, elongated "
            "morphology with slight edge brightening, consistent with "
            "mechanical abrasion damage."
        ),
    },
    "particle_contamination": {
        "modality": "sem",
        "description_template": (
            "SEM image showing multiple bright particle contaminants on the "
            "die surface. Particles range from 0.5–5 µm in diameter with "
            "high secondary-electron contrast, indicating foreign material "
            "deposition, likely organic or metallic origin."
        ),
    },
    "edge_crack": {
        "modality": "sem",
        "description_template": (
            "SEM image showing a crack originating at the die edge and "
            "propagating inward. The crack exhibits characteristic brittle "
            "fracture morphology with sharp, irregular edges, consistent "
            "with dicing or handling stress."
        ),
    },
    "center_cluster": {
        "modality": "wafer_map",
        "description_template": (
            "Wafer map showing a high-density cluster of defective dies "
            "concentrated in the central region of the wafer. This pattern "
            "is characteristic of process non-uniformity at the wafer center, "
            "potentially related to chuck temperature or gas flow anomalies."
        ),
    },
    "local_cluster": {
        "modality": "wafer_map",
        "description_template": (
            "Wafer map showing a localised cluster of defective dies in a "
            "non-central region. The spatial confinement suggests a local "
            "event such as a particle shower, liquid droplet, or localised "
            "plasma non-uniformity."
        ),
    },
    "ring_pattern": {
        "modality": "wafer_map",
        "description_template": (
            "Wafer map showing defective dies arranged in a concentric ring "
            "pattern. This distribution is associated with edge-bead removal "
            "issues, spin-coat non-uniformity, or annular plasma standing waves."
        ),
    },
    "random_defects": {
        "modality": "wafer_map",
        "description_template": (
            "Wafer map showing randomly distributed defective dies with no "
            "clear spatial pattern. The stochastic distribution suggests "
            "random particle events or low-level process instability rather "
            "than a systematic root cause."
        ),
    },
    "near_full_wafer": {
        "modality": "wafer_map",
        "description_template": (
            "Wafer map showing near-complete coverage of defective dies across "
            "the wafer. This catastrophic pattern indicates a systemic process "
            "failure such as a chemistry excursion, equipment malfunction, or "
            "severe contamination event."
        ),
    },
    "no_defect": {
        "modality": "wafer_map",
        "description_template": (
            "Wafer map showing a clean die distribution with no detected "
            "defects. All functional dies pass electrical test criteria. "
            "Process parameters are within specification."
        ),
    },
}


# ── wafer mask helper ─────────────────────────────────────────────────────────

def _wafer_mask(size: int = 256) -> np.ndarray:
    """Boolean mask: True inside the circular wafer boundary."""
    cx = cy = size // 2
    r = size // 2 - 4
    y, x = np.ogrid[:size, :size]
    return (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2


# ── SEM image generators ──────────────────────────────────────────────────────

def _sem_background(size: int, seed: int) -> np.ndarray:
    """Realistic SEM background: Gaussian noise on mid-gray."""
    rng_local = np.random.default_rng(seed)
    base = rng_local.normal(130, 18, (size, size)).clip(0, 255).astype(np.float32)
    # Low-frequency intensity gradient (common in SEM)
    gy, gx = np.mgrid[0:size, 0:size]
    gradient = 10 * np.sin(gy / size * np.pi) * np.cos(gx / size * np.pi)
    return (base + gradient).clip(0, 255).astype(np.uint8)


def generate_scratch(size: int = 512, seed: int = 0) -> Image.Image:
    rng_local = np.random.default_rng(seed)
    arr = _sem_background(size, seed).astype(np.float32)

    # Random start/end points for the scratch
    angle = rng_local.uniform(20, 70)  # degrees
    cx, cy = size // 2, size // 2
    length = int(size * rng_local.uniform(0.5, 0.85))
    dx = np.cos(np.radians(angle))
    dy = np.sin(np.radians(angle))

    x0 = int(cx - dx * length / 2)
    y0 = int(cy - dy * length / 2)
    x1 = int(cx + dx * length / 2)
    y1 = int(cy + dy * length / 2)

    # Draw scratch as dark track with width variation
    n_steps = length * 3
    for t in np.linspace(0, 1, n_steps):
        px = int(x0 + t * (x1 - x0))
        py = int(y0 + t * (y1 - y0))
        width = int(rng_local.uniform(1, 4))
        intensity = rng_local.uniform(15, 45)
        for dy_ in range(-width, width + 1):
            for dx_ in range(-width, width + 1):
                ny, nx = py + dy_, px + dx_
                if 0 <= ny < size and 0 <= nx < size:
                    dist = np.sqrt(dx_ ** 2 + dy_ ** 2)
                    if dist <= width:
                        arr[ny, nx] = intensity + rng_local.uniform(0, 10)

    # Bright edge halos along scratch
    scratch_img = Image.fromarray(arr.clip(0, 255).astype(np.uint8), mode="L")
    scratch_img = scratch_img.filter(ImageFilter.SMOOTH_MORE)
    return scratch_img.convert("RGB")


def generate_particle_contamination(size: int = 512, seed: int = 0) -> Image.Image:
    rng_local = np.random.default_rng(seed)
    arr = _sem_background(size, seed).astype(np.float32)

    n_particles = int(rng_local.uniform(8, 25))
    for _ in range(n_particles):
        cy = int(rng_local.uniform(20, size - 20))
        cx = int(rng_local.uniform(20, size - 20))
        radius = int(rng_local.uniform(2, 10))
        brightness = rng_local.uniform(200, 255)
        y, x = np.ogrid[:size, :size]
        mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
        # Bright core with halo
        arr[mask] = brightness
        halo_mask = (x - cx) ** 2 + (y - cy) ** 2 <= (radius + 3) ** 2
        arr[halo_mask & ~mask] = np.minimum(
            arr[halo_mask & ~mask] + rng_local.uniform(20, 50), 255
        )

    return Image.fromarray(arr.clip(0, 255).astype(np.uint8), mode="L").convert("RGB")


def generate_edge_crack(size: int = 512, seed: int = 0) -> Image.Image:
    rng_local = np.random.default_rng(seed)
    arr = _sem_background(size, seed).astype(np.float32)

    # Choose a random edge side: 0=top, 1=right, 2=bottom, 3=left
    side = int(rng_local.integers(0, 4))
    if side == 0:
        start = (int(rng_local.uniform(size * 0.3, size * 0.7)), 0)
    elif side == 1:
        start = (size - 1, int(rng_local.uniform(size * 0.3, size * 0.7)))
    elif side == 2:
        start = (int(rng_local.uniform(size * 0.3, size * 0.7)), size - 1)
    else:
        start = (0, int(rng_local.uniform(size * 0.3, size * 0.7)))

    # Propagate crack inward with branching
    def draw_crack(x, y, angle, depth, arr):
        if depth <= 0:
            return
        length = int(rng_local.uniform(20, 60))
        for _ in range(length):
            angle += rng_local.uniform(-15, 15)
            x = int(x + np.cos(np.radians(angle)))
            y = int(y + np.sin(np.radians(angle)))
            if not (0 <= x < size and 0 <= y < size):
                return
            w = max(1, depth // 3)
            for ddy in range(-w, w + 1):
                for ddx in range(-w, w + 1):
                    ny, nx = y + ddy, x + ddx
                    if 0 <= ny < size and 0 <= nx < size:
                        arr[ny, nx] = rng_local.uniform(10, 35)
        # Occasional branch
        if depth > 1 and rng_local.random() < 0.4:
            draw_crack(x, y, angle + rng_local.uniform(20, 45), depth - 1, arr)

    inward_angle = {0: 90, 1: 180, 2: 270, 3: 0}[side]
    draw_crack(start[0], start[1], inward_angle, depth=3, arr=arr)

    img = Image.fromarray(arr.clip(0, 255).astype(np.uint8), mode="L")
    img = img.filter(ImageFilter.SMOOTH)
    return img.convert("RGB")


# ── Wafer map generators ──────────────────────────────────────────────────────

def _render_wafer_map(defect_map: np.ndarray, size: int = 256) -> Image.Image:
    """
    Render a wafer map as an RGB image.
      - Outside wafer: black (0,0,0)
      - Good die:      dark green (30, 120, 50)
      - Defective die: red (200, 40, 40)
    """
    mask = _wafer_mask(size)
    rgb = np.zeros((size, size, 3), dtype=np.uint8)
    # Good dies
    good = mask & (defect_map == 0)
    rgb[good] = [30, 120, 50]
    # Defective dies
    bad = mask & (defect_map == 1)
    rgb[bad] = [200, 40, 40]
    return Image.fromarray(rgb, mode="RGB")


def generate_center_cluster(size: int = 256, seed: int = 0) -> Image.Image:
    rng_local = np.random.default_rng(seed)
    mask = _wafer_mask(size)
    defect_map = np.zeros((size, size), dtype=np.uint8)
    cx = cy = size // 2
    r = int(rng_local.uniform(size * 0.12, size * 0.22))
    y, x = np.ogrid[:size, :size]
    cluster = (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
    prob = rng_local.random((size, size))
    defect_map[mask & cluster & (prob < 0.80)] = 1
    # Sparse scatter outside cluster
    defect_map[mask & ~cluster & (rng_local.random((size, size)) < 0.02)] = 1
    return _render_wafer_map(defect_map, size)


def generate_local_cluster(size: int = 256, seed: int = 0) -> Image.Image:
    rng_local = np.random.default_rng(seed)
    mask = _wafer_mask(size)
    defect_map = np.zeros((size, size), dtype=np.uint8)
    wafer_r = size // 2 - 4
    # Place cluster away from center
    angle = rng_local.uniform(0, 360)
    offset = rng_local.uniform(0.3, 0.55) * wafer_r
    cx = int(size // 2 + offset * np.cos(np.radians(angle)))
    cy = int(size // 2 + offset * np.sin(np.radians(angle)))
    r = int(rng_local.uniform(size * 0.10, size * 0.18))
    y, x = np.ogrid[:size, :size]
    cluster = (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
    prob = rng_local.random((size, size))
    defect_map[mask & cluster & (prob < 0.82)] = 1
    defect_map[mask & ~cluster & (rng_local.random((size, size)) < 0.015)] = 1
    return _render_wafer_map(defect_map, size)


def generate_ring_pattern(size: int = 256, seed: int = 0) -> Image.Image:
    rng_local = np.random.default_rng(seed)
    mask = _wafer_mask(size)
    defect_map = np.zeros((size, size), dtype=np.uint8)
    cx = cy = size // 2
    ring_r = int(rng_local.uniform(size * 0.25, size * 0.38))
    ring_w = int(rng_local.uniform(size * 0.04, size * 0.08))
    y, x = np.ogrid[:size, :size]
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    ring = (dist >= ring_r - ring_w) & (dist <= ring_r + ring_w)
    prob = rng_local.random((size, size))
    defect_map[mask & ring & (prob < 0.75)] = 1
    defect_map[mask & ~ring & (rng_local.random((size, size)) < 0.015)] = 1
    return _render_wafer_map(defect_map, size)


def generate_random_defects(size: int = 256, seed: int = 0) -> Image.Image:
    rng_local = np.random.default_rng(seed)
    mask = _wafer_mask(size)
    density = rng_local.uniform(0.05, 0.20)
    prob = rng_local.random((size, size))
    defect_map = (mask & (prob < density)).astype(np.uint8)
    return _render_wafer_map(defect_map, size)


def generate_near_full_wafer(size: int = 256, seed: int = 0) -> Image.Image:
    rng_local = np.random.default_rng(seed)
    mask = _wafer_mask(size)
    density = rng_local.uniform(0.75, 0.95)
    prob = rng_local.random((size, size))
    defect_map = (mask & (prob < density)).astype(np.uint8)
    return _render_wafer_map(defect_map, size)


def generate_no_defect(size: int = 256, seed: int = 0) -> Image.Image:
    rng_local = np.random.default_rng(seed)
    mask = _wafer_mask(size)
    # Very occasional stray die (< 0.5 %)
    prob = rng_local.random((size, size))
    defect_map = (mask & (prob < 0.005)).astype(np.uint8)
    return _render_wafer_map(defect_map, size)


# ── dispatch table ────────────────────────────────────────────────────────────

GENERATORS = {
    "scratch": generate_scratch,
    "particle_contamination": generate_particle_contamination,
    "edge_crack": generate_edge_crack,
    "center_cluster": generate_center_cluster,
    "local_cluster": generate_local_cluster,
    "ring_pattern": generate_ring_pattern,
    "random_defects": generate_random_defects,
    "near_full_wafer": generate_near_full_wafer,
    "no_defect": generate_no_defect,
}

SEM_SIZE = 512
WAFER_SIZE = 256


# ── main ──────────────────────────────────────────────────────────────────────

def generate_dataset(output_dir: str = "data/synthetic_dataset", n_per_class: int = 10) -> None:
    out = Path(output_dir)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    annotations = []

    for defect_class, meta in DEFECT_META.items():
        modality = meta["modality"]
        size = SEM_SIZE if modality == "sem" else WAFER_SIZE
        generator = GENERATORS[defect_class]

        print(f"  Generating {n_per_class}x {defect_class} ({modality}) ...", end=" ")
        for i in range(n_per_class):
            seed = SEED + hash(defect_class) % 10000 + i
            img = generator(size=size, seed=seed)

            filename = f"{defect_class}_{i:02d}.png"
            img_path = img_dir / filename
            img.save(img_path)

            annotations.append({
                "image_path": str(img_path.as_posix()),
                "defect_class": defect_class,
                "modality": modality,
                "description": meta["description_template"],
                "equipment_id": f"SIM-EQ-{(i % 3) + 1:02d}",
                "lot_id": f"LOT-SIM-{(i // 3) + 1:03d}",
                "wafer_id": f"W{i + 1:02d}",
                "severity": _default_severity(defect_class),
                "source": "synthetic",
            })

        print(f"done ({n_per_class} images)")

    ann_path = out / "annotations.jsonl"
    with open(ann_path, "w") as f:
        for entry in annotations:
            f.write(json.dumps(entry) + "\n")

    total = len(annotations)
    print(f"\nDataset saved to: {out.resolve()}")
    print(f"  {total} images across {len(DEFECT_META)} classes")
    print(f"  Annotations: {ann_path}")


def _default_severity(defect_class: str) -> str:
    return {
        "scratch": "MAJOR",
        "particle_contamination": "MINOR",
        "edge_crack": "CRITICAL",
        "center_cluster": "CRITICAL",
        "local_cluster": "MAJOR",
        "ring_pattern": "MAJOR",
        "random_defects": "MINOR",
        "near_full_wafer": "CRITICAL",
        "no_defect": "NONE",
    }[defect_class]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic semiconductor defect dataset")
    parser.add_argument("--output", default="data/synthetic_dataset", help="Output directory")
    parser.add_argument("--n", type=int, default=10, help="Images per defect class")
    args = parser.parse_args()

    print(f"Generating synthetic dataset ({args.n} images/class -> {args.n * len(DEFECT_META)} total)")
    print(f"Output: {args.output}\n")
    generate_dataset(output_dir=args.output, n_per_class=args.n)
