"""
WM-811K wafer map dataset downloader and preprocessor.

WM-811K is the largest public wafer map dataset:
  - 811,457 wafer maps from real semiconductor fabs
  - 9 pattern labels: Center, Donut, Edge-Loc, Edge-Ring, Loc, Random,
                      Scratch, Near-full, None
  - Source: Wu et al. (2015), MIR-WM811K

Maps are stored as variable-size numpy arrays (dtype uint8):
  0 = outside wafer boundary
  1 = good die
  2 = defective die

This script:
  1. Downloads WM-811K via Kaggle API (or prompts for manual download)
  2. Loads the .pkl file
  3. Samples N images per label, resizes to 256×256
  4. Saves as PNG + annotations.jsonl compatible with training/prepare_dataset.py

Output layout:
  data/wm811k/
    images/
      center_cluster_00.png ...
      ring_pattern_00.png ...
      ...
    annotations.jsonl

Usage:
    # Option A — Kaggle API (recommended):
    pip install kaggle
    # Put your kaggle.json at ~/.kaggle/kaggle.json (chmod 600)
    python scripts/download_wm811k.py

    # Option B — manual download:
    # 1. Go to https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map
    # 2. Download LSWMD.pkl and place it in data/wm811k_raw/LSWMD.pkl
    python scripts/download_wm811k.py --pkl data/wm811k_raw/LSWMD.pkl

    # Limit samples per class (default 20):
    python scripts/download_wm811k.py --n 50
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

# ── label mapping: WM-811K → our defect classes ──────────────────────────────
# WM-811K uses integer labels; the mapping below follows the dataset README.
WM811K_INT_TO_NAME = {
    0: "Center",
    1: "Donut",
    2: "Edge-Loc",
    3: "Edge-Ring",
    4: "Loc",
    5: "Random",
    6: "Scratch",
    7: "Near-full",
    8: "none",
}

# Map WM-811K names → our 9 defect classes
WM811K_TO_DEFECT_CLASS = {
    "Center":    "center_cluster",
    "Donut":     "ring_pattern",
    "Edge-Loc":  "edge_crack",       # closest spatial equivalent
    "Edge-Ring": "ring_pattern",
    "Loc":       "local_cluster",
    "Random":    "random_defects",
    "Scratch":   "scratch",
    "Near-full": "near_full_wafer",
    "none":      "no_defect",
}

# Description templates per mapped class
DESCRIPTIONS = {
    "center_cluster": (
        "Wafer map from WM-811K showing a Center pattern: defective dies "
        "concentrated in the central region, indicative of process "
        "non-uniformity at the wafer center."
    ),
    "ring_pattern": (
        "Wafer map from WM-811K showing a ring/donut defect pattern: "
        "defective dies arranged concentrically, associated with "
        "spin-coat non-uniformity or edge-bead removal issues."
    ),
    "edge_crack": (
        "Wafer map from WM-811K showing an Edge-Loc pattern: defective "
        "dies localised near one edge, consistent with edge stress, "
        "dicing damage, or localised contamination."
    ),
    "local_cluster": (
        "Wafer map from WM-811K showing a Loc pattern: a localised "
        "cluster of defective dies in a non-central, non-edge region, "
        "suggesting a discrete contamination event."
    ),
    "random_defects": (
        "Wafer map from WM-811K showing a Random pattern: stochastically "
        "distributed defective dies with no systematic spatial correlation."
    ),
    "scratch": (
        "Wafer map from WM-811K showing a Scratch pattern: defective dies "
        "arranged in a linear track across the wafer, characteristic of "
        "mechanical contact damage during handling."
    ),
    "near_full_wafer": (
        "Wafer map from WM-811K showing a Near-full pattern: nearly the "
        "entire wafer surface covered by defective dies, indicating a "
        "catastrophic process failure."
    ),
    "no_defect": (
        "Wafer map from WM-811K showing a None pattern: a clean wafer "
        "with no systematic defect distribution detected."
    ),
}

SEVERITY_MAP = {
    "center_cluster": "CRITICAL",
    "ring_pattern":   "MAJOR",
    "edge_crack":     "CRITICAL",
    "local_cluster":  "MAJOR",
    "random_defects": "MINOR",
    "scratch":        "MAJOR",
    "near_full_wafer": "CRITICAL",
    "no_defect":      "NONE",
}

TARGET_SIZE = (256, 256)
KAGGLE_DATASET = "qingyi/wm811k-wafer-map"
PKL_FILENAME = "LSWMD.pkl"


# ── download helpers ──────────────────────────────────────────────────────────

def _download_via_kaggle(raw_dir: Path) -> Path:
    """Download WM-811K zip via Kaggle CLI and extract the .pkl file."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "wm811k-wafer-map.zip"

    if not zip_path.exists():
        print("Downloading WM-811K from Kaggle ...")
        result = subprocess.run(
            [
                "kaggle",
                "datasets", "download",
                "-d", KAGGLE_DATASET,
                "-p", str(raw_dir),
            ],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print("Kaggle download failed:")
            print(result.stderr)
            _print_manual_instructions(raw_dir)
            sys.exit(1)
        print("Download complete.")

    pkl_path = raw_dir / PKL_FILENAME
    if not pkl_path.exists():
        print("Extracting archive ...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(raw_dir)

    if not pkl_path.exists():
        raise FileNotFoundError(
            f"{PKL_FILENAME} not found in {raw_dir}. "
            "The Kaggle zip may have a different layout — "
            "check the contents and pass --pkl explicitly."
        )
    return pkl_path


def _print_manual_instructions(raw_dir: Path) -> None:
    print("\n" + "=" * 60)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("=" * 60)
    print("1. Go to: https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map")
    print("2. Click 'Download' (you need a free Kaggle account)")
    print(f"3. Extract LSWMD.pkl into: {raw_dir.resolve()}")
    print(f"4. Re-run: python scripts/download_wm811k.py --pkl {raw_dir / PKL_FILENAME}")
    print("=" * 60)


# ── data loading ──────────────────────────────────────────────────────────────

def load_pkl(pkl_path: Path) -> list[dict]:
    print(f"Loading {pkl_path} ...")

    import pandas as pd
    import types

    # WM-811K was pickled with old pandas; remap all removed submodules to avoid AttributeError
    _base = sys.modules.get("pandas.core.indexes.base")
    for old_mod in [
        "pandas.indexes",
        "pandas.indexes.base",
        "pandas.indexes.numeric",
        "pandas.indexes.range",
        "pandas.indexes.frozen",
        "pandas.indexes.category",
        "pandas.indexes.interval",
        "pandas.indexes.multi",
        "pandas.indexes.period",
        "pandas.indexes.timedeltas",
        "pandas.indexes.datetimes",
    ]:
        if old_mod not in sys.modules:
            mod = types.ModuleType(old_mod)
            if _base:
                # copy all attributes from pandas.core.indexes.base as fallback
                for attr in dir(_base):
                    try:
                        setattr(mod, attr, getattr(_base, attr))
                    except Exception:
                        pass
            sys.modules[old_mod] = mod

    # pandas.read_pickle handles encoding and compat better than raw pickle.load
    data = pd.read_pickle(pkl_path)

    # LSWMD.pkl is a DataFrame — convert to list of dicts
    if hasattr(data, "to_dict"):
        records = data.to_dict("records")
    else:
        records = list(data)

    print(f"  Total records: {len(records)}")
    return records


def _wafer_map_to_image(wmap: np.ndarray) -> Image.Image:
    """
    Convert a WM-811K wafer map array to a 256×256 RGB image.
      0 (outside) → black
      1 (good)    → dark green
      2 (defect)  → red
    """
    h, w = wmap.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[wmap == 1] = [30, 120, 50]
    rgb[wmap == 2] = [200, 40, 40]
    img = Image.fromarray(rgb, mode="RGB")
    img = img.resize(TARGET_SIZE, Image.NEAREST)
    return img


def _extract_label(record: dict) -> str | None:
    """Extract the string failure-type label from a WM-811K record."""
    ft = record.get("failureType")
    if ft is None:
        return None
    # Shape is typically [[label]] or [[]] for unlabelled
    if hasattr(ft, "tolist"):
        ft = ft.tolist()
    if isinstance(ft, list):
        while isinstance(ft, list):
            if len(ft) == 0:
                return None
            ft = ft[0]
    if isinstance(ft, str):
        return ft
    return None


# ── main processing ───────────────────────────────────────────────────────────

def process_dataset(
    pkl_path: Path,
    output_dir: Path,
    n_per_class: int = 20,
    seed: int = 42,
) -> None:
    records = load_pkl(pkl_path)

    rng = np.random.default_rng(seed)

    # Group records by mapped defect class
    buckets: dict[str, list[dict]] = {dc: [] for dc in set(WM811K_TO_DEFECT_CLASS.values())}
    skipped = 0

    print("Indexing records by defect class ...")
    for rec in records:
        label = _extract_label(rec)
        if label is None or label not in WM811K_TO_DEFECT_CLASS:
            skipped += 1
            continue
        defect_class = WM811K_TO_DEFECT_CLASS[label]
        buckets[defect_class].append(rec)

    print(f"  Skipped (unlabelled): {skipped}")
    for dc, recs in buckets.items():
        print(f"  {dc}: {len(recs)} records available")

    img_dir = output_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    annotations = []

    print(f"\nSampling {n_per_class} images per class and saving ...")
    for defect_class, recs in buckets.items():
        if not recs:
            print(f"  WARNING: no records for {defect_class} — skipping")
            continue

        indices = rng.choice(len(recs), size=min(n_per_class, len(recs)), replace=False)
        print(f"  {defect_class}: {len(indices)} images ...", end=" ")

        for i, idx in enumerate(indices):
            rec = recs[int(idx)]
            wmap = rec["waferMap"]
            img = _wafer_map_to_image(wmap)

            filename = f"{defect_class}_{i:02d}.png"
            img_path = img_dir / filename
            img.save(img_path)

            annotations.append({
                "image_path": str(img_path.as_posix()),
                "defect_class": defect_class,
                "modality": "wafer_map",
                "description": DESCRIPTIONS.get(defect_class, ""),
                "equipment_id": f"FAB-EQ-{(i % 5) + 1:02d}",
                "lot_id": f"LOT-WM811K-{(i // 5) + 1:03d}",
                "wafer_id": f"W{i + 1:02d}",
                "severity": SEVERITY_MAP.get(defect_class, "MINOR"),
                "source": "wm811k",
                "original_label": _extract_label(rec),
            })

        print("done")

    ann_path = output_dir / "annotations.jsonl"
    with open(ann_path, "w") as f:
        for entry in annotations:
            f.write(json.dumps(entry) + "\n")

    print(f"\nDataset saved to: {output_dir.resolve()}")
    print(f"  {len(annotations)} images across {len(buckets)} classes")
    print(f"  Annotations: {ann_path}")

    # Print class distribution
    print("\nClass distribution:")
    class_counts: dict[str, int] = {}
    for a in annotations:
        class_counts[a["defect_class"]] = class_counts.get(a["defect_class"], 0) + 1
    for dc, cnt in sorted(class_counts.items()):
        print(f"  {dc:<25} {cnt}")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and preprocess WM-811K wafer map dataset")
    parser.add_argument(
        "--pkl",
        default=None,
        help="Path to LSWMD.pkl if already downloaded. "
             "If omitted, will download via Kaggle API.",
    )
    parser.add_argument(
        "--output",
        default="data/wm811k",
        help="Output directory for processed images + annotations.jsonl",
    )
    parser.add_argument(
        "--raw-dir",
        default="data/wm811k_raw",
        help="Directory to store downloaded Kaggle zip + extracted files",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=20,
        help="Number of images to sample per defect class (default: 20)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling",
    )
    args = parser.parse_args()

    if args.pkl:
        pkl_path = Path(args.pkl)
        if not pkl_path.exists():
            print(f"ERROR: {pkl_path} does not exist.")
            sys.exit(1)
    else:
        try:
            import kaggle  # noqa: F401 — check import before subprocess
        except ImportError:
            print("kaggle package not installed. Run: pip install kaggle")
            _print_manual_instructions(Path(args.raw_dir))
            sys.exit(1)
        pkl_path = _download_via_kaggle(Path(args.raw_dir))

    process_dataset(
        pkl_path=pkl_path,
        output_dir=Path(args.output),
        n_per_class=args.n,
        seed=args.seed,
    )
