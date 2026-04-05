"""
Download and process the MixedWM38 wafer map dataset.

MixedWM38: 38,015 wafer maps, 38 defect pattern classes (8 single + 29 mixed + 1 normal)
Source: https://github.com/Junliangwangdhu/WaferMap
        https://www.kaggle.com/datasets/co1d7era/mixedtype-wafer-defect-datasets

Maps are (52x52) numpy arrays:
  0 = outside wafer
  1 = normal die
  2 = defective die

This script:
  1. Downloads MixedWM38 via Kaggle API or manual path
  2. Loads and decodes the pickle file
  3. Samples n images per single-pattern class (maps to our 9 classes)
  4. Saves as 256x256 PNG + annotations.jsonl

Usage:
    # Via Kaggle API:
    python scripts/download_mixedwm38.py

    # Via manually downloaded file:
    python scripts/download_mixedwm38.py --pkl path/to/MixedWM38.pkl

    # Via cloned GitHub repo:
    python scripts/download_mixedwm38.py --pkl data/mixedwm38_raw/WaferMap/WM38_Dataset_and_Code/MWMD.pkl
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

# ── Class mapping: MixedWM38 → our 9 defect classes ──────────────────────────
# MixedWM38 uses string labels for single-type patterns
MIXEDWM38_TO_DEFECT_CLASS = {
    "normal":    "no_defect",
    "Center":    "center_cluster",
    "Donut":     "ring_pattern",
    "Edge-Loc":  "edge_crack",
    "Edge-Ring": "ring_pattern",
    "Loc":       "local_cluster",
    "Random":    "random_defects",
    "Scratch":   "scratch",
    "Near-full": "near_full_wafer",
}

# Descriptions per class (realistic, varied from WM-811K descriptions)
DESCRIPTIONS = {
    "no_defect": (
        "Wafer map from MixedWM38 showing a Normal pattern: uniform die yield "
        "distribution across the entire wafer surface with no systematic defect "
        "signature. Good die fraction exceeds 98%."
    ),
    "center_cluster": (
        "Wafer map from MixedWM38 showing a Center defect pattern: "
        "defective die cluster concentrated within 30% of wafer radius from center, "
        "characteristic of process non-uniformity at the wafer center — "
        "commonly from CMP over-polish, CVD deposition bowl effect, or thermal gradient."
    ),
    "ring_pattern": (
        "Wafer map from MixedWM38 showing a ring/donut defect pattern: "
        "defective die arranged in a concentric annular band, associated with "
        "spin-coat non-uniformity, edge-bead removal variation, or "
        "annular temperature gradient during thermal processing."
    ),
    "edge_crack": (
        "Wafer map from MixedWM38 showing an Edge-Loc defect pattern: "
        "defective die localised near the wafer edge in a spatially confined region, "
        "consistent with dicing stress, edge-handling damage, or localised "
        "deposition non-uniformity at one edge."
    ),
    "local_cluster": (
        "Wafer map from MixedWM38 showing a Loc defect pattern: "
        "a discrete localised cluster of defective die in the interior wafer region, "
        "not at the edge or center, suggesting a contamination event, "
        "chuck particle, or localised plasma non-uniformity."
    ),
    "random_defects": (
        "Wafer map from MixedWM38 showing a Random defect pattern: "
        "stochastically distributed defective die with no systematic spatial "
        "correlation, characteristic of random particle contamination events "
        "or low-level background defectivity."
    ),
    "scratch": (
        "Wafer map from MixedWM38 showing a Scratch defect pattern: "
        "defective die arranged in a linear track across the wafer surface, "
        "characteristic of mechanical contact damage during wafer handling, "
        "chuck loading, or end-effector malfunction."
    ),
    "near_full_wafer": (
        "Wafer map from MixedWM38 showing a Near-full defect pattern: "
        "nearly the entire active wafer area covered by defective die, "
        "indicating a catastrophic process failure — typically a severe "
        "chemistry excursion, complete etch non-uniformity, or gross particle event."
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
KAGGLE_DATASET = "co1d7era/mixedtype-wafer-defect-datasets"


def _download_via_kaggle(raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "mixedtype-wafer-defect-datasets.zip"

    if not zip_path.exists():
        print("Downloading MixedWM38 from Kaggle ...")
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", str(raw_dir)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print("Kaggle download failed:")
            print(result.stderr)
            _print_manual_instructions(raw_dir)
            sys.exit(1)

    # Extract
    pkl_candidates = list(raw_dir.rglob("*.pkl"))
    if not pkl_candidates:
        print("Extracting archive ...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(raw_dir)
        pkl_candidates = list(raw_dir.rglob("*.pkl"))

    if not pkl_candidates:
        _print_manual_instructions(raw_dir)
        raise FileNotFoundError("No .pkl file found after extraction.")

    return pkl_candidates[0]


def _print_manual_instructions(raw_dir: Path) -> None:
    print("\n" + "=" * 60)
    print("MANUAL DOWNLOAD INSTRUCTIONS — MixedWM38")
    print("=" * 60)
    print("Option A — Kaggle:")
    print("  1. Go to https://www.kaggle.com/datasets/co1d7era/mixedtype-wafer-defect-datasets")
    print("  2. Download the dataset zip")
    print(f"  3. Extract .pkl file to: {raw_dir.resolve()}")
    print("Option B — GitHub:")
    print("  git clone https://github.com/Junliangwangdhu/WaferMap.git data/mixedwm38_raw")
    print("  Then pass: --pkl data/mixedwm38_raw/WM38_Dataset_and_Code/MWMD.pkl")
    print("=" * 60)


def _wafer_to_image(wmap: np.ndarray) -> Image.Image:
    """Convert MixedWM38 array to 256x256 RGB image."""
    if wmap.ndim == 3:
        wmap = wmap[:, :, 0]
    wmap = wmap.astype(np.uint8)
    h, w = wmap.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[wmap == 1] = [30, 120, 50]   # good die → dark green
    rgb[wmap == 2] = [200, 40, 40]   # defective die → red
    img = Image.fromarray(rgb, mode="RGB")
    return img.resize(TARGET_SIZE, Image.NEAREST)


def load_mixedwm38(pkl_path: Path) -> tuple[list[np.ndarray], list[str]]:
    """Load MixedWM38 file (.npz or .pkl). Returns (wafer_maps, labels)."""
    print(f"Loading {pkl_path} ...")

    suffix = pkl_path.suffix.lower()

    # ── NPZ (numpy archive) ────────────────────────────────���─────────────────
    if suffix == ".npz":
        data = np.load(pkl_path, allow_pickle=True)
        print(f"  NPZ keys: {list(data.keys())}")

        # MixedWM38 format: arr_0 = wafer maps (N,52,52), arr_1 = one-hot labels (N,8)
        wafer_maps_raw = data["arr_0"]   # shape (N, 52, 52) or (N, 52, 52, 1)
        labels_onehot  = data["arr_1"]   # shape (N, 8) — 8 basic defect types

        # One-hot → class name (columns = C2..C9 = Center,Donut,Edge-Loc,Edge-Ring,Loc,Random,Scratch,Near-full)
        # If all zeros → normal
        CLASS_NAMES = ["Center", "Donut", "Edge-Loc", "Edge-Ring",
                       "Loc", "Random", "Scratch", "Near-full"]

        maps = []
        labels = []
        for i in range(len(wafer_maps_raw)):
            wmap = wafer_maps_raw[i]
            if wmap.ndim == 3:
                wmap = wmap[:, :, 0]
            maps.append(wmap)

            oh = labels_onehot[i]
            # For single-type: exactly one bit set
            # For mixed: multiple bits set → skip (we only want single-pattern)
            bits = np.where(oh > 0)[0]
            if len(bits) == 0:
                labels.append("normal")
            elif len(bits) == 1:
                labels.append(CLASS_NAMES[bits[0]])
            else:
                # Mixed pattern — label as "mixed" (will be skipped in bucketing)
                names = "+".join(CLASS_NAMES[b] for b in bits)
                labels.append(f"mixed:{names}")

        print(f"  Loaded {len(maps)} records from NPZ")
        single = sum(1 for l in labels if not l.startswith("mixed"))
        mixed  = sum(1 for l in labels if l.startswith("mixed"))
        print(f"  Single-pattern: {single}  |  Mixed: {mixed}")
        return maps, labels

    # ── Pickle fallback ────────────────────��─────────────────────────��───────
    import pickle
    try:
        import pandas as pd
        df = pd.read_pickle(pkl_path)
        if hasattr(df, "to_dict"):
            records = df.to_dict("records")
            maps = [r.get("waferMap", r.get("WaferMap")) for r in records]
            labels = []
            for r in records:
                ft = r.get("failureType", r.get("label", "normal"))
                if hasattr(ft, "tolist"):
                    ft = ft.tolist()
                while isinstance(ft, list):
                    ft = ft[0] if ft else "normal"
                labels.append(str(ft) if ft else "normal")
            print(f"  Loaded {len(maps)} records via pandas")
            return maps, labels
    except Exception as e:
        print(f"  pandas failed ({e}), trying raw pickle ...")

    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    if isinstance(data, dict):
        maps = list(data.get("wafer_maps", data.get("WaferMap", list(data.values())[0])))
        labels = list(data.get("labels", data.get("label", ["unknown"] * len(maps))))
    elif isinstance(data, (list, np.ndarray)):
        maps   = [item[0] if isinstance(item, (list, tuple)) else item for item in data]
        labels = [item[1] if isinstance(item, (list, tuple)) else "unknown" for item in data]
    else:
        raise ValueError(f"Unknown structure: {type(data)}")

    print(f"  Loaded {len(maps)} records")
    return maps, labels


def process_dataset(pkl_path: Path, output_dir: Path, n_per_class: int, seed: int) -> None:
    wafer_maps, raw_labels = load_mixedwm38(pkl_path)
    rng = np.random.default_rng(seed)

    # Bucket by mapped defect class (single-pattern only)
    buckets: dict[str, list[int]] = {dc: [] for dc in set(MIXEDWM38_TO_DEFECT_CLASS.values())}
    skipped = 0
    for idx, label in enumerate(raw_labels):
        label_str = str(label).strip()
        if label_str not in MIXEDWM38_TO_DEFECT_CLASS:
            skipped += 1
            continue
        dc = MIXEDWM38_TO_DEFECT_CLASS[label_str]
        buckets[dc].append(idx)

    print(f"  Skipped (mixed/unlabelled): {skipped}")
    for dc, idxs in sorted(buckets.items()):
        print(f"  {dc:<25} {len(idxs)} available")

    img_dir = output_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    annotations = []

    print(f"\nSampling up to {n_per_class} per class ...")
    for dc, idxs in buckets.items():
        if not idxs:
            print(f"  WARNING: no records for {dc}")
            continue
        chosen = rng.choice(len(idxs), size=min(n_per_class, len(idxs)), replace=False)
        print(f"  {dc}: {len(chosen)} images ...", end=" ")
        for i, ci in enumerate(chosen):
            wmap = wafer_maps[idxs[int(ci)]]
            if wmap is None:
                continue
            try:
                wmap_arr = np.array(wmap)
                img = _wafer_to_image(wmap_arr)
            except Exception as e:
                print(f"(skip: {e})", end=" ")
                continue

            fname = f"mixedwm38_{dc}_{i:03d}.png"
            img.save(img_dir / fname)
            annotations.append({
                "image_path": str((img_dir / fname).as_posix()),
                "defect_class": dc,
                "modality": "wafer_map",
                "description": DESCRIPTIONS.get(dc, ""),
                "equipment_id": f"FAB-EQ-{(i % 5) + 1:02d}",
                "lot_id": f"LOT-MWM38-{(i // 5) + 1:03d}",
                "wafer_id": f"W{i + 1:02d}",
                "severity": SEVERITY_MAP.get(dc, "MINOR"),
                "source": "mixedwm38",
                "original_label": raw_labels[idxs[int(ci)]],
            })
        print("done")

    ann_path = output_dir / "annotations.jsonl"
    mode = "a" if ann_path.exists() else "w"
    with open(ann_path, mode) as f:
        for entry in annotations:
            f.write(json.dumps(entry) + "\n")

    print(f"\nMixedWM38 saved to: {output_dir.resolve()}")
    print(f"  {len(annotations)} images written")
    print(f"  Annotations {'appended to' if mode == 'a' else 'written to'}: {ann_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", default=None)
    parser.add_argument("--output", default="data/mixedwm38")
    parser.add_argument("--raw-dir", default="data/mixedwm38_raw")
    parser.add_argument("--n", type=int, default=40,
                        help="Images per class (default 40 → 320 total)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.pkl:
        pkl_path = Path(args.pkl)
    else:
        try:
            import kaggle  # noqa
        except ImportError:
            print("kaggle not installed: pip install kaggle")
            _print_manual_instructions(Path(args.raw_dir))
            sys.exit(1)
        pkl_path = _download_via_kaggle(Path(args.raw_dir))

    process_dataset(pkl_path, Path(args.output), args.n, args.seed)
