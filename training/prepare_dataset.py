"""Prepare the QLoRA fine-tuning dataset for LLaVA-1.6.

Dataset format expected:
  data/raw/
    images/          ← .png / .jpg inspection images
    annotations.jsonl ← one JSON record per line:
      {
        "image": "images/sem_001.png",
        "defect_type": "scratch",
        "description": "A linear scratch defect...",
        "root_cause": "Wafer handling contact...",
        "severity": "MAJOR"
      }

Output: data/processed/train.jsonl, val.jsonl
Each record in LLaVA instruction-tuning format:
  {
    "id": "...",
    "image": "images/sem_001.png",
    "conversations": [
      {"from": "human", "value": "<image>\nDescribe the defect in this image."},
      {"from": "gpt",   "value": "A linear scratch defect..."}
    ]
  }
"""

from __future__ import annotations

import json
import random
from pathlib import Path


DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ANNOTATIONS_FILE = RAW_DIR / "annotations.jsonl"

TRAIN_SPLIT = 0.85
RANDOM_SEED = 42

# Question templates for instruction diversity
QUESTION_TEMPLATES = [
    "Describe the defect visible in this semiconductor inspection image.",
    "What type of defect is present and where is it located?",
    "Analyse this inspection image and identify any defect patterns.",
    "What is the morphology of the defect in this {modality} image?",
    "Examine the defect and describe its characteristics in detail.",
]

SEVERITY_QUESTIONS = [
    "What is the severity of the defect shown? Explain your reasoning.",
    "Classify this defect as Critical, Major, Minor, or None with justification.",
]

ROOT_CAUSE_QUESTIONS = [
    "What is the likely root cause of this semiconductor defect?",
    "Based on the defect pattern, what process failure could have caused this?",
]


def load_annotations(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_conversation(record: dict) -> list[dict]:
    """Generate multi-turn instruction-tuning conversations for one image."""
    conversations = []
    defect_type = record.get("defect_type", "unknown")
    description = record.get("description", "")
    root_cause = record.get("root_cause", "")
    severity = record.get("severity", "MINOR")
    modality = record.get("modality", "optical")

    # Turn 1: defect description
    q = random.choice(QUESTION_TEMPLATES).format(modality=modality)
    conversations.append({"from": "human", "value": f"<image>\n{q}"})
    conversations.append({"from": "gpt", "value": description})

    # Turn 2: severity (50% chance)
    if random.random() > 0.5 and severity:
        q2 = random.choice(SEVERITY_QUESTIONS)
        a2 = (
            f"SEVERITY: {severity}\n"
            f"This defect is classified as {severity} because: {description[:150]}..."
        )
        conversations.append({"from": "human", "value": q2})
        conversations.append({"from": "gpt", "value": a2})

    # Turn 3: root cause (50% chance)
    if random.random() > 0.5 and root_cause:
        q3 = random.choice(ROOT_CAUSE_QUESTIONS)
        conversations.append({"from": "human", "value": q3})
        conversations.append({"from": "gpt", "value": root_cause})

    return conversations


def prepare_dataset() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if not ANNOTATIONS_FILE.exists():
        print(f"[WARN] Annotations file not found at {ANNOTATIONS_FILE}.")
        print("Generating synthetic demo dataset…")
        _generate_synthetic_demo()
        return

    records = load_annotations(ANNOTATIONS_FILE)
    random.seed(RANDOM_SEED)
    random.shuffle(records)

    split_idx = int(len(records) * TRAIN_SPLIT)
    train_records = records[:split_idx]
    val_records = records[split_idx:]

    for split_name, split_data in [("train", train_records), ("val", val_records)]:
        output = []
        for i, rec in enumerate(split_data):
            conversations = build_conversation(rec)
            output.append({
                "id": f"{split_name}_{i:05d}",
                "image": rec["image"],
                "conversations": conversations,
            })
        out_path = PROCESSED_DIR / f"{split_name}.jsonl"
        with open(out_path, "w") as f:
            for entry in output:
                f.write(json.dumps(entry) + "\n")
        print(f"[OK] Wrote {len(output)} records to {out_path}")


def _generate_synthetic_demo() -> None:
    """Create a minimal synthetic dataset for pipeline testing."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    defect_types = ["scratch", "particle", "edge_crack", "center_cluster"]
    severities = ["CRITICAL", "MAJOR", "MINOR", "NONE"]
    modalities = ["sem", "optical", "wafer_map"]

    random.seed(RANDOM_SEED)
    records = []
    for i in range(200):
        dtype = random.choice(defect_types)
        sev = random.choice(severities)
        mod = random.choice(modalities)
        records.append({
            "id": f"synthetic_{i:04d}",
            "image": f"images/synthetic_{i:04d}.png",
            "conversations": [
                {
                    "from": "human",
                    "value": f"<image>\nDescribe the defect in this {mod} image.",
                },
                {
                    "from": "gpt",
                    "value": (
                        f"This {mod} image shows a {dtype} defect. "
                        f"The defect exhibits characteristics consistent with "
                        f"a {sev.lower()} process excursion."
                    ),
                },
            ],
        })

    split_idx = int(len(records) * TRAIN_SPLIT)
    for split_name, split_data in [
        ("train", records[:split_idx]),
        ("val", records[split_idx:]),
    ]:
        out_path = PROCESSED_DIR / f"{split_name}.jsonl"
        with open(out_path, "w") as f:
            for entry in split_data:
                f.write(json.dumps(entry) + "\n")
        print(f"[OK] Synthetic {split_name}: {len(split_data)} records → {out_path}")


if __name__ == "__main__":
    prepare_dataset()
