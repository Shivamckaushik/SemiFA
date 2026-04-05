"""
SemiFA-FA Dataset Builder — GPT-4o Vision FA narrative generator.

Builds the highest-quality possible semiconductor FA training dataset by:
  1. Loading all available images (synthetic + WM-811K + MixedWM38)
  2. Sending each image to GPT-4o Vision with an expert FA prompt
  3. Generating 3-turn LLaVA conversation format training examples
  4. Saving to data/processed/train.jsonl and val.jsonl

Why GPT-4o Vision:
  - Sees the actual image content, not just the class label
  - Generates varied, non-templated FA text
  - Produces semiconductor-accurate terminology and specific values
  - Each narrative is unique even for same-class images

Output format (LLaVA QLoRA):
  {
    "id": "semifa_00001",
    "image": "data/synthetic_dataset/images/scratch_00.png",
    "conversations": [
      {"from": "human", "value": "<image>\n[FA question]"},
      {"from": "gpt",   "value": "[expert FA response]"},
      {"from": "human", "value": "[follow-up question]"},
      {"from": "gpt",   "value": "[follow-up answer]"},
      ...
    ]
  }

Usage:
    python scripts/build_semifa_dataset.py --api-key sk-...
    python scripts/build_semifa_dataset.py  # uses OPENAI_API_KEY env var
    python scripts/build_semifa_dataset.py --max-images 200 --workers 4
    python scripts/build_semifa_dataset.py --resume  # skip already-processed images

Cost estimate:
    ~900 images x 2 API calls (generate + parse) x ~$0.003/call = ~$5.40 total
    Use --model gpt-4o-mini to reduce cost to ~$0.30 (slightly lower quality)
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import openai
from PIL import Image

# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a senior semiconductor failure analysis engineer with 15 years of experience at leading semiconductor fabs. You are an expert in:
- SEM (Scanning Electron Microscopy) image interpretation
- Wafer map defect pattern analysis and spatial statistics
- SECS/GEM equipment alarm correlation
- Root cause analysis for yield loss events
- Failure Analysis report writing per SEMI standards

Your responses are technically precise, use correct semiconductor terminology, and cite specific equipment parameters and process steps. You write like an experienced FA engineer, not like a general AI assistant."""

FA_PROMPT_TEMPLATE = """Analyze this semiconductor inspection image and provide a complete Failure Analysis report.

Image type: {modality}
Known defect class: {defect_class}
Confidence: {confidence}
Equipment ID: {equipment_id}
Lot ID: {lot_id}
Wafer ID: {wafer_id}

Recent SECS/GEM equipment alarms (last 2 hours):
{equipment_logs}

Provide a complete FA report in EXACTLY this format:

DEFECT DESCRIPTION:
[3-4 sentences. Describe what you observe: spatial distribution, morphology, approximate defect density, affected die count estimate, specific visual characteristics. Be specific to what you see in this image.]

ROOT CAUSE HYPOTHESES:
HYPOTHESIS 1: [Most likely specific root cause — name the process step and equipment parameter]
EVIDENCE: [What in this image and the equipment logs supports this hypothesis]

HYPOTHESIS 2: [Second most likely cause]
EVIDENCE: [Supporting evidence]

HYPOTHESIS 3: [Third possibility, possibly systemic]
EVIDENCE: [Supporting evidence]

ROOT CAUSE SUMMARY: [1-2 sentences naming the primary cause with confidence level]

SEVERITY: [CRITICAL / MAJOR / MINOR / NONE]
ESTIMATED YIELD IMPACT: [X%]
SEVERITY REASONING: [1-2 sentences justifying the severity based on affected die area and defect mechanism]

CORRECTIVE ACTIONS:
1. [Specific action with parameter values, e.g. "Reduce end-effector vacuum threshold from 80 mbar to 65 mbar and re-qualify transfer recipe"]
2. [Second specific action]
3. [Third action — preventive measure]

EQUIPMENT FOCUS: [Primary equipment type and process step most likely responsible]"""

# Follow-up question pool — randomly selected per image for training diversity
SEVERITY_QUESTIONS = [
    "What severity classification does this defect warrant, and why?",
    "Classify this defect as CRITICAL, MAJOR, MINOR, or NONE with justification.",
    "What is the expected yield impact of this defect pattern?",
    "How urgent is the corrective action required for this defect?",
]

ROOT_CAUSE_QUESTIONS = [
    "What is the most likely root cause of this defect pattern?",
    "Which equipment and process step is most likely responsible for this defect?",
    "Based on the spatial pattern, what process failure mechanism caused this?",
    "What equipment alarm history would you expect to correlate with this defect?",
]

CORRECTIVE_ACTION_QUESTIONS = [
    "What corrective actions should be taken immediately for this defect?",
    "Describe the process parameter adjustments needed to prevent recurrence.",
    "What is the recommended engineering response to this failure?",
    "List the containment and corrective actions for this defect in priority order.",
]


# ── Realistic equipment log generator ────────────────────────────────────────

EQUIPMENT_SCENARIOS = {
    "scratch": [
        ["08:12:01 | end_effector_vacuum=55mbar [ALARM: VAC_LOW]",
         "08:11:45 | wafer_handler_speed=0.48m/s [ALARM: SPEED_HIGH]",
         "08:11:30 | chuck_pressure=2.9bar [ALARM: PRESSURE_HIGH]"],
        ["14:23:10 | robot_arm_torque=3.8Nm [ALARM: TORQUE_EXCEED]",
         "14:22:55 | wafer_sensor_edge=triggered [ALARM: EDGE_DETECT]",
         "14:22:40 | end_effector_vacuum=48mbar [ALARM: VAC_CRITICAL]"],
    ],
    "particle_contamination": [
        ["10:15:20 | cvd_chamber_pressure=4.2mTorr [ALARM: PRESSURE_SPIKE]",
         "10:14:40 | susceptor_temp=652C",
         "10:14:10 | chamber_clean_cycles=187 [ALARM: CLEAN_OVERDUE]"],
        ["09:30:05 | particle_counter_0.1um=842 [ALARM: PARTICLE_HIGH]",
         "09:29:50 | fan_filter_unit_dp=28Pa [ALARM: FFU_DP_HIGH]",
         "09:29:30 | ambient_humidity=68% [ALARM: HUMIDITY_HIGH]"],
    ],
    "edge_crack": [
        ["09:30:10 | dicing_blade_rpm=28500rpm",
         "09:29:55 | coolant_flow_rate=0.8L/min [ALARM: FLOW_LOW]",
         "09:29:40 | blade_wear_counter=42150 [ALARM: BLADE_WEAR]"],
        ["11:05:00 | edge_ring_clamp_force=12.4N [ALARM: CLAMP_LOW]",
         "11:04:45 | wafer_edge_sensor=0.31mm [ALARM: TILT_DETECT]",
         "11:04:30 | load_lock_pressure=8.2mTorr [ALARM: LL_PRESSURE]"],
    ],
    "center_cluster": [
        ["12:20:10 | cmp_downforce=3.1psi [ALARM: FORCE_HIGH]",
         "12:19:55 | platen_speed=87rpm",
         "12:19:20 | pad_condition_index=0.62 [ALARM: PAD_WORN]"],
        ["08:45:00 | cvd_deposition_rate_center=2.8nm/min [ALARM: RATE_HIGH]",
         "08:44:45 | showerhead_temp=412C [ALARM: TEMP_NONUNIFORM]",
         "08:44:20 | process_pressure=6.3Torr"],
    ],
    "local_cluster": [
        ["15:10:30 | plasma_density_zone3=2.1e11/cm3 [ALARM: NONUNIFORM]",
         "15:10:15 | rf_power_zone3=1850W [ALARM: POWER_DEVIATION]",
         "15:10:00 | etch_rate_zone3=245nm/min [ALARM: RATE_LOW]"],
        ["10:55:20 | robot_teach_offset=+0.42mm [ALARM: POSITION_DRIFT]",
         "10:55:05 | chuck_temp_zone2=22.8C [ALARM: TEMP_NONUNIFORM]",
         "10:54:50 | helium_backside_pressure=8.1Torr"],
    ],
    "ring_pattern": [
        ["11:05:30 | spin_speed=2200rpm [ALARM: SPEED_UNSTABLE]",
         "11:04:50 | ambient_humidity=62% [ALARM: HUMIDITY_HIGH]",
         "11:04:30 | exhaust_flow=145CFM [ALARM: EXHAUST_LOW]"],
        ["14:30:10 | anneal_temp_edge=852C [ALARM: TEMP_HIGH]",
         "14:29:55 | anneal_temp_center=821C",
         "14:29:40 | quartz_tube_condition=0.71 [ALARM: TUBE_WORN]"],
    ],
    "random_defects": [
        ["Various | ambient_particle_0.3um=124/ft3 [ALARM: PARTICLE_ELEVATED]",
         "Various | minienvironment_pressure=+0.05Pa [ALARM: PRESSURE_LOW]",
         "Various | smif_door_cycles=8847"],
        ["Continuous | process_particle_count=38/wafer [ALARM: YIELD_RISK]",
         "Continuous | robot_wrist_vibration=0.8g [ALARM: VIBRATION]",
         "Continuous | ionizer_balance=+0.4V [ALARM: CHARGE_IMBALANCE]"],
    ],
    "near_full_wafer": [
        ["09:00:15 | etch_chemistry_flow=0L/min [ALARM: CHEMISTRY_EMPTY]",
         "09:00:10 | process_aborted=TRUE [ALARM: RECIPE_ABORT]",
         "08:59:55 | chamber_pressure=892mTorr [ALARM: PRESSURE_RUNAWAY]"],
        ["13:15:20 | cvd_rf_power=0W [ALARM: RF_LOSS]",
         "13:15:10 | process_gas_flow=0sccm [ALARM: GAS_FLOW_FAIL]",
         "13:15:05 | wafer_temp=890C [ALARM: OVERTEMP_CRITICAL]"],
    ],
    "no_defect": [
        ["All parameters within spec. No alarms in last 2 hours.",
         "Last PM: 3 days ago. Equipment health score: 97/100."],
        ["No alarms. Normal process window. Yield target met.",
         "Process CPK: 1.42. Equipment utilization: 87%."],
    ],
}


def get_equipment_logs(defect_class: str, seed: int) -> str:
    rng = random.Random(seed)
    scenarios = EQUIPMENT_SCENARIOS.get(defect_class, EQUIPMENT_SCENARIOS["random_defects"])
    scenario = rng.choice(scenarios)
    return "\n".join(f"  {line}" for line in scenario)


# ── Image loading ─────────────────────────────────────────────────────────────

def image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ── GPT-4o Vision call ────────────────────────────────────────────────────────

def generate_fa_narrative(
    client: openai.OpenAI,
    model: str,
    record: dict,
    seed: int,
) -> str | None:
    """Send image to GPT-4o Vision and get complete FA narrative."""
    try:
        b64 = image_to_base64(record["image_path"])
        equipment_logs = get_equipment_logs(record["defect_class"], seed)
        prompt = FA_PROMPT_TEMPLATE.format(
            modality=record.get("modality", "wafer_map"),
            defect_class=record["defect_class"],
            confidence=f"{random.uniform(78, 97):.1f}%",
            equipment_id=record.get("equipment_id", "FAB-EQ-01"),
            lot_id=record.get("lot_id", "LOT-2024-001"),
            wafer_id=record.get("wafer_id", "W01"),
            equipment_logs=equipment_logs,
        )
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}},
                    {"type": "text", "text": prompt},
                ]},
            ],
            max_tokens=1200,
            temperature=0.7,
            seed=seed,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"\n  [ERROR] {record['image_path']}: {e}")
        return None


def build_conversations(record: dict, fa_narrative: str, seed: int) -> list[dict]:
    """Build multi-turn LLaVA conversation from FA narrative."""
    rng = random.Random(seed)
    conversations = []

    # Turn 1: Full FA analysis (primary training signal)
    primary_questions = [
        "Analyze this semiconductor inspection image and provide a complete failure analysis report.",
        "You are a semiconductor FA engineer. Analyze this wafer inspection image and generate a structured FA report.",
        "Examine this semiconductor defect image and provide: defect description, root cause hypotheses, severity classification, and corrective actions.",
        f"This is a {record.get('modality', 'wafer_map')} semiconductor inspection image. Perform a complete failure analysis.",
    ]
    conversations.append({"from": "human", "value": f"<image>\n{rng.choice(primary_questions)}"})
    conversations.append({"from": "gpt", "value": fa_narrative})

    # Turn 2: Follow-up on severity or root cause (adds training diversity)
    follow_up_pool = SEVERITY_QUESTIONS + ROOT_CAUSE_QUESTIONS + CORRECTIVE_ACTION_QUESTIONS
    follow_q = rng.choice(follow_up_pool)

    # Extract relevant section from narrative for focused answer
    answer_hint = ""
    if "SEVERITY" in follow_q.upper() or "YIELD" in follow_q.upper():
        for line in fa_narrative.split("\n"):
            if line.startswith("SEVERITY:") or line.startswith("ESTIMATED YIELD"):
                answer_hint += line + "\n"
        if not answer_hint:
            answer_hint = f"Based on the analysis above, this is a {record.get('severity', 'MAJOR')} defect."
    elif "ROOT CAUSE" in follow_q.upper() or "EQUIPMENT" in follow_q.upper():
        in_rc = False
        for line in fa_narrative.split("\n"):
            if "ROOT CAUSE" in line:
                in_rc = True
            if in_rc:
                answer_hint += line + "\n"
            if in_rc and line.startswith("SEVERITY"):
                break
        if not answer_hint:
            answer_hint = "Based on the equipment logs and defect morphology, the root cause is identified above."
    else:
        for line in fa_narrative.split("\n"):
            if "CORRECTIVE" in line or line.strip().startswith("1.") or line.strip().startswith("2."):
                answer_hint += line + "\n"
        if not answer_hint:
            answer_hint = "The corrective actions are outlined in the FA report above."

    if answer_hint.strip():
        conversations.append({"from": "human", "value": follow_q})
        conversations.append({"from": "gpt", "value": answer_hint.strip()})

    return conversations


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_all_annotations(*jsonl_paths: str) -> list[dict]:
    records = []
    seen_images = set()
    for p in jsonl_paths:
        path = Path(p)
        if not path.exists():
            print(f"  [SKIP] {path} not found")
            continue
        count = 0
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                img = rec.get("image_path", "")
                if img in seen_images:
                    continue
                if not Path(img).exists():
                    continue
                seen_images.add(img)
                records.append(rec)
                count += 1
        print(f"  Loaded {count} records from {path.name}")
    return records


# ── Main ──────────────────────────────────────────────────────────────────────

def main(
    api_key: str | None,
    model: str,
    max_images: int,
    workers: int,
    output_dir: str,
    resume: bool,
    train_split: float,
    seed: int,
) -> None:
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("ERROR: Pass --api-key sk-... or set OPENAI_API_KEY env var.")

    client = openai.OpenAI(api_key=key)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Load all images ──
    print("\nLoading annotations from all sources ...")
    records = load_all_annotations(
        "data/synthetic_dataset/annotations.jsonl",
        "data/wm811k/annotations.jsonl",
        "data/mixedwm38/annotations.jsonl",
    )
    print(f"  Total unique images found: {len(records)}")

    if not records:
        print("\nERROR: No images found. Run these first:")
        print("  python scripts/generate_synthetic_dataset.py --n 50")
        print("  python scripts/download_wm811k.py --n 20")
        print("  python scripts/download_mixedwm38.py --n 40")
        sys.exit(1)

    # Class distribution
    from collections import Counter
    dist = Counter(r["defect_class"] for r in records)
    print("\n  Class distribution:")
    for cls, cnt in sorted(dist.items()):
        print(f"    {cls:<28} {cnt}")

    # Subsample if requested
    random.seed(seed)
    random.shuffle(records)
    if max_images and len(records) > max_images:
        records = records[:max_images]
        print(f"\n  Subsampled to {len(records)} images (--max-images {max_images})")

    # Resume: skip already-processed images
    done_path = out / "done_images.txt"
    done_set: set[str] = set()
    if resume and done_path.exists():
        done_set = set(done_path.read_text().splitlines())
        before = len(records)
        records = [r for r in records if r["image_path"] not in done_set]
        print(f"  Resume: skipping {before - len(records)} already processed images")

    print(f"\n  Images to process: {len(records)}")
    print(f"  Model: {model}")
    print(f"  Workers: {workers}")
    est_cost = len(records) * 0.003 if "mini" not in model else len(records) * 0.0003
    print(f"  Estimated API cost: ~${est_cost:.2f}")
    print(f"  Estimated time: ~{len(records) * 6 / workers / 60:.0f} min\n")

    # ── Generate narratives ──
    results: list[dict] = []
    errors = 0
    lock_path = out / "narratives_raw.jsonl"

    def process_one(i_rec: tuple[int, dict]) -> dict | None:
        i, rec = i_rec
        if rec["image_path"] in done_set:
            return None
        narrative = generate_fa_narrative(client, model, rec, seed=seed + i)
        if not narrative:
            return None
        conversations = build_conversations(rec, narrative, seed=seed + i)
        return {
            "id": f"semifa_{i:05d}",
            "image": rec["image_path"],
            "defect_class": rec["defect_class"],
            "source": rec.get("source", "unknown"),
            "conversations": conversations,
        }

    print("Generating FA narratives with GPT-4o Vision ...")
    with open(lock_path, "a") as lock_f, open(done_path, "a") as done_f:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_one, (i, rec)): rec
                       for i, rec in enumerate(records)}
            for completed, future in enumerate(as_completed(futures), 1):
                rec = futures[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        lock_f.write(json.dumps(result) + "\n")
                        lock_f.flush()
                        done_f.write(rec["image_path"] + "\n")
                        done_f.flush()
                        cls = rec["defect_class"]
                        print(f"  [{completed:>4}/{len(records)}] {cls:<25} {Path(rec['image_path']).name}")
                    else:
                        errors += 1
                except Exception as e:
                    errors += 1
                    print(f"\n  [ERROR] {rec['image_path']}: {e}")

    print(f"\n  Generated: {len(results)}  |  Errors: {errors}")

    # ── Also load previously processed if resuming ──
    if resume and lock_path.exists():
        existing = []
        with open(lock_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        existing.append(json.loads(line))
                    except Exception:
                        pass
        # Deduplicate by id
        seen_ids = {r["id"] for r in results}
        for r in existing:
            if r["id"] not in seen_ids:
                results.append(r)
                seen_ids.add(r["id"])
        print(f"  Total after merge: {len(results)}")

    if not results:
        print("\nERROR: No results generated.")
        sys.exit(1)

    # ── Train/val split ──
    random.seed(seed)
    random.shuffle(results)
    split = int(len(results) * train_split)
    train_data = results[:split]
    val_data = results[split:]

    for name, data in [("train", train_data), ("val", val_data)]:
        path = out / f"{name}.jsonl"
        with open(path, "w") as f:
            for entry in data:
                f.write(json.dumps(entry) + "\n")
        print(f"  Wrote {len(data):>4} records → {path}")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("DATASET BUILD COMPLETE")
    print("=" * 60)
    print(f"  Train: {len(train_data)} examples")
    print(f"  Val:   {len(val_data)} examples")
    print(f"  Total: {len(results)} examples")
    print(f"  Output: {out.resolve()}")
    print()
    print("Class distribution in final dataset:")
    dist2 = Counter(r["defect_class"] for r in results)
    for cls, cnt in sorted(dist2.items()):
        print(f"  {cls:<28} {cnt}")
    print()
    print("=" * 60)
    print("NEXT STEP: Run QLoRA fine-tuning")
    print("=" * 60)
    print("Update training/qlora_finetune.py config:")
    print(f"  train_jsonl = '{out}/train.jsonl'")
    print(f"  val_jsonl   = '{out}/val.jsonl'")
    print(f"  image_root  = '.'  # images use absolute paths")
    print()
    print("Then run on Colab A100:")
    print("  python training/qlora_finetune.py")
    print()
    print("Or on free T4 (slower, ~10 hours):")
    print("  python training/qlora_finetune.py  # with gradient_checkpointing=True")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build SemiFA QLoRA training dataset")
    parser.add_argument("--api-key", default=None, help="OpenAI API key")
    parser.add_argument("--model", default="gpt-4o",
                        help="gpt-4o (best) or gpt-4o-mini (cheaper, ~10x less)")
    parser.add_argument("--max-images", type=int, default=0,
                        help="Cap total images (0 = all). Use 50 for a test run.")
    parser.add_argument("--workers", type=int, default=3,
                        help="Parallel API workers (default 3 to stay within rate limits)")
    parser.add_argument("--output", default="data/processed",
                        help="Output directory for train.jsonl and val.jsonl")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Skip images already in done_images.txt (default: True)")
    parser.add_argument("--train-split", type=float, default=0.85,
                        help="Fraction for training (default 0.85)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(
        api_key=args.api_key,
        model=args.model,
        max_images=args.max_images,
        workers=args.workers,
        output_dir=args.output,
        resume=args.resume,
        train_split=args.train_split,
        seed=args.seed,
    )
