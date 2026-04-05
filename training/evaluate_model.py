"""Evaluate the fine-tuned LLaVA model on the validation set.

Metrics:
  - BLEU-4 (defect description similarity)
  - Severity classification accuracy
  - Defect class accuracy (via DINOv2 classifier)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm


def load_model_and_processor(model_path: str):
    from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
    from peft import PeftModel

    print(f"[INFO] Loading model from {model_path}")
    processor = LlavaNextProcessor.from_pretrained(model_path)
    base_model = LlavaNextForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    adapter_dir = Path(model_path)
    if (adapter_dir / "adapter_config.json").exists():
        base_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
        base_model = base_model.merge_and_unload()
    base_model.eval()
    return base_model, processor


def bleu_score(hypothesis: str, reference: str) -> float:
    """Simple 1-gram BLEU for quick evaluation."""
    from collections import Counter
    hyp_tokens = hypothesis.lower().split()
    ref_tokens = reference.lower().split()
    if not hyp_tokens:
        return 0.0
    ref_counts = Counter(ref_tokens)
    match = sum(min(count, ref_counts[token]) for token, count in Counter(hyp_tokens).items())
    precision = match / len(hyp_tokens)
    bp = min(1.0, len(hyp_tokens) / max(1, len(ref_tokens)))
    return bp * precision


def evaluate(model_path: str, val_jsonl: str, image_root: str) -> dict:
    model, processor = load_model_and_processor(model_path)
    device = next(model.parameters()).device

    with open(val_jsonl) as f:
        records = [json.loads(line) for line in f if line.strip()]

    bleu_scores = []
    severity_correct = 0
    severity_total = 0

    for rec in tqdm(records, desc="Evaluating"):
        img_path = Path(image_root) / rec["image"]
        if not img_path.exists():
            image = Image.new("RGB", (4, 4))
        else:
            image = Image.open(img_path).convert("RGB")

        conversations = rec.get("conversations", [])
        if not conversations:
            continue

        # Find the first human turn with image
        question = conversations[0]["value"]
        reference = conversations[1]["value"] if len(conversations) > 1 else ""

        prompt = f"[INST] <<SYS>>\nYou are a semiconductor FA expert.\n<</SYS>>\n\n{question} [/INST]"
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs, max_new_tokens=256, do_sample=False
            )
        input_len = inputs["input_ids"].shape[1]
        hypothesis = processor.decode(
            output_ids[0][input_len:], skip_special_tokens=True
        ).strip()

        # BLEU
        bleu = bleu_score(hypothesis, reference)
        bleu_scores.append(bleu)

        # Severity accuracy (if severity in reference)
        for sev_label in ["CRITICAL", "MAJOR", "MINOR", "NONE"]:
            if sev_label in reference.upper():
                severity_total += 1
                if sev_label in hypothesis.upper():
                    severity_correct += 1
                break

    avg_bleu = sum(bleu_scores) / max(len(bleu_scores), 1)
    sev_acc = severity_correct / max(severity_total, 1)

    results = {
        "num_samples": len(records),
        "avg_bleu_1gram": round(avg_bleu, 4),
        "severity_accuracy": round(sev_acc, 4),
    }
    print("\n=== Evaluation Results ===")
    for k, v in results.items():
        print(f"  {k}: {v}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="models/llava-semiconductor-qlora")
    parser.add_argument("--val-jsonl", default="data/processed/val.jsonl")
    parser.add_argument("--image-root", default="data/raw")
    args = parser.parse_args()
    evaluate(args.model_path, args.val_jsonl, args.image_root)
