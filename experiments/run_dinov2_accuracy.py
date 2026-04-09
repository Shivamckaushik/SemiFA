"""
DINOv2 classification accuracy experiment for the paper.

Trains a lightweight MLP head on DINOv2 embeddings using the synthetic +
WM-811K datasets, then reports per-class and overall accuracy.

This is Table 2 in the paper: "DINOv2 Defect Classification Accuracy".

Runs on CPU — no GPU needed (~5-10 min).

Usage:
    python experiments/run_dinov2_accuracy.py
    python experiments/run_dinov2_accuracy.py --epochs 30 --save-results
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


def train_test_split_manual(X, y, test_size=0.25, random_state=42):
    """Stratified train/test split without sklearn."""
    rng = np.random.default_rng(random_state)
    train_idx, test_idx = [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_test = max(1, int(len(idx) * test_size))
        test_idx.extend(idx[:n_test].tolist())
        train_idx.extend(idx[n_test:].tolist())
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

DEFECT_CLASSES = [
    "scratch", "particle_contamination", "edge_crack",
    "center_cluster", "local_cluster", "ring_pattern",
    "random_defects", "near_full_wafer", "no_defect",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(DEFECT_CLASSES)}
DINO_MODEL_ID = "facebook/dinov2-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Model ────────────────────────────────────────────────────────────────────

class DefectMLP(nn.Module):
    def __init__(self, num_classes: int = 9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_annotations(*jsonl_paths: str) -> list[dict]:
    records = []
    for p in jsonl_paths:
        path = Path(p)
        if not path.exists():
            print(f"  WARNING: {path} not found — skipping")
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    if rec.get("defect_class") in CLASS_TO_IDX:
                        records.append(rec)
    return records


def extract_embeddings(
    records: list[dict],
    processor,
    backbone,
    batch_size: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract DINOv2 CLS embeddings for all images."""
    embeddings, labels = [], []
    total = len(records)

    for start in range(0, total, batch_size):
        batch = records[start : start + batch_size]
        imgs = []
        valid_batch = []
        for rec in batch:
            try:
                img = Image.open(rec["image_path"]).convert("RGB")
                imgs.append(img)
                valid_batch.append(rec)
            except Exception as e:
                print(f"  Skip {rec['image_path']}: {e}")

        if not imgs:
            continue

        inputs = processor(images=imgs, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = backbone(**inputs)
        cls_embs = out.last_hidden_state[:, 0, :].cpu().numpy()

        for emb, rec in zip(cls_embs, valid_batch):
            embeddings.append(emb)
            labels.append(CLASS_TO_IDX[rec["defect_class"]])

        done = min(start + batch_size, total)
        print(f"  Embedded {done}/{total}", end="\r")

    print()
    return np.array(embeddings, dtype=np.float32), np.array(labels, dtype=np.int64)


# ── Training ──────────────────────────────────────────────────────────────────

def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 50,
    lr: float = 1e-3,
) -> DefectMLP:
    model = DefectMLP(num_classes=len(DEFECT_CLASSES)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    X = torch.tensor(X_train).to(DEVICE)
    y = torch.tensor(y_train).to(DEVICE)

    model.train()
    for epoch in range(1, epochs + 1):
        opt.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        opt.step()
        scheduler.step()
        if epoch % 10 == 0:
            with torch.no_grad():
                preds = model(X).argmax(dim=1)
                acc = (preds == y).float().mean().item()
            print(f"  Epoch {epoch:3d}/{epochs}  loss={loss.item():.4f}  train_acc={acc:.1%}")

    return model


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model: DefectMLP, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    model.eval()
    X = torch.tensor(X_test).to(DEVICE)
    with torch.no_grad():
        logits = model(X)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()

    correct = (preds == y_test).sum()
    total = len(y_test)
    overall_acc = correct / total

    # Per-class accuracy
    per_class: dict[str, dict] = {}
    for idx, cls in enumerate(DEFECT_CLASSES):
        mask = y_test == idx
        if mask.sum() == 0:
            continue
        cls_correct = (preds[mask] == y_test[mask]).sum()
        per_class[cls] = {
            "n_samples": int(mask.sum()),
            "correct": int(cls_correct),
            "accuracy": float(cls_correct / mask.sum()),
        }

    return {
        "overall_accuracy": float(overall_acc),
        "correct": int(correct),
        "total": int(total),
        "per_class": per_class,
        "predictions": preds.tolist(),
        "ground_truth": y_test.tolist(),
    }


def print_results(results: dict) -> None:
    print("\n" + "=" * 60)
    print("DINOV2 CLASSIFICATION ACCURACY — PAPER TABLE 2")
    print("=" * 60)
    print(f"Overall accuracy: {results['overall_accuracy']:.1%}  "
          f"({results['correct']}/{results['total']} correct)\n")

    print(f"{'Defect Class':<28} {'N':>4}  {'Accuracy':>10}")
    print("-" * 46)
    for cls in DEFECT_CLASSES:
        if cls not in results["per_class"]:
            continue
        pc = results["per_class"][cls]
        bar = "#" * int(pc["accuracy"] * 20)
        print(f"  {cls:<26} {pc['n_samples']:>4}  {pc['accuracy']:>8.1%}  {bar}")
    print("-" * 46)
    print(f"  {'OVERALL':<26} {results['total']:>4}  "
          f"{results['overall_accuracy']:>8.1%}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main(epochs: int = 50, save_results: bool = True) -> None:
    print(f"Device: {DEVICE}")
    print(f"Epochs: {epochs}\n")

    # ── Load datasets ──
    print("Loading annotations...")
    records = load_annotations(
        "data/synthetic_dataset/annotations.jsonl",
        "data/wm811k/annotations.jsonl",
    )
    print(f"  Total labeled images: {len(records)}")

    class_counts = defaultdict(int)
    for r in records:
        class_counts[r["defect_class"]] += 1
    for cls, cnt in sorted(class_counts.items()):
        print(f"    {cls:<28} {cnt}")

    if len(records) < 10:
        print("\nERROR: Not enough data. Run generate_synthetic_dataset.py first.")
        return

    # ── Load DINOv2 ──
    print(f"\nLoading {DINO_MODEL_ID}...")
    t0 = time.time()
    processor = AutoImageProcessor.from_pretrained(DINO_MODEL_ID)
    backbone = AutoModel.from_pretrained(DINO_MODEL_ID).to(DEVICE)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # ── Extract embeddings ──
    print("\nExtracting DINOv2 embeddings...")
    t1 = time.time()
    X, y = extract_embeddings(records, processor, backbone)
    print(f"  {len(X)} embeddings extracted in {time.time()-t1:.1f}s")
    print(f"  Embedding shape: {X.shape}")

    # ── Train/test split ──
    X_train, X_test, y_train, y_test = train_test_split_manual(
        X, y, test_size=0.25, random_state=42
    )
    print(f"\nTrain: {len(X_train)}  |  Test: {len(X_test)}")

    # ── Zero-shot baseline (random MLP) ──
    print("\n[Baseline] Random-weight MLP (zero-shot):")
    baseline_model = DefectMLP().to(DEVICE)
    baseline_model.eval()
    baseline_results = evaluate(baseline_model, X_test, y_test)
    print(f"  Zero-shot accuracy: {baseline_results['overall_accuracy']:.1%} "
          f"(expected ~{1/len(DEFECT_CLASSES):.1%} random)")

    # ── Train MLP head ──
    print(f"\nTraining MLP head ({epochs} epochs)...")
    t2 = time.time()
    trained_model = train_mlp(X_train, y_train, epochs=epochs)
    print(f"  Training completed in {time.time()-t2:.1f}s")

    # ── Evaluate ──
    print("\nEvaluating on test set...")
    results = evaluate(trained_model, X_test, y_test)
    results["baseline_accuracy"] = baseline_results["overall_accuracy"]
    results["n_train"] = len(X_train)
    results["n_test"] = len(X_test)
    results["epochs"] = epochs
    results["model"] = DINO_MODEL_ID

    print_results(results)

    # ── Save results ──
    if save_results:
        out_path = Path("experiments/dinov2_accuracy_results.json")
        out_path.parent.mkdir(exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {out_path}")

    print("\nFor the paper:")
    print(f"  'DINOv2-base + MLP achieves {results['overall_accuracy']:.1%} accuracy")
    print(f"   on {results['total']} test images across 9 defect classes,")
    print(f"   vs {results['baseline_accuracy']:.1%} zero-shot baseline.'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--save-results", action="store_true", default=True)
    args = parser.parse_args()
    main(epochs=args.epochs, save_results=args.save_results)
